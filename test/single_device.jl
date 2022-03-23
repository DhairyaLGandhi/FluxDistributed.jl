model = ResNet34();

# First test the controlled case with a single GPU
# and deterministic data

function check_data_parallel(m, data = rand(Float32, 224,224,3,3); field = :weight)
  gpu_model, gpu_data = gpu(m), gpu(data)
  gpu_model2 = deepcopy(gpu_model)
  
  batchedgrads = gradient(gpu_model) do model
    sum(model(gpu_data))
  end

  colons = ntuple(_ -> Colon(), ndims(gpu_data) - 1)
  distributedgrads_x3 = map(1:3) do i
    gradient(deepcopy(gpu_model2)) do model
      sum(model(gpu_data[colons...,i:i]))
    end
  end

  final = reduce(distributedgrads_x3[2:end], init = distributedgrads_x3[1]) do x, y
    Functors.fmap(x, y) do x, y
      ResNetImageNet._accum(x,y)
    end
  end

  @testset "$(typeof(m))" begin
    @testset "Check accuulating grads using _accum" begin
      compare(final, batchedgrads)
    end

    @testset "Manually accumulating grads against batched" begin
      compare(get_sum(getfirst.(distributedgrads_x3, field)), getfirst(batchedgrads, field))
    end
  end
end

get_sum(::AbstractVector{<:Nothing}) = nothing
get_sum(x) = +(x...)

# Check gradients are collected correctly for single layers
@testset "Distributed Gradient accumulation" begin
  check_data_parallel(Conv((7,7), 3 => 4))
  check_data_parallel(Dense(10, 3), rand(Float32, 10, 3))
  check_data_parallel(Chain(Dense(10,5), Dense(5, 3)), rand(Float32, 10, 3))
  check_data_parallel(MaxPool((3,3)), rand(Float32, 3,3,3,3))
  for field in (:λ, :β, :γ)
    check_data_parallel(BatchNorm(3), rand(Float32, 3,3,3,3), field = field)
  end

  # norm layers dont work on gpus with `track_stats = false`
  # which seems arbitrary considering we can choose to not update the params
  # on the Functor side
  # Making the model testmode! will allow it to run anyway :/
  # the consequence being that we unwantingly disable dropout
  # on real models
  check_data_parallel(Flux.testmode!(Chain(Conv((7,7), 3 => 3), BatchNorm(3))), rand(Float32, 32,32,3,3), field = :weight)
  check_data_parallel(Flux.testmode!(Chain(Conv((7,7), 3 => 3), BatchNorm(3))), field = :weight)
  check_data_parallel(Flux.testmode!(ResNet34()), field = :weight)
  # the no-testmode! version will fail several tests
  # check_data_parallel(ResNet34(), field = :weight)
end

# manually adding the first weight element from the first layer for every image independently

function test_grad_syncing_in_train(loss, m, nt, buffer, opt,
                                    data = rand(Float32, 224,224,3,3),
                                    labels = Flux.onehotbatch(rand(1:10, 3), 10))
  ds_and_ms = nt.ds_and_ms
  sts = nt.sts

  # NOTE: Do not use nt.dls here since we want to isolate the various parts as we test
  # them, and the dataloading is a separate component that needs to be tested

  gpu_model, gpu_data, gpu_labels = gpu(m), gpu(data), gpu(labels)
  gpu_model2 = deepcopy(gpu_model)

  batchedgrads, = gradient(gpu_model) do model
    loss(model(gpu_data), gpu_labels)
  end

  colons_data = ntuple(_ -> Colon(), ndims(gpu_data) - 1)
  colons_labels = ntuple(_ -> Colon(), ndims(gpu_labels) - 1)
  ts = []
  for ((dev,m), j) in zip(ds_and_ms, 1:size(data, ndims(data)))
    x = gpu_data[colons_data..., j:j]
    y = gpu_labels[colons_labels..., j:j]
    gs = Threads.@spawn begin
      train_step_cpu(loss, buffer, dev, m, x, y)
    end
    push!(ts, Base.errormonitor(gs))
  end
  gs = ts
  wait.(gs)
  final = ResNetImageNet.sync_buffer(buffer)
  final, batchedgrads
end

# Copy the train_step function minus the CUDA.device! which will error w/o a GPU
# TODO: add `@device! dev ex` which checks if CUDA.functional CUDA.device!(dev) do ex() end else ex end
function train_step_cpu(loss, buffer, dev, m, x, y)
  gs, = gradient(m -> loss(m(x), y), m)
  ResNetImageNet.markbuffer!(buffer[dev], gs, dev)
  gs
end

function train_step_cpu(loss, buffer, dev::Int, m, x, y)
  gs, = gradient(m -> loss(m(x), y), m)
  ResNetImageNet.markbuffer!(buffer[dev], gs, dev)
  gs
end

function check_distributed_opt(opt, ds_and_ms, buffer, gs, sts)
  new_ms = []
  new_ds_and_ms = map(ds_and_ms) do dnm
    # dnm = dev, m
    dev, m = dnm
    g = buffer[dev]
    t_opt = Threads.@spawn begin
      m, st = ResNetImageNet.update(opt, dnm, g, gs, sts[dev])
      sts[dev] = st
      m
    end
    (dev, fetch(Base.errormonitor(t_opt)))
  end
  map(x -> x[2], new_ds_and_ms), sts
end

@testset "Workflow" begin
  loss = Flux.Losses.logitcrossentropy
  data = rand(Float32, 32, 32, 3, 3)
  labels = Flux.onehotbatch(rand(1:10, 3), 1:10)
  m = Chain(Conv((7,7), 3 => 3), Flux.flatten, Dense(2028, 10))
  opt = ResNetImageNet.Optimisers.Momentum()
  nt, buffer = if CUDA.functional()
    classes = 1:1000
    key = open(BlobTree, DataSets.dataset("imagenet_cyclops")) do data_tree
      ResNetImageNet.train_solutions(data_tree, path"LOC_train_solution.csv", classes)
    end

    if length(CUDA.devices()) == 1
      zmodel = ResNetImageNet.destruct(m)
      st = ResNetImageNet.Optimisers.state(opt, m)
      buffer = Dict(i => deepcopy(gpu(zmodel)) for i = 1:size(data, ndims(data)))
      ds_and_ms = ntuple(i -> (i, deepcopy(gpu(m))), size(data, ndims(data)))
      sts = Dict(i => deepcopy(gpu(st)) for i = 1:size(data, ndims(data)))
      (sts = sts, ds_and_ms = ds_and_ms), buffer
    else
      @warn "in the many gpu case"
      nt, buffer = prepare_training(m, key,
                                  CUDA.devices(),
                                  opt,
                                  1,
                                  cycle = 1,
                                  buffersize = 1)
    end
  else
    @warn "in the no gpu case"
    ds_and_ms = ntuple(i -> (i, deepcopy(m)), size(data, ndims(data)))
    zmodel = ResNetImageNet.destruct(m)
    st = ResNetImageNet.Optimisers.state(opt, m)
    buffer = Dict(i => deepcopy(zmodel) for i = 1:size(data, ndims(data)))
    sts = Dict(i => deepcopy(st) for i = 1:size(data, ndims(data)))
    (sts = sts, ds_and_ms = ds_and_ms), buffer
  end

  distributedgrad, batchedgrad = test_grad_syncing_in_train(loss, m, nt, buffer, opt, data, labels)
  compare(distributedgrad, batchedgrad)


  # Check distrbiuted optimization
  batchedmodel, batchedstate = opt(m, batchedgrad, st)
  distributedmodels, distributedstates = check_distributed_opt(opt, nt.ds_and_ms, buffer, distributedgrad, nt.sts)
  compare(batchedmodel, distributedmodels[1])
  compare(distributedmodel[2], distributedmodel[3])
  compare(distributedstates[1], batchedstates) 
end

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

  batchedgrads = gradient(gpu_model) do model
    loss(model(gpu_data), gpu_labels)
  end

  colons = ntuple(_ -> Colon(), ndims(gpu_data) - 1)
  ts = []
  # for j in 1:size(data, ndims(data))
  for ((dev,m), j) in zip(ds_and_ms, 1:size(data, ndims(data)))
    x = gpu_data[colons..., j:j]
    y = gpu_labels[colons..., j:j]
    # m = deepcopy(gpu_model2)
    gs = Threads.@spawn train_step(loss, buffer, dev, m, x, y)
    push!(ts, Base.errormonitor(gs))
  end
  gs = ts
  wait.(gs)
  final = sync_buffer(buffer, dodiv = false)
  compare(final, batchedgrads)
end


@testset "Workflow" begin
  loss = Flux.Losses.logitcrossentropy
  data = rand(Float32, 32, 32, 3, 3)
  labels = Flux.onehotbatch(rand(1:10, 3), 1:10)
  m = Chain(Conv((7,7), 3 => 3), Flux.flatten, Dense(2028, 3))
  opt = ResNetImageNet.Optimisers.Momentum()
  nt, buffer = if CUDA.functional()
    classes = 1:1000

    key = open(BlobTree, DataSets.dataset("imagenet_cyclops")) do data_tree
      ResNetImageNet.train_solutions(data_tree, path"LOC_train_solution.csv", classes)
    end

    nt, buffer = prepare_training(m, key,
                                  CUDA.devices(),
                                  opt,
                                  1,
                                  cycle = 1,
                                  buffersize = 1)
  else
    ds_and_ms = ntuple(i -> (i, deepcopy(m)), 1:size(data, ndims(data)))
    zmodel = ResNetImageNet.destruct(m)
    st = ResNetImageNet.Optimisers.state(opt, m)
    buffer = Dict(i => deepcopy(zmodel) for i = 1:size(data, ndims(data)))
    sts = Dict(i => deepcopy(st) for i = 1:size(data, ndims(data)))
    (sts = sts, ds_and_ms = ds_and_ms), buffer
  end
end

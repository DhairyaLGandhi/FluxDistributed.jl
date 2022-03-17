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
end

# manually adding the first weight element from the first layer for every image independently


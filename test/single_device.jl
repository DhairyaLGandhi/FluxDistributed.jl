model = ResNet34();

# First test the controlled case with a single GPU
# and deterministic data

function check_data_parallel(m, data = rand(Float32, 224,224,3,3))
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
      compare(get_sum(distributedgrads_x3), batchedgrads[1].weight)
    end
  end
end

function get_sum(x)
  x[1][1].weight + x[2][1].weight + x[3][1].weight
end

# Check gradients are collected correctly for single layers
@testset "Distributed Gradient accumulation" begin
  check_data_parallel(Conv((7,7), 3 => 4))
  check_data_parallel(Dense(10, 3), rand(Float32, 10, 3))
end

# manually adding the first weight element from the first layer for every image independently


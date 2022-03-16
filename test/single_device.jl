model = ResNet(34);

# First test the controlled case with a single GPU
# and deterministic data
data = rand(Float32, 224, 224, 3, 3);

gpu_model, gpu_data = CUDA.device!(CUDA.device()) do
  gpu(model), gpu(data)
end

gpu_model2 = deepcopy(gpu_model);

gpu_grads = gradient(gpu_model) do model
  sum(model(gpu_data))
end

gpu_grads_x3 = map(1:3) do i
  gradient(deepcopy(gpu_model2)) do model
    sum(model(gpu_data[:,:,:,i:i]))
  end
end

final = reduce(gpu_grads_x3[2:end], init = gpu_grads_x3[1]) do x, y
  Functors.fmap(x, y) do x, y
    ResNetImageNet._accum(x,y)
  end
end

# manually adding the first weight element from the first layer for every image independently
function get_sum(gpu_grads_x3)
  gpu_grads_x3[1][1].layers[1][1].layers[1].weight + gpu_grads_x3[2][1].layers[1][1].layers[1].weight + gpu_grads_x3[3][1].layers[1][1].layers[1].weight
end

@test "Check grads added correctly using `ResNetImageNet._accum`" get_sum(gpu_grads_x3) ≈ final[1].layers[1][1].layers[1].weight
@test "Check manually added grads against batched grads" get_sum(gpu_grads_x3) ≈ gpu_grads[1].layers[1][1].layers[1].weight
# compare(final, gpu_grads)

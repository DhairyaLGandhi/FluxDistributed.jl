model = ResNet(34)

# First test the controlled case with a single GPU
# and deterministic data
data = rand(Float32, 224, 224, 3, 3)

gpu_model, gpu_data = CUDA.device!(CUDA.device()) do
  gpu(model), gpu(data)
end

gpu_grads = gradient(deepcopy(gpu_model)) do model
  sum(model(gpu_data))
end

gpu_grads_x3 = map(1:3) do i
  gradient(gpu_model) do model
    sum(model(gpu_data[:,:,:,i:i]))
  end
end

final = reduce(gpu_grads_x3[2:end], init = gpu_grads_x3[1]) do x, y
  Functors.fmap(x, y) do x, y
    ResNetImageNet._accum(x,y)
  end
end

compare(final, gpu_grads)

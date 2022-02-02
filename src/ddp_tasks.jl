# t = map(0:3) do i
#   (500, CUDA.CuDevice(i))
# end
# 
# dl = Flux.Data.DataLoader((t,))

# function sth(model, key, devices, nsamples)
#   t = map(devices, nsamples) do dev, n
#     (dev, n)
#   end |> x -> (x...,)
# 
#   dl = Flux.Data.DataLoader((t,)) do (dev, n)
#     Threads.@spawn begin
#       data = minibatch(nothing, key,
#                        nsamples = n,
#                        class_idx = 1:1000)
#       dl2 = Flux.Data.DataLoader(data,
#                                  batchsize = 48) do x
#         CUDA.device!(dev) do
#           @show CUDA.device()
#           gpu(x)
#         end
#       end
#     end
#   end
# end

export prepare_training, train_ddp

function prepare_training(resnet, key, devices, nsamples)
  data = minibatch(nothing, key, nsamples = nsamples, class_idx = 1:1000)
  shards = Flux.Data.DataLoader(data, batchsize = nsamples ÷ length(devices))
  ds = Dict()
  @sync for (dev, shard) in zip(devices, shards)
    Threads.@spawn begin
      ds[dev] = CUDA.device!(dev) do
        gpu(resnet),
        Flux.Data.DataLoader(gpu, shard, batchsize = 48)
      end
    end
  end
  ds
end

function train(loss, dev, m, dl)
  CUDA.device!(dev) do
    @info "going to forward" CUDA.device()
    for d in dl
      x, y = d
      # gradient(m) do m
      #   loss(m(x), y)
      # end
      # @info "going to forward" CUDA.device() typeof(d)
      # @show loss(m(x), y)
      @show size(x), size(y)
    end
  end
end

function train_ddp(loss, ds)
  res = []
  for (dev, (m, d)) in pairs(ds)
    t = Threads.@spawn begin
      train(loss, dev, m, d)
    end
    push!(res, t)
  end
  wait.(res)
end

function train(loss, m, dl, opt, dev, (ip, op))
  # STEP 1: Take data from dataloader: this data is on `dev`
  for (x,y) in dl
    # STEP 2: Take gradient on the appropriate dev: use model on `dev`; get grads on `dev`
    gm, = CUDA.device!(dev) do
       gradient(m) do m
         loss(m(x), y)
       end
    end

    # STEP 3: Copy gradients into a buffer on a GPU meant to reduce grads
    # `ip` is a buffer associated with this `dev`; n devices = n buffers
    gm = Functors.fmap(ip, gm) do x, y
      copyto!(x, y)
      y
    end

    # STEP 4: `op` holds the reduced grads from every device, copy them back into this `dev`
    # This happens since `x` is on this `dev` and CUDA does appropriate DtoD
    gmnew = Functors.fmap(gm, op) do x, y
      copyto!(x, y)
    end

    # STEP 5: Optimise. `m`, `gmnew` and `state` are all on `dev` and should return new `m`
    m, state = opt(m, gmnew, state)
  end
end

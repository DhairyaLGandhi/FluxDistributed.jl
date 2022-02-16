# function sth(model, key, devices, nsamples)
#   t = map(devices, nsamples) do dev, n
#     (dev, n)
#   end |> x -> (x...,)
# 
#   @show t
#   dl = Flux.Data.DataLoader((t,)) do (dev, n)
#     @show n, dev
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


_zero(x::AbstractArray) = zero(x)
_zero(x::Base.RefValue) = Ref(_zero(x[]))
_zero(x) = x


maybeT(x::NamedTuple) = x
maybeT(x) = (x,)

mywalk(f, x::T) where T = begin
  fs = fieldnames(T)
  if fs isa NTuple{N,Int} where N
    return map(f, Functors.children(x))
  end
  NamedTuple{fs}(maybeT(map(f, Functors.children(x))))
end

function destruct(o::T) where T
  Functors.fmapstructure(o, walk = mywalk) do x
    _zero(x)
  end
end

function prepare_training(resnet, key, devices, opt, nsamples; HOST = CUDA.CuDevice(0))
  ds = Dict()
  ixs = map(shuffle!, collect.(collect(Iterators.partition(1:size(key, 1), size(key, 1) ÷ length(devices)))))[1:length(devices)]
  ks = map(x -> @view(key[x,:]), ixs)
  ns = Iterators.repeated(nsamples, 25) # ntuple(_ -> nsamples, 5) # length(devices))
  buffer = Dict()
  zmodel = destruct(resnet)
  st = Optimisers.init(opt, resnet)
  for dev in devices
    Threads.@spawn begin
      buffer[dev] = CUDA.device!(HOST) do
        gpu(zmodel), gpu(st)
      end
    end
  end
  dls = []
  devs_and_ms = []
  for (k,dev) in zip(ks, devices)
    CUDA.device!(dev) do
      push!(devs_and_ms, (dev, gpu(resnet)))
      dl = Flux.Data.DataLoader((ns,), buffersize = 5) do x
        shard = minibatch(nothing, k, nsamples = x, class_idx = 1:1000)
        CUDA.device!(dev) do
          gpu(shard)
        end
      end
      push!(dls, dl)
    end
  end
  (ds_and_ms = devs_and_ms, dls = dls), buffer
end


# function train(loss, dev, m, dl)
#   CUDA.device!(dev) do
#     @info "going to forward" CUDA.device()
#     for d in dl
#       x, y = d
#       # gradient(m) do m
#       #   loss(m(x), y)
#       # end
#       # @info "going to forward" CUDA.device() typeof(d)
#       # @show loss(m(x), y)
#       @show size(x), size(y)
#     end
#   end
# end
# 
# function train_ddp(loss, ds)
#   res = []
#   for (dev, (m, d)) in pairs(ds)
#     t = Threads.@spawn begin
#       # CUDA.device!(dev) do
#       #   @show CUDA.device()
#         train(loss, dev, m, d)
#       # end
#     end
#     push!(res, t)
#   end
#   @show "ended stuff"
#   wait.(res)
# end

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
      x
    end

    # STEP 5: Optimise. `m`, `gmnew` and `state` are all on `dev` and should return new `m`
    m, state = opt(m, gmnew, state)
  end
end

_copyto!(::Nothing, ::Nothing) = nothing
_copyto!(x::Base.RefValue, y::Base.RefValue) = Ref(_copyto!(x[], yp[]))
_copyto!(x, y) = copyto!(x, y)
_copyto!(x, ::Nothing) = nothing
_copyto!(::Nothing, x) = nothing

function markbuffer!(dest, src, dev)
  Functors.fmap(src, dest) do x, y
    _copyto!(y, x)
    x
  end
end

function getbuffer!(dest, src, dev)
  Functors.fmap(dest, src) do x, y
    _copyto!(x, y)
    x
  end
end

function sync_buffer(buffer)
  vals = map(first, values(buffer))
  # vals = collect(values(buffer))
  final = reduce(vals[2:end], init = vals[1]) do x,y
    Functors.fmap(x, y) do x, y
       isnothing(x) && return y
       isnothing(y) && return x
       _accum(x,y)
    end
  end
end

function step(buffer, dev, m, x, y)
  gs, = CUDA.device!(dev) do
    gradient(m -> ResNetImageNet.loss(m(x), y), m)
  end
  markbuffer!(first(buffer[dev]), gs, dev)
  # markbuffer!(buffer[dev], gs, dev)
  gs
end

# implementation of the `train` function from above
# with actual semantics baked in
# intended to be moved to the train method as pieces finish
function train(setup, buffer, HOST = CUDA.CuDevice(0))
  dls = nt.dls
  ds_and_ms = nt.ds_and_ms
  big_batches = zip(dls...)
  for mbs in big_batches
    t = Threads.@spawn begin
      t_ = map(ds_and_ms, mbs) do (dev, m), bs
        @info "results:" dev=dev host=HOST
        gs = Threads.@spawn step(buffer, dev, m, bs...)
      end
    end
    gs = fetch(t)
    wait.(gs)
    final = sync_buffer(buffer)
    get_tasks = map(ds_and_ms, gs) do dnm, g
     dev, m = dnm
     get_task = Threads.@spawn getbuffer!(fetch(g), final, dev)
    end
    ts = fetch.(get_tasks)
  end
end

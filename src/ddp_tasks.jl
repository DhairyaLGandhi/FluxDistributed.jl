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

function prepare_training(resnet, key, devices, nsamples; HOST = CUDA.CuDevice(0))
  ds = Dict()
  ixs = map(shuffle!, collect.(collect(Iterators.partition(1:size(key, 1), size(key, 1) ÷ length(devices)))))[1:length(devices)]
  ks = map(x -> @view(key[x,:]), ixs)
  # @sync for (dev, shard) in zip(devices, shards)
  ns = ntuple(_ -> nsamples, 20) # length(devices))
  buffer = Dict()
  zmodel = destruct(resnet)
  for dev in devices
    Threads.@spawn begin
      buffer[dev] = CUDA.device!(HOST) do
        gpu(zmodel)
      end
    end
  end
  for (k,dev) in zip(ks, devices)
    Threads.@spawn begin
      ds[dev] = CUDA.device!(dev) do
        gpu(resnet),
        Flux.Data.DataLoader((ns,), buffersize = 5) do x
          shard = minibatch(nothing, k, nsamples = x, class_idx = 1:1000)
          CUDA.device!(dev) do
            gpu(shard)
          end
        end
      end
    end
  end
  ds, buffer
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
  while true
    if check(MARKER, dev, writable)
      break
    end
  end
  Functors.fmap(src, dest) do x, y
    _copyto!(y, x)
    x
  end
  mark!(MARKER, dev, readable)
end

function check(MARKER, dev, status)
  MARKER[dev][] == status
end

"""
    mark!(MARKER, dev)

Mark that dev is changing state in MARKER

MARKER is a Dict with keys as all the devices and the values being 2-tuples of atomic booleans which represent (hot gradient, can be overwritten)
"""
function mark!(MARKER, dev, status)
  Threads.atomic_xchg!(MARKER[dev], status)  
end

function getbuffer!(dest, src, dev)
  while true
    if check(MARKER, dev, readable)
      break
    end
  end
  Functors.fmap(dest, src) do x, y
    _copyto!(x, y)
    x
  end
  mark!(MARKER, dev, writable)
end

const writable = true
const readable = false
const MARKER = Dict(dev => Threads.Atomic{Bool}(writable) for dev in CUDA.devices())

function reset!(MARKER)
  for k in keys(MARKER)
    mark!(MARKER, k, writable)
  end
end

# implementation of the `train` function from above
# with actual semantics baked in
# intended to be moved to the train method as pieces finish
function train(setup, buffer, HOST = CUDA.CuDevice(0))
  ts = []
  reset!(MARKER)
  for (dev,(m,dl)) in pairs(setup)
    t = Threads.@spawn begin
      for (x,y) in dl # STEP 1
    
        gm, = CUDA.device!(dev) do # STEP 2
          gradient(m -> ResNetImageNet.loss(m(x), y), m)
        end
        markbuffer!(buffer[dev], gm, dev) # STEP 3
        getbuffer!(gm, buffer[dev], dev) # STEP 4
        m, st = opt(m, gm, st) # STEP 5
        @info "results:" dev=dev nt=isa(gs, NamedTuple) s=sum(m(x)) t=(typeof(x), typeof(y)) host=HOST
      end
    end
    push!(ts, t)
  end
  ts
end
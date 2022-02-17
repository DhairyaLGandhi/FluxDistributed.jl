using Serialization, Statistics, Random
using .Threads

_zero(x::AbstractArray) = zero(x)
_zero(x::Base.RefValue) = Ref(_zero(x[]))
_zero(x::Function) = nothing
_zero(x::T) where T <: Union{MaxPool, AdaptiveMeanPool, Flux.Zeros} = nothing
_zero(x) = x
_zero(x::Real) = nothing

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
_copyto!(x::Function, y::Function) = nothing # x
_copyto!(x::T, y::S) where {T <: Real, S <: Real} = convert(T, y)
_copyto!(x::T, y::T) where T <: Union{MaxPool, AdaptiveMeanPool, Flux.Zeros} = y
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

function train_step(loss, buffer, dev, m, x, y)
  gs, = CUDA.device!(dev) do
    gradient(m -> loss(m(x), y), m)
  end
  markbuffer!(buffer[dev], gs, dev)
  gs
end

function sync_buffer(buffer)
  vals = collect(values(buffer))
  final = reduce(vals[2:end], init = vals[1]) do x,y
    Functors.fmap(x, y) do x, y
       isnothing(x) && return y
       isnothing(y) && return x
       ResNetImageNet._accum(x,y)
    end
  end
  final = Functors.fmap(final) do x
    isnothing(x) && return x
    ResNetImageNet._dodiv(x, Float32(length(vals)))
  end
  # once you have the final grads - broadcast them (copy them to every entry in buffer)
  # note that the grads never leave the HOST, move them to every GPU in the optimisation
  # step
  get_tasks = Threads.@spawn begin
    ts = []
    for (dev,g) in pairs(buffer)
      get_task = Threads.@spawn getbuffer!(g, final, dev)
      push!(ts, get_task)
    end
    ts
  end
  ft = fetch(get_tasks)
  wait.(ft)
  final 
end

log_loss_and_acc(loss, (dev, model), val::Nothing) = nothing
function log_loss_and_acc(loss, (dev, model), val; k = (1,5,10))
  l, fw = CUDA.device!(dev) do
    gval = gpu(val)
    loss(model(gval[1]), gval[2]), cpu(model(gval[1]))
  end
  acc = map(k -> topkaccuracy(fw, val[2]; k = k), k)
  @info "val_loss" loss=l
  for (j,a) in zip(k, acc)
    @info "acc_$j" acc=a
  end
end

loss(x, y) = -sum(y .* Flux.logsoftmax(x) ) ./ Float32(size(y,2))

function train(loss, nt, buffer, opt; val = nothing)
  dls = nt.dls
  ds_and_ms = nt.ds_and_ms
  sts = nt.sts
  big_batches = zip(dls...)
  for (j,mbs) in enumerate(big_batches)
    if j % 10 == 0
      println("Cycle: $j")
      log_loss_and_acc(loss, first(ds_and_ms), val)
    end
    t = Threads.@spawn begin
      t_ = map(ds_and_ms, mbs) do (dev, m), bs
        gs = Threads.@spawn train_step(loss, buffer, dev, m, bs...)
      end
    end
    gs = fetch(t)
    wait.(gs)
    final = sync_buffer(buffer)

    # move final grads to every GPU - fetch(g) has the right
    # grads for dev in it, overwrite with final
    # and optimise
    get_tasks = map(ds_and_ms, gs) do dnm, g
      t = Threads.@spawn begin
        dev, m = dnm

        t_ = Threads.@spawn begin
          getbuffer!(fetch(g), final, dev)
        end

        t_opt = Threads.@spawn begin
          m, st = CUDA.device!(dev) do
            opt(m, fetch(t_), sts[dev])
          end
          sts[dev] = st
          m
        end

        (dev, fetch(t_opt))
      end
    end
    ds_and_ms = fetch.(get_tasks)
  end
  map(ds_and_ms) do dnm
    dev, m = dnm
    CUDA.device!(dev) do
      dev, cpu(m)
    end
  end
end

function prepare_training(resnet, key, devices, opt, nsamples;
	                   HOST = CUDA.CuDevice(0),
			   cycle = 5,
			   classes = 1:1000,
			   buffersize = 5)
  ds = Dict()
  ixs = map(shuffle!, collect.(collect(Iterators.partition(1:size(key, 1), size(key, 1) ÷ length(devices)))))[1:length(devices)]
  ks = map(x -> @view(key[x,:]), ixs)
  ns = Iterators.repeated(nsamples, cycle) # ntuple(_ -> nsamples, 5)
  buffer = Dict()
  zmodel = destruct(resnet)
  st = ResNetImageNet.Optimisers.state(opt, resnet)
  for dev in devices
    Threads.@spawn begin
      buffer[dev] = CUDA.device!(HOST) do
        gpu(zmodel)
      end
    end
  end
  dls = []
  devs_and_ms = []
  sts = Dict()
  for (k,dev) in zip(ks, devices)
    CUDA.device!(dev) do
      push!(devs_and_ms, (dev, gpu(resnet)))
      sts[dev] = gpu(st)
      dl = Flux.Data.DataLoader((ns,), buffersize = buffersize) do x
        shard = minibatch(nothing, k, nsamples = x, class_idx = classes)
        CUDA.device!(dev) do
          gpu(shard)
        end
      end
      push!(dls, dl)
    end
  end
  (ds_and_ms = devs_and_ms, dls = dls, sts = sts), buffer
end

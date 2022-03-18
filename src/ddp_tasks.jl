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

loss(x, y) = -sum(y .* Flux.logsoftmax(x) ) ./ Float32(size(y,2))
# function train(loss, m, dl, opt, dev, (ip, op))
#   # STEP 1: Take data from dataloader: this data is on `dev`
#   for (x,y) in dl
#     # STEP 2: Take gradient on the appropriate dev: use model on `dev`; get grads on `dev`
#     gm, = CUDA.device!(dev) do
#        gradient(m) do m
#          loss(m(x), y)
#        end
#     end
# 
#     # STEP 3: Copy gradients into a buffer on a GPU meant to reduce grads
#     # `ip` is a buffer associated with this `dev`; n devices = n buffers
#     gm = Functors.fmap(ip, gm) do x, y
#       copyto!(x, y)
#       y
#     end
# 
#     # STEP 4: `op` holds the reduced grads from every device, copy them back into this `dev`
#     # This happens since `x` is on this `dev` and CUDA does appropriate DtoD
#     gmnew = Functors.fmap(gm, op) do x, y
#       copyto!(x, y)
#       x
#     end
# 
#     # STEP 5: Optimise. `m`, `gmnew` and `state` are all on `dev` and should return new `m`
#     m, state = opt(m, gmnew, state)
#   end
# end


_copyto!(::Nothing, ::Nothing) = nothing
_copyto!(x::Base.RefValue, y::Base.RefValue) = Ref(_copyto!(x[], y[]))
_copyto!(x, y) = copyto!(x, y)
_copyto!(x, ::Nothing) = nothing
_copyto!(::Nothing, x) = nothing
_copyto!(x::Function, y::Function) = nothing # x
function markbuffer!(dest, src, dev)
  Functors.fmap(src, dest) do x, y
    _copyto!(y, x)
    x
  end
  synchronize()
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

function check_nans(nt::Union{Tuple,NamedTuple})
  any(check_nans, nt)
end
check_nans(x::AbstractArray) = any(isnan, x)
check_nans(::Nothing) = false
check_nans(x) = isnan(x)

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

  final 
end

_isapprox(x::AbstractArray, y::AbstractArray) = x ≈ y
_isapprox(::Nothing, ::Nothing) = true
# _isapprox(x::Base.RefValue, y::Base.RefValue) = _isapprox(x[], y[])
# _isapprox(x, y) = true
function ensure_synced(buffer, final)
  a = Ref{Bool}(true)
  for (dev, g) in pairs(buffer)
    Functors.fmap(g, final) do x, y
      if !_isapprox(x, y)
        a[] = false
      end
      x
    end
  end
  a[]
end

log_loss_and_acc(loss, (dev, model), val::Nothing, dataset = "val") = nothing
function log_loss_and_acc(loss, (dev, model), val, dataset = "val"; k = (1,5,10))
  l, fw = CUDA.device!(dev) do
    gval = gpu(val)
    loss(model(gval[1]), gval[2]), cpu(model(gval[1]))
  end
  println("$(dataset)_loss: $l")
  acc = map(k -> topkaccuracy(softmax(fw), val[2]; k = k), k)
  @info "$(dataset)_$(dev)" loss=l # dead=sum(model[end].weight)
  for (j,a) in zip(k, acc)
    @info "$(dataset)_$(j)_$(dev)" acc=a
  end
end

function log_loss_and_acc(loss, dnm, val::AbstractDataFrame, dataset = "val"; kw...)

  v = open(BlobTree, DataSets.dataset("imagenet_local")) do data_tree
    minibatch(data_tree, val, class_idx = sort(unique(val[!, :class_idx])), nsamples = 300, dataset = dataset)
  end
  log_loss_and_acc(loss, dnm, v; kw...)
end

# function log_loss_and_acc(loss, dnm::AbstractVector, val::AbstractDataFrame, dataset = "val"; kw...)
#   for (dev, m) in dnm
#     v = minibatch(nothing, val, class_idx = sort(unique(val[!, :class_idx])), nsamples = 300, dataset = dataset)
#     log_loss_and_acc(loss, (dev,m), v, dataset; kw...)
#   end
# end
# 
# function log_loss_and_acc(loss, dnm::AbstractVector, val, dataset = "val"; kw...)
#   for (dev, m) in dnm
#     log_loss_and_acc(loss, (dev,m), val, dataset; kw...)
#   end
# end

function update(opt, (dev,m), g, final, st)
  m, st = CUDA.device!(dev) do
    grad = fetch(g)
    getbuffer!(grad, final, dev)
    synchronize()
    m, st = opt(m, grad, st)
    synchronize()
    m, st
  end
end

function train(loss, nt, buffer, opt; val = nothing, sched = identity)
  dls = nt.dls
  ds_and_ms = nt.ds_and_ms
  sts = nt.sts
  big_batches = zip(dls...)
  num_missed = 0

  # Step 1: get the minibatches from every dataloader
  # Make sure not to use these without a `device!` block
  for (j, mbs) in enumerate(big_batches)
    try
      if j % 10 == 0
        println("Cycle: $j")
        if j % 50 == 0
          log_loss_and_acc(loss, first(ds_and_ms), val, "val")
          log_loss_and_acc(loss, first(ds_and_ms), cpu(first(mbs)), "train")
        end
      end

      # if j > 2000 && j % 1200 == 0
      #   opt = sched(opt)
      #   Wandb.update_config!(lg, Dict("eta" => opt.eta), allow_val_change = true)
      # end

      # Step 2: Get the grads on every GPU
      # `train_step` returns the grads allocated on every
      # dev. fetch(g) should return the memory on the GPU
      # reuse this memory in Step 4.
      ts = []
      for ((dev,m), bs) in zip(ds_and_ms, mbs)
        gs = Threads.@spawn train_step(loss, buffer, dev, m, bs...)
        push!(ts, Base.errormonitor(gs))
      end
      gs = ts
      wait.(gs)

      # Step 3: Sync the buffer gradients
      final = sync_buffer(buffer)

      # move final grads to every GPU - fetch(g) has the right
      # grads for dev in it, overwrite with final
      # and optimise
      get_tasks = map(ds_and_ms, gs) do dnm, g
        t = Threads.@spawn begin
          dev, m = dnm
          t_opt = Threads.@spawn begin
            m, st = update(opt, dnm, g, final, sts[dev])
            sts[dev] = st
            m
          end
          (dev, fetch(Base.errormonitor(t_opt)))
        end
        Base.errormonitor(t)
      end
      ds_and_ms = fetch.(get_tasks)

    catch e
      if e isa TaskFailedException && e.task.exception isa CUDA.OutOfGPUMemoryError
        println("Found error: $(e.task.exception)")
        continue

      else
        rethrow(e)
      end
    end
  end
  println("Num Missed: $num_missed")
  map(ds_and_ms) do dnm
    dev, m = dnm
    CUDA.device!(dev) do
      dev, cpu(m)
    end
  end
end

function prepare_training(resnet, key, devices, opt, nsamples;
	                   HOST = first(devices),
                           epochs = nothing,
			   cycle = 5,
			   classes = 1:1000,
			   buffersize = 5)
  ds = Dict()
  cycle = !isnothing(epochs) ? size(key, 1) * epochs ÷ length(devices) ÷ nsamples : cycle
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
      dl = open(BlobTree, DataSets.dataset("imagenet_local")) do data_tree
        dl = Flux.Data.DataLoader((ns,), buffersize = buffersize) do x
          shard = minibatch(data_tree, k, nsamples = x, class_idx = classes)
          CUDA.device!(dev) do
            gpu(shard)
          end
        end
      end
      push!(dls, dl)
    end
  end
  (ds_and_ms = devs_and_ms, dls = dls, sts = sts), buffer
end

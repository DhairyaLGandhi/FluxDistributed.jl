using Flux, Metalhead, BSON, Dates
using Flux.CUDA
using DataSets
using Optimisers, Functors

"""
  syncgrads(input_channels, output_channels; verbose = false)

Starts a task to monitor all the `input_channels` to receive a signal and performs synchronisation of all the terms in the input channels.

The gradients from these channels are expected to be of the form of a `NamedTuple` as produced by Zygote. A typical example would be

```julia
julia> resnet = ResNet(); # from Metalhead.jl or could be any model we wish to train

julia> using Zygote

julia> gs, _ = gradient(resnet, rand(Float32, 224, 224, 3, 1)) do m, x
         sum(m(x))
       end;
```

`gs` is what would be sent in the channels from every worker.

All the input channels are expected to be started at the remote processes with a size 1 so only one gradient may be published at one time. The output channels are similar with the channel being started on the process where the sync in expected to happen.

Typical configuration would look like:

```julia
input_channels = (RemoteChannel(() -> Channel(1), p) for p in workers())
output_channels = (RemoteChannel(() -> Channel(1), 1) for p in workers())
```

Set `verbose = true` to get more detailed information as the synchronisation happens.
"""
function syncgrads(ip, op; verbose = false)
  CUDA.allowscalar(false)
  @info "Starting Syncing workers!"
  while true

    all(isready, ip) || continue
    verbose && @info "All workers ready"
    grads = [take!(c) for c in ip]
    verbose && @info "take! from all workers done"
    vals = grads # map(x -> movetogpu(x.params, x), grads)

    vals = gpu.(vals)
    verbose && @info "Moved all grads to gpus"
    if all(isnothing, vals)
      @info "Got abort signal from every worker"
      break
      # continue
    end

    verbose && @info "Going to reduce grads"

    # for named tuple grads
    final = reduce(vals[2:end], init = vals[1]) do x,y
      Functors.fmap(x, y) do x, y
         isnothing(x) && return y
         isnothing(y) && return x
         _accum(x,y)
      end
    end

    final = Functors.fmap(final) do x
      isnothing(x) && return x
      _dodiv(x, 4.f0)
    end

    verbose && @info "All grads synced"
    
    f = cpu(final)
    for i in op
      put!(i, f)
    end

    verbose && @info "Set the updated_grads"
  end

end

getgrads(loss, dt, m, args...; workers = workers(), kwargs...) =
  getgrads(loss, dt, [m for _ in workers], args...; workers, kwargs...)

maybetuple(x) = (x,)
maybetuple(x::Tuple) = x

loss(x, y) = -sum(y .* Flux.logsoftmax(x) ) ./ Float32(size(y,2))
function getgrads(loss, data_tree,
		  ms::Vector, key, rc, opc;
		  workers = workers(),
		  devices = devices(),
		  verbose = false,
		  nsamples = 2000,
		  batchsize = 48,
		  opt = Optimisers.ADAM(),
		  class_idx,
		  cycles = 1,
                  saveweights = false,
		  sts = [nothing for _ in ms],
		  vals = [nothing for _ in ms],)

  asyncmap(zip(ms, sts, vals, workers, devices, rc, opc)) do (m,st,val,p,d,c,op)

    remotecall_wait(p) do

      device!(d)
      # CUDA.allowscalar(false)
      CUDA.math_mode!(CUDA.DEFAULT_MATH)

      open(BlobTree, DataSets.dataset("imagenet")) do dt
        gm = gpu(m)

        if isnothing(val)
          @info "Worker $(p) does not have validation set"
          val_ = minibatch(dt, key; nsamples = 100, class_idx = class_idx)
        else
          val_ = val
        end

        gval = gpu(val_)
        vl = loss(gm(gval[1]), gval[2])
        @info "Worker $(p) ready to train"
        @info "Worker $(p) starting with loss: $(vl)"

	if isnothing(st)
          st = Optimisers.state(opt, gm)
	end

        for n = 1:cycles
          if n % 5 == 0
            @info "Cycle: $(n)"
          end
          data = minibatch(dt, key; nsamples = nsamples, class_idx = class_idx)
          dl = Flux.Data.DataLoader(gpu, data, batchsize = batchsize)
          for (i,d) in enumerate(dl)
            x, y = d
            y_ = maybetuple(y)
            dm, _, _ = gradient(gm, x, y_) do gm, x, y
              loss(gm(x), y...)
            end

            verbose && @info "Worker $(p) got grads"
            put!(c, cpu(dm))

            verbose && @info "Worker $(p) put grads in channel"
            gnew = gpu(take!(op))
            verbose && @info "Worker $(p) got new grads"

            gm, st = opt(gm, gnew, st)
            if i % 10 == 0
              @info "Worker $(p) at $(i) with loss: $(loss(gm(gval[1]), gval[2]))"
            end
          end
          if saveweights && n % 20 == 0
            @info "before saving"
            model = cpu(gm)
            BSON.@save "weights/$(p)/resnet_50_cycle_$(n)_$(Dates.now()).bson" model
            @info "after saving"
          end
        end
        @info "Worker $(p) finishing with loss: $(loss(gm(gval[1]), gval[2]))"
        @info "Worker $(p) finished - signalling"
        # put!(c, nothing)
        return cpu(gm), cpu(st)
      end
    end
  end
end

"""
    start(loss, data_tree, key, model,
          input_channels, output_channels;
          class_idx,
          verbose = false,
          opt_args = (),
          opt = Optimisers.ADAM(),
          kwargs...)

The high level function that performs training of the model over the specified data and configuration.

* `loss`: Typical loss function used to optimise the model with the data. It is fed all the data that every iteration of a `DataLaader` produces, such that the first element of the produced data is first send to the model. The calling signature of the loss looks like:

    x, y, z... = iterate(dataloader)
    loss(model(x), y, z...)

* `data_tree`: the data tree that one would associate with a dataset described by `DataSets.dataset`.

* `key`: the key  to the training data. Typically the LOC_train__solutions.csv` for the case of ILSVRC.

* `model`: model to be trained

* input_channels, output_channels: See [`syncgrad`](@ref) for details

Keyword Arguments:

This is a non-exhaustive list of keyword arguments currently supported.

* class_idx: a list of the labels to be trained on, a Vector, or Base.OneTo
* o: The type of optimiser to use. Optimisers.jl provides a number of supported optimisers.
* opt_args: A tuple containing arguments to the optimiser as described by `o`
* opt: The output of `o(opt_args...)`. Useful to provide initial optimisers. Can also be associated with schedulers and decays as required.
* cycles: the number of times the dataset is sampled in a single call to `start`.
* batchsize: the number of observations per batch per GPU.
* nsamples: The number of datapoints to be sampled at once. This subset of data is loaded by every process independently and creates a `DataLoader` from it.
* sts: A vector of length `nworkers()` which each contain the current state of the optimiser. This is initialised as `[Optimisers.state(opt, model) for _ in workers()]`
* saveweights: Defaults to `false`, set to `true` to save training checkpoints
* vals: A vector of validation sets to be used as validation sets while training. These also add logging statements while training. Disable with (and defaults to) `[nothing for _ in workers()]`. 
* workers: A list of processes used to train the model
* devices: A `DeviceSet()` or iterable of `Device()`; used to target the GPU used to train.
* verbose: Defaults to `false`. Set to `true` to enable logging of helpful information while debugging and training.
"""
function start(loss, data_tree, key,
	       resnet = Chain(ResNet().layers[1:end-1], Dense(1000, 200)),
	       workers = nworkers(),
	       rcs = (RemoteChannel(() -> Channel(1), p) for p in workers),
	       updated_grads_channel = (RemoteChannel(() -> Channel(1), 1) for p in workers);
	       class_idx, verbose = false,
	       o = Optimisers.ADAM,
	       opt_args = (),
	       opt = o(opt_args...),
	       kwargs...)

  futures = getgrads(loss,data_tree, resnet, key, rcs, updated_grads_channel;
		     class_idx = class_idx,
		     verbose = verbose,
		     opt = opt,
		     kwargs...)

  futures
end

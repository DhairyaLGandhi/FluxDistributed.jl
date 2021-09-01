using Flux, Metalhead, BSON, Dates
using Flux.CUDA
using DataSets
using Optimisers, Functors

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

getgrads(loss, dt, m::Chain, args...; workers = workers(), kwargs...) =
  getgrads(loss, dt, [m for _ in workers], args...; workers, kwargs...)


loss(x, y) = -sum(y .* Flux.logsoftmax(x) ) ./ Float32(size(y,2))
function getgrads(loss, data_tree,
		  ms::Vector{<:Chain}, key, rc, opc;
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

      open(BlobTree, DataSets.dataset("imagenet_cyclops")) do dt
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
            dm, _, _ = gradient(gm, x, y) do gm, x, y
              loss(gm(x), y)
            end

            verbose && @info "Worker $(p) got grads"
            put!(c, cpu(dm))

            verbose && @info "Worker $(p) put grads in channel"
            gnew = gpu(take!(op))
            verbose && @info "Worker $(p) got new grads"

            CUDA.@allowscalar begin
              gm, st = gpu.(opt(cpu(gm), cpu(gnew), cpu(st)))
            end
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

function main(loss, m, ps, data, opt; worker, device, val, epochs = 2, batchsize = batchsize)
  for e in 1:epochs
    dl = Flux.Data.DataLoader(gpu, data, batchsize = batchsize, epochs = epochs)
    for (i,d) in enumerate(dl)
      gs = Flux.gradient(ps) do
        loss(d...)
      end
      Flux.Optimise.update!(opt, ps, gs)
    end
  end
end

function tmp(m, data, val, p, d; batchsize = 16, epochs = 2, opt_args = (0.001, (0.9, 0.99)))
  device!(d)
  gm = gpu(m)
  gval = gpu(val)
  loss(x, y) = -sum(y .* Flux.logsoftmax(gm(x)) ) ./ Float32(size(y,2))
  opt = ADAM(opt_args...)
  main(loss, gm, Flux.params(gm), data, opt;
       worker = p, device = d, val = gval, epochs = epochs,
       batchsize = batchsize)

  cpu(gm), sum(loss(gval...)) # , opt
end

function distribute(m, val, key, data_tree;
                    workers = workers(), batchsize = 16,
                    epochs = 2, opt_args = (0.001, (0.9, 0.99)))
  futures = asyncmap(workers) do p
   remotecall_wait(p) do
     device!(0)
     CUDA.allowscalar(false)

     model = open(BlobTree, DataSets.dataset("imagenet_cyclops")) do new_data_tree
      train_data = new_data_tree[path"ILSVRC/Data/CLS-LOC/train"]
      data = minibatch(train_data, key, nsamples = 2000)
      tmp(m, data, val, p, 0; batchsize, epochs, opt_args)
     end
   end
 end
end

function run_distributed(data_tree, sol_key, m, val; kwargs...)
  cycles = kwargs[:cycles]
  resnet = m
  opt_args = haskey(kwargs, :opt_args) ? kwargs[:opt_args] : (0.001, (0.9,0.99))

  for i in 1:cycles
    @info "Starting cycle $i"
    opt_args = i % 10 == 0 ? (opt_args[1] / 5, opt_args[2]) : opt_args
    
    a, t = @timed fetch.(distribute(resnet, val, sol_key, data_tree,
                      batchsize = kwargs[:batchsize],
                      epochs = kwargs[:epochs],
                      opt_args = opt_args,
                      workers = workers()))

    vl, (resnet, _) = findmin(x -> x[2], a)

    @info "Cycle complete" cycle=i loss=vl cycletime=t
  end
  resnet
end

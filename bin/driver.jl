using Distributed, Dates, DataFrames, DataSets, Optimisers

addprocs(4, exeflags = "--project")
@everywhere include("main.jl")

# key = ResNetImageNet.CSV.read("ext/final_keys.csv", DataFrame)
key = open(BlobTree, DataSets.dataset("imagenet_cyclops")) do data_tree
  ResNetImageNet.train_solutions(data_tree, path"LOC_train_solution.csv", 1:1000)
end

l = ResNetImageNet.CSV.read("ext/final_labels.csv", DataFrame)

ls = open(BlobTree, DataSets.dataset("imagenet_cyclops")) do dt
  ResNetImageNet.labels(dt)
end

classes = mapreduce(vcat, l[!,:label]) do x
  findall(y -> y == x, ls[!,:label])
end

rcs = (RemoteChannel(() -> Channel(1), p) for p in workers()) |> collect

updated_grads_channel = (RemoteChannel(() -> Channel(1), 1) for p in workers()) |> collect

function run_distributed(key, resnet, loss, rcs, updated_grads_channel; classes = 1:1000, kwargs...)
  rs = [resnet for _ in workers()];
  opt = Optimisers.ADAM()

  fs = ResNetImageNet.start(loss, 1,
			 key, rs, workers(),
			 rcs, updated_grads_channel;
                         class_idx = classes,
                         opt = opt,
			 kwargs...)
  rs = fetch.(fs)
  zrs, sts = [r[1] for r in rs], [r[2] for r in rs]


  # stop_syncing = (put!(i, nothing) for i in rcs)
  rs = zrs
end

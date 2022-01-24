using CSV, DataFrames
using ImageMagick
using DataSets
import FileIO
using .Threads

const IMAGENET_BASE = "/home/dhairyalgandhi/imagenet"
const ILSVRC = "ILSVRC"
const ILSVRC_BASE = joinpath(IMAGENET_BASE, "ILSVRC")
const TASK = "CLS-LOC"
const ANNOTATIONS = "Annotations"
const BASE_PATH = "/home/dhairyalgandhi/imagenet/ILSVRC/Data/CLS-LOC"

function labels(data_tree, labels_file = path"LOC_synset_mapping.txt")
  lines = open(IO, data_tree[labels_file]) do io 
    readlines(io)
  end
  t = split.(lines, limit = 2)
  ls = SubString{String}[]
  ds = SubString{String}[]
  for r in t
    push!(ls, r[1])
    push!(ds, r[2])
  end
  DataFrame(:label => ls,
            :description => ds)
end

function minibatch(data_tree, key; nsamples = 16, class_idx, kwargs...)
  s = @view key[rand(1:size(key,1), nsamples), :]
  minibatch(data_tree, s.ImageId, s.class_idx; class_idx, kwargs...)
end

function fproc(data_tree, dest, path)
  x = open(IO, data_tree[path]) do io
    Metalhead.preprocess(ImageMagick.load(io))
  end
  dest .= Flux.normalise(dropdims(x, dims = 4))
end

function fproc(dest::AbstractArray, path, dataset::AbstractString)
  # try
  dest .= Flux.normalise(dropdims(Metalhead.preprocess(joinpath(BASE_PATH, dataset, path)), dims = 4))
  # catch e
  #   @show e, catch_backtrace()
  # end
end

function minibatch(data_tree, img_idxs, img_classes;
                   class_idx = 1:200, dataset = "train")
  arr = zeros(Float32, 224, 224, 3, length(img_idxs))
  ps = makepaths.(img_idxs, dataset)

  ## For some reason @sync -- @async created a deadlock.
  ## The run had to be stopped after 1hr and nothing seemed to be happening.
  @sync for (i,p) in zip(eachslice(arr, dims =4), ps)
    # fproc(data_tree, @view(arr[:,:,:,i]), p)
    u = Threads.@spawn fproc(i, p, dataset)
  end
  arr, Flux.onehotbatch(img_classes, class_idx)
end

function makepaths(img, dataset)
  if dataset == "train"
    joinpath(first(split(img, "_", limit = 2)), img * ".JPEG")
  elseif dataset == "val"
    img * ".JPEG"
  end
end

# function makepaths(imgs, dataset, base = ["ILSVRC", "Data", "CLS-LOC"])
#   if dataset == "train"
#     return DataSets.RelPath([base..., dataset, first(split(imgs, "_", limit = 2)), imgs * ".JPEG"])
#   elseif dataset == "val"
#     return DataSets.RelPath([base..., dataset, img * ".JPEG"])
#   end
# end

function train_solutions(data_tree, train_sol_file = path"LOC_train_solution.csv", classes = 1:200)
  df = open(IO, data_tree[train_sol_file]) do io
    CSV.File(io) |> DataFrame
  end
  s = split.(df[!,:PredictionString])
  l = labels(data_tree)
  cs = map(s) do v
    p = filter(x -> startswith(x, "n"), v)
  end
  df[!,:classes] = cs

  df[!, :class_idx] = map(df[!, :classes]) do v
    cs = map(c -> findfirst(isequal(c), l[!, :label]), v)
    all(isequal(cs[1]), cs) ? cs[1] : cs
  end
  filter!(x -> x.class_idx in classes, df)
  df
end

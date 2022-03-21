using CSV, DataFrames
using ImageMagick
using DataSets
import FileIO
using JpegTurbo
using .Threads

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

struct ImageNet{K}
  key::K
  valkey::K
end

struct DataCache{T, C}
  dataset::T
  cache::C
end

DataCache(d) = DataCache(d, Dict())

dataset(db::DataCache) = db.dataset
cached(db, key) = key in keys(db.cache)

function cache(f, db, key)
  get!(f, db.cache, key)
end

function fproc(data_tree, dest::AbstractArray, path, img_class, dataset::AbstractString, db::DataCache{<:ImageNet}; class_idx)
  f = () -> fproc(data_tree, dest, path)
  preprocessed_img, class = cache(f, db, path)
  # Randomly flip images horizontally
  preprocessed_img_2 = rand() > 0.6 ? @views(preprocessed_img[:, end:-1:1, :]) : preprocessed_img
  dest .= preprocessed_img_2
end

function fproc(data_tree, dest, path)
  # TODO: this should be using `open(data_tree[path])`
  # but FileIO.load doesn't close files
  x = open(data_tree[path]) do io
    preprocess(jpeg_decode(io))
  end
  dest .= Flux.normalise(dropdims(x, dims = 4))
end

function makepaths(imgs, dataset, base = ["ILSVRC", "Data", "CLS-LOC"])
  if dataset == "train"
    return DataSets.RelPath([base..., dataset, first(split(imgs, "_", limit = 2)), imgs * ".JPEG"])
  elseif dataset == "val"
    return DataSets.RelPath([base..., dataset, imgs * ".JPEG"])
  end
end

function minibatch(data_tree, cache::DataCache{<:ImageNet}; nsamples = 16, class_idx, kwargs...)
  key = dataset(cache).key
  minikey = @view key[rand(1:size(key,1), nsamples), :]
  minibatch(data_tree, minikey.ImageId, minikey.class_idx, cache; class_idx, kwargs...)
end

function minibatch(data_tree, img_idxs, img_classes, db::DataCache{<:ImageNet};
                   class_idx = 1:200, dataset = "train")
  arr = zeros(Float32, 224, 224, 3, length(img_idxs))
  paths = makepaths.(img_idxs, dataset)

  us = []
  ## For some reason @sync -- @async created a deadlock.
  ## The run had to be stopped after 1hr and nothing seemed to be happening.
  for (dest, img_path, cl) in zip(eachslice(arr, dims = 4), paths, img_classes)
    u = Threads.@spawn fproc(data_tree, dest, img_path, cl, dataset, db; class_idx = class_idx) # TODO: make sticky
    push!(us, Base.errormonitor(u))
  end
  wait.(us)
  arr, Flux.onehotbatch(img_classes, class_idx)
end


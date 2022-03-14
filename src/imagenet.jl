using CSV, DataFrames
using ImageMagick
using DataSets
import FileIO
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

function minibatch(data_tree, key; nsamples = 16, class_idx, kwargs...)
  s = @view key[rand(1:size(key,1), nsamples), :]
  minibatch(data_tree, s.ImageId, s.class_idx; class_idx, kwargs...)
end

function fproc(data_tree, dest, path)
  datasets_path = data_tree[path]
  localpath = joinpath(datatsets_path.root.path, joinpath(datasets_path.path.components...))
  x = open(newpath) do io
    preprocess(FileIO.load(FileIO.File{FileIO.format"JPEG"}(io)))
  end
  # x = open(IO, data_tree[path]) do io
  #   preprocess(ImageMagick.load(io))
  # end
  dest .= Flux.normalise(dropdims(x, dims = 4))
end

function minibatch(data_tree, img_idxs, img_classes;
                   class_idx = 1:200, dataset = "train")
  arr = zeros(Float32, 224, 224, 3, length(img_idxs))
  ps = makepaths.(img_idxs, dataset)

  ## For some reason @sync -- @async created a deadlock.
  ## The run had to be stopped after 1hr and nothing seemed to be happening.
  @sync for (i,p) in enumerate(ps)
    Threads.@spawn fproc(data_tree, @view(arr[:,:,:,i]), p)
  end
  arr, Flux.onehotbatch(img_classes, class_idx)
end

function makepaths(imgs, dataset, base = ["ILSVRC", "Data", "CLS-LOC"])
  if dataset == "train"
    return DataSets.RelPath([base..., dataset, first(split(imgs, "_", limit = 2)), imgs * ".JPEG"])
  elseif dataset == "val"
    return DataSets.RelPath([base..., dataset, imgs * ".JPEG"])
  end
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

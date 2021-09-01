using CSV, DataFrames
using ImageMagick
using DataSets
import FileIO

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

function minibatch(data_tree, img_idxs, img_classes;
                   class_idx = 1:200, dataset = "train")
  arr = zeros(Float32, 224, 224, 3, length(img_idxs))
  ps = makepaths.(img_idxs, dataset)

  ## For some reason @sync -- @async created a deadlock.
  ## The run had to be stopped after 1hr and nothing seemed to be happening.
  for (i,p) in enumerate(ps)
    fproc(data_tree, @view(arr[:,:,:,i]), p) #, dataset)
  end
  arr, Flux.onehotbatch(img_classes, class_idx)
end

function makepaths(imgs, dataset)
  if dataset == "train"
    return DataSets.RelPath(["ILSVRC", "Data", "CLS-LOC", dataset, first(split(imgs, "_", limit = 2)), imgs * ".JPEG"])
  elseif dataset == "val"
    return DataSets.RelPath(["ILSVRC", "Data", "CLS-LOC", dataset, img * ".JPEG"])
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

function maxk!(ix, a, k; initialized=false)
  partialsortperm!(ix, a, 1:k, rev=true, initialized=initialized)
  @views collect(zip(ix[1:k], a[ix[1:k]]))
end

maxk(a, k; kwargs...) = maxk!(collect(1:length(a)), a, k; kwargs...)

function kacc(kk, oc)
  tot = length(oc)
  corr = 0
  for (i,trueix) in enumerate(oc)
    predixs = [x[1] for x in kk[i]]
    if trueix in predixs
      corr += 1
    end
  end
  corr / tot
end

function topkaccuracy(output, labels; k = 1)
  n = size(output, 2)
  ix = repeat(1:size(output, 1), 1, n)
  preds = maxk!.(eachcol(ix), eachcol(output), k)
  l = Flux.onecold(labels)
  kacc(preds, l) 
end

function showpreds(kk, ground, l; class_idx)
  io = IOBuffer()
  tr = [class_idx[g] for g in ground]
  for (i,(s,t)) in enumerate(zip(kk,tr))
    print(io, "Sample $i: \n")
    classes = [class_idx[x[1]] for x in s]
    corr = findfirst(==(t), classes)
    if isnothing(corr)
      corr = -1
    end
    vals = [x[2] for x in s]
    ls = filter(x -> x.class_idx in classes, l)
    ls = sort(ls, order(:class_idx, by = y -> findfirst(x -> x == y, classes)))
    ls = ls[!, :description]
    for (i,(v,d)) in enumerate(zip(vals, ls))
      print(io, "  ", v, '\t', d)
      if i == corr
        print(io, "  ", "✅")
      end
      print(io, '\n')
    end
    print(io, '\n')
  end
  println(String(take!(io)))
end

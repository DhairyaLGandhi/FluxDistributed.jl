macro device!(dev, ex)
  if CUDA.functional()
    return quote
      CUDA.device!($(esc(dev))) do
        $(esc(ex))
      end
    end
  else
    return ex
  end
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

# using MLDatasets

# MLDatasets.CIFAR10.download(i_accept_the_terms_of_use = true)
const TRAIN_IMG = trainimgs(Metalhead.CIFAR10)

# function minibatch(X = train_x, Y = train_y; nsamples = 10)
#   N = ndims(X)
#   ix = rand(1:size(X, ndims(X)), nsamples)
#   data = Float32.(X[ntuple(i -> i == N ? ix : Colon(), N)...])
#   data, Y[ix]
# end

function assemble(A; dims)
  sf = size(first(A))
  stacked = similar(first(A), (sf[1:end-1]..., length(A)))
  n = ntuple(x -> x == dims ? 9 : Colon(), max(dims, length(sf)))
  for i in 1:length(A)
    stacked[ntuple(x -> x == dims ? i : Colon(), max(dims, length(sf)))...] = A[i]
  end
  stacked
end

# function minibatch(x, ix)
#   imgs = x[ix]
#   # z = zeros(Float32, 224, 224, 3, length(imgs))
#   # Threads.@threads for i in 1:length(imgs)
#   #   z[:,:,:,i] = Metalhead.preprocess(imgs[i].img)
#   # end
#   xs = Metalhead.preprocess.(j.img for j in imgs)
#   ys = [j.ground_truth.class for j in imgs]
#   xs2 = assemble(xs, dims = 4) # cat(xs..., dims = 4)
#   ys2 = Flux.onehotbatch(ys, 1:10)
#   xs2, ys2
# end
#
# function minibatch(x = TRAIN_IMG; nsamples = 16)
#   ix = rand(1:length(x), nsamples)
#   minibatch(x, ix)
# end

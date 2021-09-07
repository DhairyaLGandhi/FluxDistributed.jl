module ResNetImageNet

using Flux, CUDA
using Metalhead
using BSON, Zygote
using Distributed
using Dates, DataSets
using Functors, Optimisers

export minibatch, train_solutions, syncgrads

# include("test.jl")
include("utils.jl")
include("overloads.jl")
include("imagenet.jl")
include("sync.jl")

end # module

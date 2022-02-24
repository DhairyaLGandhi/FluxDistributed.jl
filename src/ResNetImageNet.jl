module ResNetImageNet

using Flux, CUDA
using Metalhead
using BSON, Zygote
using Distributed
using Dates, DataSets
using Functors, Optimisers

export minibatch, train_solutions, syncgrads
export prepare_training

# include("test.jl")
include("preprocess.jl")
include("utils.jl")
include("overloads.jl")
include("imagenet.jl")
# include("sync.jl")
include("ddp_tasks.jl")

end # module

module ResNetImageNet

using Flux, CUDA
using Metalhead
using BSON, Zygote
using Distributed
using Dates, DataSets
using Functors, Optimisers
using Wandb, Logging

export minibatch, train_solutions, syncgrads
export prepare_training

const CONFIG = Dict("cycles" => 6,
                    "batchsize" => 48,
                    "nsamples" => 600,
                    "verbose" => false,
                    "saveweights" => false)

function __init__()
  if get(ENV, "DDP_USE_WANDB", "false") in ["true", true, 1, "1"]
    if !haskey(ENV, "WANDB_DIR")
      ENV["WANDB_DIR"] = joinpath(ENV["HOME"], ".wandb")
      # run(`export WANDB_DIR=$(joinpath(ENV["HOME"], ".wandb"))`)
    end
    global lg = WandbLogger(project = "DDP_Trantor",
                            name = string(now()),
                            config = CONFIG,
                            step_increment = 1)
    global_logger(lg)
  else
    global lg = CONFIG
  end
end

Wandb.get_config(lg::AbstractDict, str) = get!(lg, str, nothing)

# include("test.jl")
include("preprocess.jl")
include("utils.jl")
include("overloads.jl")
include("imagenet.jl")
# include("sync.jl")
include("ddp_tasks.jl")

end # module

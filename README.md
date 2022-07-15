# Data Parallel Training for Flux.jl

[![Docs](https://img.shields.io/badge/Docs-dev-blue)](https://dhairyalgandhi.github.io/FluxDistributed.jl/dev)

Modern large scale deep learning models have increased in size and number of parameters substaintially. This aims to provide tools and mechanisms to scale training of Flux.jl models over multiple GPUs. 

Supports both task based and process based parallelism. The former is suited to single-node parallelism, and the latter to multi-node training. Multi-node training is handled by the same design as multiple locally installed GPU clusters using process based parallelism.

## Setting Up a DataSet

There is a `Data.toml` in the repo which has a few example Datasets. The package uses the `"imagenet_local"` data set by default. Make sure to update the path to where in the system the a dataset is available. In the future, this requirement will be lifted in favour of an API to configure this path in code.

## Single Node Parallelism

### Basic Usage:

Start Julia with the environment of the package activated. This is currently necessary. Start julia with more threads than available. Finally, set up the environment via `] instantiate`.

Here is an example of a simple task based parallel training demo.

```julia
julia> using FluxDistributed, Metalhead, Flux, CUDA, Optimisers, DataSets

julia> classes = 1:1000
1:1000

julia> model = ResNet34();

julia> key = open(BlobTree, DataSets.dataset("imagenet")) do data_tree
         FluxDistributed.train_solutions(data_tree, path"LOC_train_solution.csv", classes)
       end;

julia> val = open(BlobTree, DataSets.dataset("imagenet")) do data_tree
         FluxDistributed.train_solutions(data_tree, path"LOC_val_solution.csv", classes)
       end;

julia> opt = Optimisers.Momentum()
Optimisers.Momentum{Float32}(0.01f0, 0.9f0)

julia> setup, buffer = prepare_training(model, key,
                                        CUDA.devices(),
                                        opt, # optimizer
                                        96,  # batchsize per GPU
                                        epochs = 2);

julia> loss = Flux.Losses.logitcrossentropy
logitcrossentropy (generic function with 1 method)

julia> FluxDistributed.train(loss, setup, buffer, opt,
                            val = val,
                            sched = identity);
```

Here `model` describes the model to train, `key` describes a table of data and how it may be accessed. For the purposes of the demo, this is taken from the `LOC_train_solution.csv` published by ImageNet alongside the images. Look at `train_solutions` which would allow access to the training validation and test sets.

`loss` is a typical loss function used to train a large neural network. The current system is set up for supervised learning, with support for semi supervised learning coming soon. More information can be found in the documentation.

## For process based parallelism - multi-node parallelism

### Basic Usage

* In the `bin` directory, you would find `driver.jl` which has all the initial configuration and a high level function designed to start the training loop
* One would notice that the function takes in several arguments and even more keyword arguments. Most of them have sensible defaults but it is worth mentioning a few ones of interest.
* Start a `julia` command with more threads than the number of logical cores available
* By default, `driver.jl` starts 4 workers, implying training would happen on 4 GPUs. This may be tweaked according to the number of GPUs available.
* This demo is written with heterogenous file loading in mind, such that the data to be trained may be from a local filsystem or hosted remotely (such as on Amazon AWS S3 bucket).

`rcs` and `updated_grads_channel` are `RemoteChannel` between the first process and all the child processes. These are used to send gradients back and forth in order to synchronise them to perform data parallel training.

`syncgrads` starts a task on the main process which continually monitors for gradients coming in from all the available processes and does a manual synchrnisation and sends the updated gradients back to the processes. These gradients are what ultimately trains sent to optimise the model.

### Logging Support

In order to hook into your favourite MLOps backend, it is possible to set up logging with various backends. By default, a trace of the training metrics is printed to console. In order to replace that, it is possible to use Julia's logging infrastructure to set up a different logger.

#### Wandb.jl

[Wandb.jl](https://github.com/avik-pal/Wandb.jl) has unofficial bindings for the wandb.ai platform. 

```julia
using FluxDistributed, Flux, Metalhead
using Wandb, Logging

lg = WandbLogger(project = "DDP",
                 name = "Resnet-34,
                 config = Dict(:saveweights => false, :cycles => 100),
                 step_increment = 1)

with_logger(lg) do
  FluxDistributed.train(...)
end
```

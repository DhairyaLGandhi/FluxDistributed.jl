# Data Parallel Training for Flux.jl

[![Docs](https://img.shields.io/badge/Docs-dev-blue)](https://dhairyalgandhi.github.io/ResNetImageNet.jl/dev)

Modern large scale deep learning models have increased in size and number of parameters substaintially. This aims to provide tools and mechanisms to scale training of Flux.jl models over multiple GPUs. 

Currently uses multi-process parallelism so that it is possible to ducktype multi-node training in the same design as multiple locally installed GPUs.

## Basic Usage

* In the `bin` directory, you would find `driver.jl` which has all the initial configuration and a high level function designed to start the training loop
* One would notice that the function takes in several arguments and even more keyword arguments. Most of them have sensible defaults but it is worth mentioning a few ones of interest.
* By default, `driver.jl` assumes that one GPU per worker would be attached, and that they may be accessible via CUDA.jl
* By default, `driver.jl` starts 4 workers, implying training would happen on 4 GPUs. This may be tweaked according to the number of GPUs available.
* This demo is written with heterogenous file loading in mind, such that the data to be trained may be from a local filsystem or hosted remotely (such as on Amazon AWS S3 bucket).

### To start:

Start Julia with the environment of the package activated. This is currently necessary. It is also necesary that the environment be instaniated. This may be done via `] instantiate`.


```julia
julia> include("driver.jl") # expects 4 GPUs, but may be tweaked
run_distributed (generic function with 1 method)

julia> synct = @async syncgrads(rcs, updated_grads_channel)
[ Info: Starting Syncing workers!
Task (runnable) @0x00007ffac5b65120

julia> resnet = ResNet().layers[1:end-1];

julia> run_distributed(key, resnet, ResNetImageNet.loss,
                       rcs, updated_grads_channel)
```

Here `resnet` describes the model to train, `key` describes a table of data and how it may be accessed. For the purposes of the demo, this is taken from the `LOC_train_solution.csv` published by ImageNet alongside the images. Look at `train_solutions` which would allow access to the training validation and test sets.
`rcs` and `updated_grads_channel` are `RemoteChannel` between the first process and all the child processes. These are used to send gradients back and forth in order to synchronise them to perform data parallel training.

`syncgrads` starts a task on the main process which continually monitors for gradients coming in from all the available processes and does a manual synchrnisation and sends the updated gradients back to the processes. These gradients are what ultimately trains sent to optimise the model.

`loss` is a typical loss function used to train a large neural network. The current system is set up for supervised learning, with support for semi supervised learning coming soon. More information can be found in the documentation.

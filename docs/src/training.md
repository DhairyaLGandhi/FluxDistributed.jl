# Training pipeline

Data parallel training happens over several GPUs which themselves may be spread across sevral nodes. While there are several architectures proposed for high throughput and fast training of neural networks with a large amount of data, detailing them is out of scope for this document. Here we will focus on the tooling necessary to achieve training over several GPUs and the API.

## Batching the Data

There are several strategies that can be employed for parallel loading of data. Typically, the process involves sharding of data by the number of accelerator units, and creating data loaders which can load data from the disk asynchronously with the training loop. In order to serve all the accelerators, one can call a `DataParallelDataLoader` or create `N` instances of iterable producing mini-batches (where `N` refers to the number of accelerator units).

In this package, the `prepare_training` function uses a modified version of a Flux DataLoader which can simultaneously feed `N` accelerators and load data from disk in parallel with the training. Data is managed through DataSets.jl (it is also useful for setting up any custom dataset, and more information can be found in the [datasets document](../datasets.md).

```@docs
ResNetImageNet.prepare_training
```

## Syncing Gradients

Distributed training requires the gradients to be synchronized during the process of training. Using this, a model can get the effect of training over the cumulative data used by every accelerator at every step. This is done for the following reasons:

* `N` instances of the training pipeline are instantiated on `N` accelerators.
* Every accelerator then injests data independently and produces gradients corresponding to its specific sub-batch of data.
* By reducing the gradients over all the instances, we can simulate training over all the sub-batches at once.
  * By default, the gradients are averaged over all avaliable instances of the model, but this behaviour can be customised in [`ResNetImageNet.sync_buffer`]()

### Single Node Parallelism

```@docs
ResNetImageNet.sync_buffer
```

`sync_buffer` currently requires maintaining preallocated memory on one of the accelerator units in order to not pay the price for allocations with every synchnorization step. It makes a call to `copyto!` and in case the accelerators are not connected via a P2P mechanism, can cause implicit serialization to the CPU, hurting performance significantly.

### Multi Node Parallelism

!!! Note
    The multi-node parallelism pipeline is currently disabled, but available in the repository for expermential purposes via `ResNetImageNet.syncgrads`.

```@docs
ResNetImageNet.syncgrads
```

Note that `syncgrads` currently requires serialization of gradients from every device with every iteration of the data loader. This is inefficient and has been surpassed with techniques involving "data layers" such as Nvidia NCCL or UCX which work to perform reduction over several GPUs in a better optimised manner. This is under developement in the Julia ecosystem as well.

## High Level training pipeline

A basic pipeline would look very similar to how `Flux.train!` functions. We need a cost function, some data and parameters and we are off to the races. There are some subtlties however. Rather than using the gradients that every copy of the model produces with the data it used, data parallel training does not directly use these gradients. Instead, the gradients from every model have to be reduced together to average it out to get the effect of training over all the data used to perform a single training step. This amounts to:

```julia
# state => state of optimiser
# model => model to be trained
# opt => optimiser
# data => collection of data
for d in data
  x, y = d  # x => input data, y => labels
  gs = gradient(model) do m
    loss(m(x), y)
  end

  # note the pseudo code call in the next line
  @synchronize_gradients_over_all_GPUs
  model, state = opt(model, updated_grads, state)
end
```

This looks very similar to the typical supervised learning training loop from Flux.

In fact, it is! With the addition of the synchrnisation part, we can also extend it to several forms of semi-supervised and unsupervised learning scenarios. This is part of the future work of this pacakge, and something actively being researched in the Julia community and elsewhere.

```@docs
ResNetImageNet.prepare_training
ResNetImageNet.train_step
ResNetImageNet.update
ResNetImageNet.train
```

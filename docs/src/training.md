# Training pipeline

Data parallel training happens over several GPUs which themselves may be spread across sevral nodes. While there are several architectures proposed for high throughput and fast training of neural networks with a large amount of data, it is out of scope for this document. Here we will focus on the tooling necessary to achieve training over several GPUs and the API.

## Syncing Gradients

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
ResNetImageNet.start
```

# DataSets

DataSets needed to train large neural networks in a distributed fashion also require the data be highly available over the network. It is also desirable to have access to the dataset from a local filesystem for testing, debugging etc.

This is done via the [DataSets.jl](https://github.com/JuliaComputing/DataSets.jl) package which includes abstractions over how the data is loaded, indexed and found over several backends, including tree-like structures (local filesystems) or hosted over the network (such as in Amazon S3 buckets).

Using DataSets.jl we can represent a data storage driver which can be used to perform tasks such downloading and encoding etc with a shared API across backends. This package also includes a `Data.toml` which describes the ImageNet dataset used to train our models. This may be swapped out for a different dataset as needed.

```@docs
FluxDistributed.minibatch
FluxDistributed.train_solutions
FluxDistributed.labels
FluxDistributed.topkaccuracy
FluxDistributed.showpreds
```

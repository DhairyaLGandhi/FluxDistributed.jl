var documenterSearchIndex = {"docs":
[{"location":"training/#Training-pipeline","page":"Training","title":"Training pipeline","text":"","category":"section"},{"location":"training/","page":"Training","title":"Training","text":"Data parallel training happens over several GPUs which themselves may be spread across sevral nodes. While there are several architectures proposed for high throughput and fast training of neural networks with a large amount of data, detailing them is out of scope for this document. Here we will focus on the tooling necessary to achieve training over several GPUs and the API.","category":"page"},{"location":"training/#Batching-the-Data","page":"Training","title":"Batching the Data","text":"","category":"section"},{"location":"training/","page":"Training","title":"Training","text":"There are several strategies that can be employed for parallel loading of data. Typically, the process involves sharding of data by the number of accelerator units, and creating data loaders which can load data from the disk asynchronously with the training loop. In order to serve all the accelerators, one can call a DataParallelDataLoader or create N instances of iterable producing mini-batches (where N refers to the number of accelerator units).","category":"page"},{"location":"training/","page":"Training","title":"Training","text":"In this package, the prepare_training function uses a modified version of a Flux DataLoader which can simultaneously feed N accelerators, load data and move it to the accelerator in parallel with the training. This way, one can write custom loading and preprocessing scripts to be run in parallel with the training, and evern overlapping network costs to move data to the accelerator without training loop needing to wait for the data to be available to it. Data is managed through DataSets.jl (it is also useful for setting up any custom dataset, and more information can be found in the datasets document.","category":"page"},{"location":"training/","page":"Training","title":"Training","text":"FluxDistributed.prepare_training","category":"page"},{"location":"training/#Syncing-Gradients","page":"Training","title":"Syncing Gradients","text":"","category":"section"},{"location":"training/","page":"Training","title":"Training","text":"Distributed training requires the gradients to be synchronized during the process of training. Using this, a model can get the effect of training over the cumulative data used by every accelerator at every step. This is done for the following reasons:","category":"page"},{"location":"training/","page":"Training","title":"Training","text":"N instances of the training pipeline are instantiated on N accelerators.\nEvery accelerator then injests data independently and produces gradients corresponding to its specific sub-batch of data.\nBy reducing the gradients over all the instances, we can simulate training over all the sub-batches at once.\nBy default, the gradients are averaged over all avaliable instances of the model, but this behaviour can be customised in FluxDistributed.sync_buffer","category":"page"},{"location":"training/#Single-Node-Parallelism","page":"Training","title":"Single Node Parallelism","text":"","category":"section"},{"location":"training/","page":"Training","title":"Training","text":"FluxDistributed.sync_buffer","category":"page"},{"location":"training/","page":"Training","title":"Training","text":"sync_buffer currently requires maintaining preallocated memory on one of the accelerator units in order to not pay the price for allocations with every synchnorization step. It makes a call to copyto! and in case the accelerators are not connected via a P2P mechanism, can cause implicit serialization to the CPU, hurting performance significantly.","category":"page"},{"location":"training/#Multi-Node-Parallelism","page":"Training","title":"Multi Node Parallelism","text":"","category":"section"},{"location":"training/","page":"Training","title":"Training","text":"!!! Note     The multi-node parallelism pipeline is currently disabled, but available in the repository for expermential purposes via FluxDistributed.syncgrads.","category":"page"},{"location":"training/","page":"Training","title":"Training","text":"FluxDistributed.syncgrads","category":"page"},{"location":"training/","page":"Training","title":"Training","text":"Note that syncgrads currently requires serialization of gradients from every device with every iteration of the data loader. This is inefficient and has been surpassed with techniques involving \"data layers\" such as Nvidia NCCL or UCX which work to perform reduction over several GPUs in a better optimised manner. This is under developement in the Julia ecosystem as well.","category":"page"},{"location":"training/#High-Level-training-pipeline","page":"Training","title":"High Level training pipeline","text":"","category":"section"},{"location":"training/","page":"Training","title":"Training","text":"A basic pipeline would look very similar to how Flux.train! functions. We need a cost function, some data and parameters and we are off to the races. There are some subtlties however. Rather than using the gradients that every copy of the model produces with the data it used, data parallel training does not directly use these gradients. Instead, the gradients from every model have to be reduced together to average it out to get the effect of training over all the data used to perform a single training step. This amounts to:","category":"page"},{"location":"training/","page":"Training","title":"Training","text":"# state => state of optimiser\n# model => model to be trained\n# opt => optimiser\n# data => collection of data\nfor d in data\n  x, y = d  # x => input data, y => labels\n  gs = gradient(model) do m\n    loss(m(x), y)\n  end\n\n  # note the pseudo code call in the next line\n  @synchronize_gradients_over_all_GPUs\n  model, state = opt(model, updated_grads, state)\nend","category":"page"},{"location":"training/","page":"Training","title":"Training","text":"This looks very similar to the typical supervised learning training loop from Flux.","category":"page"},{"location":"training/","page":"Training","title":"Training","text":"In fact, it is! With the addition of the synchronization part, we can also extend it to several forms of semi-supervised and unsupervised learning scenarios. This is part of the future work of this pacakge, and something actively being researched in the Julia community and elsewhere.","category":"page"},{"location":"training/","page":"Training","title":"Training","text":"FluxDistributed.train_step\nFluxDistributed.update\nFluxDistributed.train","category":"page"},{"location":"#Data-Parallel-Training-for-Flux-models","page":"Home","title":"Data Parallel Training for Flux models","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Training large deep learning models over several nodes and architectures is required to efficiently scale training of larger number of parameters that cannot reasonbly be trained in a single machine or node.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package includes tools necessary to efficiently scale and train deep learning models with high throughput.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package includes a setup and configuration environment in bin/driver.jl which can be tweaked depending on the needs.","category":"page"},{"location":"datasets/#DataSets","page":"Datasets","title":"DataSets","text":"","category":"section"},{"location":"datasets/","page":"Datasets","title":"Datasets","text":"DataSets needed to train large neural networks in a distributed fashion also require the data be highly available over the network. It is also desirable to have access to the dataset from a local filesystem for testing, debugging etc.","category":"page"},{"location":"datasets/","page":"Datasets","title":"Datasets","text":"This is done via the DataSets.jl package which includes abstractions over how the data is loaded, indexed and found over several backends, including tree-like structures (local filesystems) or hosted over the network (such as in Amazon S3 buckets).","category":"page"},{"location":"datasets/","page":"Datasets","title":"Datasets","text":"Using DataSets.jl we can represent a data storage driver which can be used to perform tasks such downloading and encoding etc with a shared API across backends. This package also includes a Data.toml which describes the ImageNet dataset used to train our models. This may be swapped out for a different dataset as needed.","category":"page"},{"location":"datasets/","page":"Datasets","title":"Datasets","text":"FluxDistributed.minibatch\nFluxDistributed.train_solutions\nFluxDistributed.labels\nFluxDistributed.topkaccuracy\nFluxDistributed.showpreds","category":"page"}]
}

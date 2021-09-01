Data Parallel Training for Flux.jl

Modern large scale deep learning models have increased in size and number of parameters substaintially. This aims to provide tools and mechanisms to scale training of Flux.jl models over multiple GPUs. 

Currently uses multi-process parallelism so that it is possible to ducktype multi-node training in the same design as multiple locally installed GPUs.

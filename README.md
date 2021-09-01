Data Parallel Training for Flux.jl

Modern large scale deep learning models have increased in size and number of parameters substaintially. Therefore, it is required that they be trained on several GPUs at once.

Currently uses multi-process parallelism so that it is possible to ducktype multi-node training in the same design as multiple locally installed GPUs.

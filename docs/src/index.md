# Data Parallel Training for Flux models

Training large deep learning models over several nodes and architectures is required to efficiently scale training of larger number of parameters that cannot reasonbly be trained in a single machine or node.

This package includes tools necessary to efficiently scale and train deep learning models with high throughput.

This package includes a setup and configuration environment in `bin/driver.jl` which can be tweaked depending on the needs.

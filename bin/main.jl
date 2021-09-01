using ResNetImageNet
using ResNetImageNet.BSON
using ResNetImageNet.Metalhead
using ResNetImageNet.Flux
using ResNetImageNet.DataSets

using Distributed
@everywhere using Flux, BSON, CUDA, Metalhead, Zygote, Distributed, DataSets, ResNetImageNet

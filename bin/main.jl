using FluxDistributed
using FluxDistributed.BSON
using FluxDistributed.Metalhead
using FluxDistributed.Flux
using FluxDistributed.DataSets

using Distributed
@everywhere using Flux, BSON, CUDA, Metalhead, Zygote, Distributed, DataSets, FluxDistributed

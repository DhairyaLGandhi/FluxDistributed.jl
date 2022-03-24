using ResNetImageNet, DataSets, Flux, Metalhead
using CUDA
using ResNetImageNet.Optimisers, ResNetImageNet.Functors
using Test

function compare(y::Tuple, ŷ::Tuple)
  foreach((a,b) -> compare(a, b), y, ŷ)
end

function compare(y::NamedTuple, ŷ::NamedTuple)
  foreach((a,b) -> compare(a, b), y, ŷ)
end

function compare(a::AbstractArray, b::AbstractArray)
  @test isapprox(a, b, rtol = 1f-4, atol = 1f-4)
end

function compare(a::Base.RefValue, b::Base.RefValue)
  compare(a[], b[])
end

function compare(::Nothing, ::Nothing)
  @test true
end

function compare(a::T, b::T) where T
  fs = fieldnames(T)
  foreach(f -> compare(getfield(a, f), getfield(b, f)), fs)
end

compare(x::Union{NTuple{N,Nothing}, Tuple{Vararg{Nothing}}}, ::Nothing) where N = nothing

function compare(a, b)
  @test a ≈ b
end

getfirst(x::NamedTuple, f) = getfirst(x.f)
getfirst(x::NamedTuple{(:layers,)}, f) = getfirst(x.layers, f)
getfirst(x::Tuple, f) = getfirst(first(x), f)
getfirst(x::NamedTuple, f) = getfirst(getfield(x, f), f)
getfirst(x, f) = x

include("single_device.jl")


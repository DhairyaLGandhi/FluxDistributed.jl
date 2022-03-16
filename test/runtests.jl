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
  @test a ≈ b
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

function compare(a, b)
  @test a ≈ b
end

@testset "Single Device vs. Many Devices" begin
  include("single_device.jl")
end

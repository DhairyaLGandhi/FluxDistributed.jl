using ResNetImageNet, DataSets, Flux, Metalhead
using CUDA
using ResNetImageNet.Optimisers, ResNetImageNet.Functors
using Test

function compare(y::Tuple, ŷ)
  foreach((a,b) -> compare(a, b), y, ŷ)
end

function compare(y::NamedTuple, ŷ)
  foreach((a,b) -> compare(a, b), y, ŷ)
end

function compare(a::AbstractArray, b)
  @testset "Arrays" begin
    @test a ≈ b
  end
end

function compare(a::Base.RefValue, b::Base.RefValue)
  compare(a[], b[])
end

function compare(::Nothing, ::Nothing)
  @testset "Nothings" begin
    @test true
  end
end

function compare(a, b)
  @testset "Generic" begin
    @test a ≈ b
  end
end

@testset "Single Device vs. Many Devices" begin
  include("single_device.jl")
end

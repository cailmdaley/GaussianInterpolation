module GaussianInterpolation

using CMBLensing
using CMBLensing: Pix

using CMB: Pl
using Distributions: MvNormal
using Distances: Metric, Euclidean, Haversine, pairwise!
using FFTW
using ImageFiltering
using Interpolations
using LinearAlgebra
using Match
using NearestNeighbors: KDTree, knn
using OffsetArrays
using ProgressMeter
using Parameters
using Statistics
using StaticArrays
import Statistics: cov
import Base: |, eltype, getindex, +, -, *, length, similar, parent, iterate, show, eachindex

export @unpack

include("coordinates.jl")
include("kernel.jl")
include("gp.jl")
include("models.jl")
include("predict.jl")
include("design.jl")
include("interpolation.jl")

end

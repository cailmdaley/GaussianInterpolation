module GaussianInterpolation

using Reexport
@reexport using CMBLensing
@reexport using Unitful, UnitfulAstro

using CMB: Pl
using Distances: pairwise, Euclidean, Haversine, Metric
using ImageFiltering
using Interpolations: interpolate, Linear, BSpline, ScaledInterpolation, scale
using LinearAlgebra
using NearestNeighbors: KDTree, knn
using ProgressMeter
using Statistics

export InterpolatedCovariance
export interp_flexible, interp_imfilter, interp_knn
export coords
export gp_mean, gp_var

include("coordinates.jl")
include("covariance.jl")
include("interpolation.jl")

end

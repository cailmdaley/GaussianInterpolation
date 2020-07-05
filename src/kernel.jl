export KernelFunction, MapKernel, CℓKernel, CompositeKernel
export InterpKernel
export Stationary, NonStationary
export parent
export cov, cov!

### Kernel Properties
####################
abstract type KernelProperty end
abstract type Stationary <: KernelProperty end
abstract type NonStationary <: KernelProperty end

### Abstract Kernel Functions
############
abstract type KernelFunction{KP<:KernelProperty, T} end

eltype(K::KernelFunction{KP, T}) where {KP, T} = T
isstationary(K::KernelFunction{KP, T}) where {KP, T} = KP <: Stationary

### Map-based Kernel Functions
#############
@with_kw struct MapKernel{KP,T,IT} <: KernelFunction{KP,T}
	itp::IT
	M::Int
    vec::Vector{T} = zeros(T, M)
    mat::Matrix{T} = zeros(T, M, M)
end
function MapKernel{KP}(m::Matrix{T}, M) where {KP, T}
    axes_func = KP <: Stationary ? axes_centered : axes
    itp = CubicSplineInterpolation(axes_func(m), m, extrapolation_bc=0.0)
    MapKernel{KP, T, typeof(itp)}(itp=itp, M=M)
end
MapKernel(f::FlatMap, M)     = MapKernel{NonStationary}(f.Ix, M)
MapKernel(f::FlatFourier, M) = MapKernel{Stationary}(fftshift(Map(f).Ix), M)
MapKernel(op::LinOp, M)      = MapKernel(diag(op), M)

@inline getindex(K::MapKernel, xs...)    = K.itp(xs...)
@inline getindex(K::MapKernel, x::CartesianIndex{2}) = K.itp(Tuple(x)...)

length(K::MapKernel) = length(K.vec)
parent(K::MapKernel) =  K.itp.itp.itp.coefs.parent


# callables that handle Vector vs. Matrix and Stationary vs. NonStationary
(K::MapKernel{Stationary})(T::Type{Vector},  X, x) = cov!(K.vec, K, X)
(K::MapKernel{Stationary})(T::Type{Matrix}, X, x) = cov!(K.mat, K, X)

(K::MapKernel{NonStationary})(T::Type{Vector}, X, x) = Diagonal(cov!(K.vec, K, X, -x))
(K::MapKernel{NonStationary})(T::Type{Matrix}, X, x) = cov!(K.mat, K, X, -x)

# Covariance Loops
##################

# Vector, Vector -> Matrix
function cov!(out::Matrix, K::MapKernel, X)
	(Xy, Xx) = split_coords(X)
	M = size(out)[1]
    for i in 1:M, j in 1:M
		out[j, i] = K[Xy[i] - Xy[j], Xx[i] - Xx[j]]
	end
	out
end

# Vector, Coord -> Diagonal Matrix
function cov!(out::Matrix, K::MapKernel, X, x)
	(Xy, Xx), (xy, xx) = split_coords(X), x
	out .= 0
    for i in 1:K.M
		out[i, i] = K[Xy[i] - xy, Xx[i] - xx]
	end
	out
end

# Vector -> Vector
function cov!(out::Vector, K::MapKernel, X)
	Xy, Xx = @views X[:, 1], X[:, 2]
    for i in 1:K.M
		out[i] = K[Xy[i], Xx[i]]
	end
	out
end

#Vector, Coord -> Vector
function cov!(out::Vector, K::MapKernel, X, x)
	(Xy, Xx), (xy, xx) = split_coords(X), x
    for i in 1:K.M
		out[i] = K[Xy[i] - xy, Xx[i] - xx]
	end
	out
end

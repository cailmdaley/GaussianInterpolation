module GaussianInterpolation

using CMBLensing
using FFTW
using Interpolations
using LinearAlgebra
using Match
using NearestNeighbors: KDTree, knn
using ProgressMeter
using Statistics
using StaticArrays
using SparseArrays
import Statistics: cov
import Base: |, eltype, getindex, +, -, *, length, similar, parent, iterate, show, eachindex

export @unpack

axes_centered(A) = Tuple((ax .- offset for (ax, offset) in zip(axes(A), size(A) .÷ 2 .+ 1)))
	
function coords(m::AbstractMatrix)
    I = CartesianIndices(m)
    X = Matrix{Int32}(undef, length(I), 2)
    for (i, ind) in zip(LinearIndices(I), I)
        X[i, :] .= Tuple(ind)
    end
    X
end
coords(f::FlatMap) = coords(f.Ix)
coords(f::FlatFourier) = coords(Map(f).Ix)
coords(f::FieldTuple) = coords(first(f))


# Functions for filling covariance matrices / vectors
function cov!(out::AbstractVector, Kitp, X)
    for i in 1:length(out)
        out[i] = Kitp(X[i, 1], X[i, 2])
    end
    out
end
function cov!(out::AbstractMatrix, Kitp, X)
    M = size(out)[1]
    for i in 1:M, j in 1:M
        out[j, i] = Kitp(X[i, 1] - X[j, 1], X[i, 2] - X[j, 2])
    end
    out
end	


# Kernels and Masks
export make_kernel, make_mask
make_kernel(L::LinOp) = make_kernel(diag(L))
make_kernel(f::FlatFourier) = make_kernel(fftshift(Map(f).Ix))
make_kernel(m::Matrix) = CubicSplineInterpolation(axes_centered(m), m, extrapolation_bc=0.0)
function kernel_dict(C::LinOp, predict::NTuple) 
    names = (Symbol(p...) for p in predict)
    Dict(name => make_kernel(C[name]) for name in names)
end

make_mask(L::LazyBinaryOp)= make_mask(L.b)
make_mask(D::DiagOp{<:Field{Map}})= diag(D)
make_mask(D::DiagOp{<:FlatS02})= diag(D)[:I]

export sparse_gp
function sparse_gp(ds::DataSet; m=15)
	X = coords(ds.d)
	Xtree = KDTree(permutedims(float.(X)))
	skip_predicate = (i -> iszero(mask[i]) ? true : false)
    design_inds(i) = knn(Xtree, X[i, :], m, false, skip_predicate)[1]

	Kf, Kn = make_kernel.((ds.Cf, ds.Cn))
	mask = make_mask(ds.M)
	
    function compute_row!(i, J, V)
        js = design_inds(i)
        Xj = X[js, :] .- X[i:i, :]
        
        s = zeros(T, m)
        S = zeros(T, (m,m))
        N = similar(S)
        cov!(s, Kf, Xj)
        cov!(S, Kf, Xj)
        cov!(N, Kn, Xj)
        
        inds = (i-1)*m+1:i*m
        J[inds] = js
        V[inds] = (S + N) \ s
    end

    @unpack Nside, T = fieldinfo(ds.d)
    I = Vector{Int32}((m:m*Nside^2 + (m-1)) .÷ m)
    J = similar(I)
    V = similar(I,T)
    for i in 1:Nside^2
        compute_row!(i, J, V)
    end
    SparseMatGP(sparse(I,J,V,Nside^2,Nside^2))
end

mutable struct SparseMatGP
    M::SparseMatrixCSC
end
*(gp::SparseMatGP, f::FlatMap{P}) where P = FlatMap{P}(reshape(gp.M * f[:], size(f.Ix)))
*(gp::SparseMatGP, f::FlatFourier) = gp * Map(f)



export classic_gp
function classic_gp(ds::DataSet, predict::NTuple, m::Int; ϕ=nothing)
    @unpack Nside, T, Δx = fieldinfo(ds.d)

    # Kernels, Covariance Matrices, and Mask
    Ks, Kn = kernel_dict.((ds.Cf, ds.Cn), Ref(predict))
    s, S, N = zeros(T, m), zeros(T, (m,m)), zeros(T, (m,m))
    mask = make_mask(ds.M)

    # Coordinates and Nearest Nearest Neighbors Tree
    X_predict = coords(ds.d)
    if ϕ !== nothing
        αy, αx = getindex.(Map.(∇ * ϕ ./ Δx ), :)
        X_obs = X_predict .+ [αx αy]
    else
        X_obs = X_predict
    end
    Xtree = KDTree(permutedims(float.(X_obs)))
    
    # Local GP Design
	skip_predicate = (i -> iszero(mask[i]) ? true : false)
    design_inds(i) = knn(Xtree, X_obs[i, :], m, false, skip_predicate)[1]

    function apply(f::FlatField{P, T, M}) where {P, T, M}
        f_in, f_out = Dict{Symbol, Matrix{T}}(), Dict{Symbol, Matrix{T}}()
        for (p_in, p_out) in predict
            f_in[p_in] = f[Symbol(string(p_in)*"x")]
            f_out[p_out] = similar(f_in[p_in])
        end
        for i in 1:Nside^2
            # calculate reference point coordinates
            js = design_inds(i)
            Xj = X_obs[js, :] .- X_predict[i:i, :]
            
            # loop over things we're predicting
            for (in, out) in predict
                for (m, kernel) in ((s, Ks), (S, Ks), (N, Kn)) 
                    cov!(m, kernel[Symbol(in, out)], Xj)
                end
                f_out[out][i] = ((S + N) \ s) ⋅ f_out[in][js]
            end
        end
        f_out
    end

    apply
end


# module end
end

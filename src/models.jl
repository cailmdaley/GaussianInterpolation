export BaseGP, MaskedGP
export solve

struct BaseGP{T, IT} <: GPModel
    Kf::MapKernel{Stationary,    T, IT}
    Kn::MapKernel{Stationary,    T, IT}
end
BaseGP(gp::GP{N}) where N = BaseGP(MapKernel.((gp.ds.Cf, gp.ds.Cn),  N)...)

struct MaskedGP{T, P, MM, IT} <: GPModel
    M::FlatMap{P,T,MM}
    Kf::MapKernel{Stationary,    T, IT}
    Kn::MapKernel{Stationary,    T, IT}
end
MaskedGP(gp::GP{N}) where N = MaskedGP(diag(gp.ds.M.b), MapKernel.((gp.ds.Cf, gp.ds.Cn),  N)...)


function covariance(model::Union{BaseGP, MaskedGP}, x, X)
    s = model.Kf(Matrix, X, x)
    n = model.Kn(Matrix, X, x)
    s + n
end

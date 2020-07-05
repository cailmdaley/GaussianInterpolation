export GP
export Design, GPModel
export Lens, Delens, Filter
export materialize

abstract type Predict{names} end
abstract type Lens{names} <: Predict{names} end
abstract type Delens{names} <: Predict{names} end
abstract type Filter{names} <: Predict{names} end

abstract type Design end
abstract type GPModel end


struct GP{N, P<:Predict, M<:GPModel, D<:Design}
    ds::DataSet
end
function GP(ds, n::Int=9, predict_keys::Vararg{Symbol, N}=:μ) where N
    GP{n, Filter{predict_keys}, MaskedGP, KNN}(ds)
end
GP(ds, predict_keys::Vararg{Symbol, N}) where N = GP(ds, 9, predict_keys)

predict_keys(::GP{N, P}) where {N, names, P <: Predict{names}} = names
function materialize(gp::GP{N,P,M,D}, d) where {N,P,M,D}
    data = PredictData(gp, d)
    model = M(gp)
    design = D(gp, data)
    @namedtuple(data, model, design)
end

function (gp::GP)(d)
    out = interp(materialize(gp, d)...)
    isone(length(out)) ? first(out) : (; zip(keys(get_predict_kernels(gp)), out)...)
end
*(gp::GP, d) = gp(d)

export PredictData

struct PredictData{P, T, M, T′, IT}
    d::FlatMap{P, T, M}
    Xs_obs::Matrix{T′}

    ks_predict::NTuple{N, MapKernel{Stationary, T, IT}} where N
    ds_predict::NTuple{N, FlatMap{P, T, M}} where N
    Xs_predict::Matrix{T′}
end
function PredictData(gp::GP, d::FlatMap; Xs_obs=coords(d.Ix), Xs_predict=Xs_obs)
    ks_predict = values(get_predict_kernels(gp))
    ds_predict = Tuple(similar(d) for k in ks_predict)
    PredictData(d, Xs_obs, ks_predict, ds_predict, Xs_predict)
end
PredictData(gp::GP, d::FlatFourier; kwargs...) = PredictData(gp, Map(d), kwargs...)

function get_predict_kernels(gp::GP{N}) where N
    ks = Dict()
    for k in predict_keys(gp)
        if k == :μ
            ks[:μ] = MapKernel(gp.ds.Cf, N)
        elseif k == :∇
            ∇y, ∇x = MapKernel.(∇ * diag(gp.ds.Cf), N)
            ks[:∇y] = ∇y
            ks[:∇x] = ∇x
        end
    end
    NamedTuple{Tuple(keys(ks))}(values(ks))
end

coords_predict(data::PredictData, i)  = @view data.Xs_predict[i:i, :]
coords_design(data::PredictData, is) = @view data.Xs_obs[is, :]

eachindex(data::PredictData) = eachindex(data.d)

function predict!(data::PredictData{names}, model, i, is) where names
    x = coords_predict(data, i)
    X = coords_design(data, is) .- x
    weights = covariance(model, x, X) \ data.d[is]
    for (out, k) in zip(data.ds_predict, data.ks_predict)
        out[i] = weights ⋅ k(Vector, X, x)
    end
end

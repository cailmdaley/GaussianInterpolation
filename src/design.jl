export KNN

@with_kw struct KNN{T} <: Design
    Xtree::KDTree{SArray{Tuple{2}, T, 1, 2}, Euclidean, T}
    n::Int
end
KNN(gp::GP{N}, data) where N = KNN(KDTree(permutedims(float.(data.Xs_obs))), N)

function design_is(data, model::MaskedGP, design::KNN, i)
    skip_predicate = (i -> iszero(model.M[i]) ? true : false)
    knn(design.Xtree, vec(coords_predict(data, i)), design.n, false, skip_predicate)[1]
end

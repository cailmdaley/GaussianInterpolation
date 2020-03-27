"""
Solves Gaussian process regression problem.
kf: m x n matrix
S: n x n matrix
N: n x n matrix
where m and n  are the number of prediction and reference points, respectively.
"""
gp_mean(y,  kf, Kf, Kn) = kf ⋅ ((Kf + Kn) \ y)
gp_var(σ₀², kf, Kf, Kn) = σ₀² - kf' ⋅ ((Kf + Kn) \ kf)

function interp_flexible(d::FlatMap, Kf, Kn, N; var=false, condnum=false, progress=false)
	f̄ = similar(d)
	if var
		σ² = similar(d)
	end

	Δx = fieldinfo(d).Δx
	r = Int(sqrt(N)) ÷ 2 # kernel width
	field_inds = CartesianIndices(d[:Ix])
	@showprogress (progress ? 1 : Inf) for ind in field_inds
		i,j = [min(max(Tuple(ind)[i], 1 + r), size(field_inds)[i] - r)
		       for i in 1:2]
		kernel_inds = CartesianIndex(i,j) .+ CartesianIndices((-r:r, -r:r))
		X  = coords(kernel_inds) .* Δx
		x₀ = coords([ind]) .* Δx

		covariances = Kf(x₀, X), Kf(X, X), Kn(X, X)
		f̄[ind] = gp_mean(vec(d[kernel_inds]), covariances...)
		if var
			σ²[ind] = gp_var(Kf(0), covariances...)
		end

	end

	meanvar = var ? (f̄, σ²) : f̄
	condnum ? (meanvar..., cond(sum(covariances[2:3]))) : meanvar
end

function interp_knn(d, Kf, Kn, N; var=false, cond=false, progress=false)
	f̄ = similar(d)
	if var
		σ² = similar(d)
	end

	Δx = fieldinfo(d).Δx
	r = Int(sqrt(N)) ÷ 2 # kernel width
	field_inds = CartesianIndices(d[:Ix])
	kdtree = KDTree(coords(field_inds) .* Δx)
	@showprogress (progress ? 1 : Inf) "Performing convolution..." for ind in field_inds
		x₀ = coords([ind]) .* Δx
		kernel_inds, dvec = knn(kdtree, x₀, N)
		X = coords(field_inds[kernel_inds...]) .* Δx

		covariances = Kf(x₀, X), Kf(X, X), Kn(X, X)
		f̄[ind] = gp_mean(vec(d[kernel_inds...]), covariances...)
		if var
			σ²[ind] = gp_var(Kf(0), covariances...)
		end
	end
	var ? (f̄, σ²) : f̄
end

function interp_imfilter(d::FlatMap{P,T,M}, Kf, Kn, N; var=false, condnum=false) where {P,T,M}
	if var
		σ² = similar(d)
	end

	Δx = fieldinfo(d).Δx
	w  = Int(sqrt(N)); @assert isodd(w) "Kernel width must be odd"

	X  = coords(CartesianIndices((w, w))) .* Δx
	x₀ = X[:, (w ÷ 2) * (1 + w) + 1][:,:]


	C = Kf(X,X) + Kn(X,X)
	kff = Kf(x₀, X)
	kernel = kff * inv(C)

	f̄ = FlatMap{P}(imfilter(d[:Ix], centered(reshape(kernel, w, w)), NA()))

	meanvar = var ? (f̄,  Kf(0) - kff ⋅ (C \ kff')) : f̄
	condnum ? (meanvar..., cond(C)) : meanvar
end

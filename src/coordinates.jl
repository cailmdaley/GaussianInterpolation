export coords, coords!
export axes_centered, split_coords
export IndexArray

const IndexArray = AbstractArray{CartesianIndex{2}}

axes_centered(A) = Tuple((ax .- offset for (ax, offset) in zip(axes(A), size(A) .÷ 2 .+ 1)))
split_coords(X) = @views X[:, 1], X[:,2]
ispoint(x) = size(x) === (1,2)

"""
coords
Returns a 2 x (mn) coordinate matrix given a m x n CartesianIndices-like
"""
# Single-Index
coords(y, x) = [y x]
coords(ind::Tuple) = coords(ind...)
coords(ind::CartesianIndex) = coords(Tuple(ind))
function coords!(X::AbstractMatrix, ind::CartesianIndex)
	X .= Tuple(ind)
	X
end

# CartesianIndices
function coords!(X::AbstractMatrix, inds::IndexArray)
	for (i, ind) in zip(LinearIndices(inds), inds)
		X[i, :] .= Tuple(ind)
	end
	X
end
function coords(inds::IndexArray)
	X = Matrix{Int}(undef, length(inds), 2)
	coords!(X, inds)
end

# Normal Arrays
coords(a::AbstractArray) = coords(CartesianIndices(a))
coords!(X::AbstractMatrix, a::AbstractArray) = coords!(X, CartesianIndices(a))


ℓtoθ(ℓ) = π / ℓ
θtoℓ(θ) = round(Int, π / θ)

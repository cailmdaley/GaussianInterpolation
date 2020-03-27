function coords(inds::Union{CartesianIndices, Array{CartesianIndex}})
	"""
	Returns a 2 x (mn) coordinate matrix given a m x n CartesianIndices-like
	"""
	ind_tuples = vec(Tuple.(inds))
	permutedims(hcat(first.(ind_tuples), last.(ind_tuples)))
end

coords(m::Matrix) = coords(CartesianIndices(m))

ℓtoθ(ℓ) = 2π / ℓ |> u"arcminute"
θtoℓ(θ) = round(Int, 2π / θ |> upreferred)

struct InterpolatedCovariance{T <: Real}
	itp::ScaledInterpolation
	metric::Metric
end

# Base Constructor
function InterpolatedCovariance(Cℓ, θs, T, metric)
  """
	Calculates pixel-space covariances from Cℓs at angular separations θs
	using Legendre polynomials, then constructs a CMBInterpolator from
	these covariances
	"""
	ℓmax  = Cℓ.ℓ[end]
	ℓs    = 2:ℓmax
	coeff = @. 2(ℓs + 1) / (4π) * Cℓ(ℓs)
	Pℓs   = Pl.(0:ℓmax, cos.(θs))[:, 3:end]
	covariances = T.(vec(sum(permutedims(coeff) .* Pℓs, dims=2)))
    
    if typeof(metric) <: Haversine
        θs *= 180 / π
    end
	covariance_itp = scale(interpolate(covariances, BSpline(Linear())), θs)
    
	InterpolatedCovariance{T}(covariance_itp, metric)
end

# Convenience Constructor
function InterpolatedCovariance(Cℓ::InterpolatedCℓs, metric::Metric=Euclidean())
    InterpolatedCovariance(Cℓ, 0:1u"arcminute":π, Float64, metric)
end

# callable for pairwise distances (for covariance matrix construction)
function (itp::InterpolatedCovariance)(a::AbstractArray, b=a)
	itp.itp.(pairwise(itp.metric, a, b, dims=2))
end

# callable for calculating covariance from distance
function (itp::InterpolatedCovariance)(d::Number)
	itp.itp(d)
end

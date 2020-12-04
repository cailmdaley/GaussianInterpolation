using GaussianInterpolation, CMBLensing
using GaussianInterpolation: coords, cov!
using BenchmarkTools
using BenchmarkTools: TrialEstimate, TrialJudgement

function print_benchmark(id, results...)
    (id !== nothing) && printstyled(id, "\n", color=:magenta, bold=true)
    for result in results
        show(stdout, "text/plain", result)
    end
    print("\n\n")
end
function print_benchmark(prefix, groups::Vararg{BenchmarkGroup})
    for group_id in sort([key for key in keys(first(groups))])
        id = (prefix === nothing) ? group_id : join([prefix, group_id], '-')
        print_benchmark(id, [group[group_id] for group in groups]...)
    end
end
print_benchmark(groups::Vararg{BenchmarkGroup}) = print_benchmark(nothing, groups...)

macro loop(body)
  quote
    for i = 1:256^2
      $(esc(body))
    end
  end
end

##
suite = BenchmarkGroup()
suite["coords"] = @benchmarkable coords($rand(256, 256))

m = 20
Nside = 32
@unpack f, f̃, ds, n, ϕ = load_sim_dataset(
    θpix=1, Nside=Nside, T=Float64, μKarcminT=6, pol=:I,
	pixel_mask_kwargs=(num_ptsrcs=10, edge_padding_deg=0));

## 
X = coords(f)
x = X[1:1, :]
K = make_kernel(rand(Nside, Nside))
out_vec = zeros(m)
out_mat = zeros(m, m)

suite["cov! vector"] = @benchmarkable @loop cov!($out_vec, $K,  $X)
suite["cov! matrix"] = @benchmarkable @loop cov!($out_mat, $K, $X)

# tune!(suite); BenchmarkTools.save("benchmark_params.json", params(suite))
loadparams!(suite, BenchmarkTools.load("benchmark_params.json")[1], :evals, :samples);

results = minimum(run(suite))
## 
results_old = BenchmarkTools.load("benchmark.json")[1]
comparison = judge(results, results_old, time_tolerance = 0.05)
print_benchmark(comparison, results)

BenchmarkTools.save("benchmark.json", results)


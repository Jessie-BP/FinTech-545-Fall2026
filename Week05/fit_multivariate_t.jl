# A Worked Fit -- Week 05
#
# Fit a multivariate t to five assets by the rank based route: mu at the sample
# mean, R from Kendall's tau, the scales pinned to nu by the sample standard
# deviations, and nu itself by a one dimensional profile.
#
# This is the fit in the "Fitting the Multivariate t" section. The copula
# comparison that follows it is in choosing_between_copulas.jl.

using DataFrames
using CSV
using Distributions
using StatsBase
using LinearAlgebra
using LoopVectorization
using Random
using JuMP
using Ipopt
using Printf

include("../library/return_calculate.jl")
include("../library/fitted_model.jl")
include("../library/simulate.jl")
include("../library/multivariate_t.jl")

stocks = ["SPY", "AAPL", "MSFT", "JPM", "XOM"]

prices = CSV.read("DailyPrices.csv", DataFrame)
returns = return_calculate(prices, dateColumn="Date")[!, stocks]

X = Matrix(returns)
m, n = size(X)

# --- Step 1: the sample moments ------------------------------------------
mu = vec(mean(X, dims=1))
sd = vec(std(X, dims=1))

# A t fitted to each series on its own. Not used by the multivariate fit --
# it is there to show what one shared nu has to stand in for.
println("\nSample moments, and a t fitted to each series separately")
@printf("  %-6s %9s %9s %12s\n", "Asset", "Mean (%)", "Sd (%)", "Separate nu")
for (j, s) in enumerate(stocks)
    fm = fit_general_t(X[:, j])
    @printf("  %-6s %9.3f %9.3f %12.2f\n",
            s, 100 * mu[j], 100 * sd[j], dof(fm.errorModel.ρ))
end

# --- Step 2: R from Kendall's tau ----------------------------------------
# rho = sin(pi*tau/2) is exact for any elliptical distribution, and nu does not
# appear in it, so R can be estimated once and held fixed while nu is searched.
tau = corkendall(X)
R = sin.(pi .* tau ./ 2)
for i in 1:n
    R[i, i] = 1.0
end

# The tau matrix is PSD on complete data. The entrywise sin() transform is not
# guaranteed to preserve that, so check and repair. fix_correlation does the
# repair; the eigenvalue is reported here so the reader sees whether it fired.
minEig = min_eigenvalue(R)
if minEig < -1e-8
    @printf("R is not PSD (min eigenvalue %.6f). Repairing.\n", minEig)
    R = fix_correlation(R)
else
    @printf("\nR is PSD. Min eigenvalue %.4f\n", minEig)
end



# --- Step 3: profile nu ---------------------------------------------------
# A multivariate t carries a scale matrix, not a covariance. cov = nu/(nu-2)*S,
# so at a given nu the scale of variable i has to be sd_i*sqrt((nu-2)/nu), and
# every parameter of the distribution becomes a function of nu alone.
mvt_ll(nu) = mvt_loglikelihood(X, mu, sd, R, nu)

# Search on theta = 1/nu so the resolution goes where it matters. The interval
# stops at 0.49 rather than 0.5 to keep nu off its lower bound, where the scale
# above collapses to zero and S stops being invertible.
nuhat, _ = profile_nu(mvt_ll)

println("\nThe log likelihood as a function of nu")
for nu in [4.0, 6.0, nuhat, 10.0, 15.0, 20.0, 40.0]
    @printf("  nu = %5.1f   ll = %9.1f%s\n", nu, mvt_ll(nu), nu == nuhat ? "   <- max" : "")
end

@printf("\nnu = %.1f  (m = %d observations, n = %d variables)\n", nuhat, m, n)

# --- Simulating from the fit ---------------------------------------------
# The mixture representation: one chi-square draw scales the whole normal
# vector, and that common shock is what puts weight in the joint tail.
S = mvt_scale(sd, R, nuhat)
sim = simulate_mvt(mu, S, nuhat, 100000)

# Compare against the package's own multivariate t. If these agree, use
# whichever you are more comfortable with.
Random.seed!(99)
sim2 = Matrix(rand(MvTDist(nuhat, mu, Matrix(S)), 100000)')

println("\nMixture simulation against the built in draw, 100,000 iterations")
@printf("  %-6s %10s %10s %10s %10s\n", "Asset", "sd mix", "sd built", "kurt mix", "kurt built")
for (jj, s) in enumerate(stocks)
    @printf("  %-6s %10.4f %10.4f %10.2f %10.2f\n", s,
            std(sim[:, jj]), std(sim2[:, jj]),
            kurtosis(sim[:, jj]), kurtosis(sim2[:, jj]))
end

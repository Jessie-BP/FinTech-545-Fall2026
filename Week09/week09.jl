# =====================================================================
# Week 09 -- Factor Models: run every worked example
#
# The worked example is split into standalone scripts, one per item in
# Section 10 of the notes.  Each can be run on its own, e.g.
#
#     julia --project=Week09 Week09/week09_02_ff3_time_series.jl
#
# This file just runs all of them in order.  Each script is loaded into its
# own module so that variables from one (R, B, Σ, ...) cannot leak into the
# next -- exactly as if you had run them one at a time.
#
#   week09_common.jl                    shared data loading and model fits
#   week09_01_singular_covariance.jl    Notes 1     rank 59, null-space portfolio
#   week09_02_ff3_time_series.jl        Notes 4.1   FF3 by stacked OLS
#   week09_03_pca_statistical.jl        Notes 4.3   PCA model, scree, comparison
#   week09_04_cross_sectional.jl        Notes 4.2   industry model, Fama-MacBeth
#   week09_05_woodbury.jl               Notes 5.4   Woodbury identity and timing
#   week09_06_ex_ante_risk.jl           Notes 6-7   exposures, VaR/ES, tracking error
#   week09_07_portfolio_construction.jl Notes 8     min variance, factor neutral
#   week09_08_residual_correlation.jl   Notes 9.3   is D really diagonal?
# =====================================================================

scripts = [
    "week09_01_singular_covariance.jl",
    "week09_02_ff3_time_series.jl",
    "week09_03_pca_statistical.jl",
    "week09_04_cross_sectional.jl",
    "week09_05_woodbury.jl",
    "week09_06_ex_ante_risk.jl",
    "week09_07_portfolio_construction.jl",
    "week09_08_residual_correlation.jl",
]

for (i, s) in enumerate(scripts)
    println("\n", "="^70, "\n", s, "\n", "="^70)
    path = joinpath(@__DIR__, s)
    Core.eval(Main, :(module $(Symbol("Script", i)); include($path); end))
end

# =====================================================================
# Week 09 -- Script 8: Is D really diagonal?  The residual correlation test
#
# Notes, Section 9.3 "Residual Correlation, Testing the Diagonal D Assumption"
#   -> Table "Largest FF3 residual correlations"
#   -> mean |residual corr|, FF3 vs cross-sectional, against the noise floor
#
# Assumption 3 of the factor model (Notes 2.2) says cov(ε_i, ε_j) = 0 for
# i ≠ j.  That is what makes D diagonal and collapses the parameter count.
# It is also directly testable: compute the correlation matrix of the
# residuals and look at the off-diagonal entries.
#
# Run from anywhere:   julia --project=Week09 Week09/week09_08_residual_correlation.jl
# =====================================================================

include("week09_common.jl")

returns, stocks, _ = load_returns()
n  = length(stocks)
ff = fit_ff3(returns, stocks)                    # Script 2
xs = fit_cross_sectional(ff.d, stocks)    # Script 4
T, m = ff.T, ff.m

# ---------------------------------------------------------------------
# All residual correlations, sorted
# ---------------------------------------------------------------------
# cor(E) is n x n.  Only the upper triangle (i < j) is distinct: 4,950 pairs.
# Keep (|corr|, i, j) so we can sort by size and still know which stocks.
Ec = cor(ff.E)
Ex = cor(xs.E)
pairs = [(abs(Ec[i, j]), i, j) for i in 1:n for j in (i+1):n]
sort!(pairs, rev = true)

# ---------------------------------------------------------------------
# The average: what pure noise looks like
# ---------------------------------------------------------------------
# Even if the true residual correlation were exactly zero, a sample
# correlation from T-m-1 = 56 effective observations has standard deviation
# about 1/√56 ≈ 0.134.  The expected ABSOLUTE value of a N(0, s²) draw is
# s·√(2/π) ≈ 0.8 s ≈ 0.107.  That is the noise floor.
noise = 1 / sqrt(T - m - 1)
mean_abs(C) = mean(abs(C[i, j]) for i in 1:n for j in (i+1):n)
println("=== Average residual correlation ===")
@printf("mean |residual corr|: FF3 %.3f   cross-sectional %.3f   noise floor ≈ %.3f\n",
        mean_abs(Ec), mean_abs(Ex), sqrt(2 / pi) * noise)
# Both sit above the floor, and the industry factors take out part of the
# excess.  But an average over 4,950 pairs cannot say WHICH pairs are the
# problem, or whether they matter for a given portfolio.
println("The average says there is excess correlation, not where it is.")

# ---------------------------------------------------------------------
# The tail: where the signal is
# ---------------------------------------------------------------------
# Look at the largest pairs and ask whether they make ECONOMIC sense.
# Random noise would give random pairs.  What we get are share classes of
# the same company and same-industry pairs: a missing industry factor.
#
# The second column re-runs the check on the cross-sectional model's
# residuals (market + industry factors).  Pairs in the same broad sector
# should fall; pairs that are the same company, or a narrower sub-industry
# than our 8 sectors, will not.
println("\n=== Largest FF3 residual correlations ===")
@printf("  %-7s %-7s %10s %16s   %s\n", "", "", "FF3", "+ industries", "sectors")
for (v, i, j) in pairs[1:10]
    si, sj = SECTORS[stocks[i]], SECTORS[stocks[j]]
    @printf("  %-7s %-7s %+10.3f %+16.3f   %s\n",
            stocks[i], stocks[j], Ec[i, j], Ex[i, j], si == sj ? si : "$si / $sj")
end
println("\nDual share classes, and industry pairs the FF3 model has no factor for.")
println("A model's real test is its risk forecast -- see Script 7.")

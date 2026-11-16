# =====================================================================
# Week 09 -- Script 7: Portfolio construction with factor covariances
#
# Notes, Section 8.1 "Optimization with a Factor Covariance"
#   -> Table "Minimum-variance portfolios from four covariance estimates"
# Notes, Section 8.2 "Factor-Neutral Construction"
#   -> Table "Cost of imposing factor neutrality"
#
# Minimum variance needs Σ⁻¹1.  For the sample covariance (rank 59) that
# does not exist.  For every factor model it does.  Here we solve the same
# problem with three factor covariances and compare predicted risk to what
# the portfolio actually did.
#
# Run from anywhere:   julia --project=Week09 Week09/week09_07_portfolio_construction.jl
# =====================================================================

include("week09_common.jl")

returns, stocks, _ = load_returns()
n  = length(stocks)
ff = fit_ff3(returns, stocks)                    # Script 2
pc = fit_pca(cov(Matrix(ff.d[!, stocks])), 3)    # Script 3
xs = fit_cross_sectional(ff.d, stocks)    # Script 4

# Daily returns on the dates all three models were fit on.
R = Matrix(ff.d[!, stocks])

# ---------------------------------------------------------------------
# Minimum variance:  w* = Σ⁻¹1 / 1'Σ⁻¹1      (minvar() in week09_common.jl)
# ---------------------------------------------------------------------
# "Predicted" is the model's own forecast, sqrt(w'Σw).
# "Realized" is the standard deviation of the portfolio's actual daily
# returns, R*w.  Both are measured over the same 60 days the models were
# estimated on, so this is an IN-SAMPLE check -- which makes the FF3 and PCA
# misses below even more telling: the models cannot reproduce the risk of
# the very data they were fit to.
#
# Also report max weight and gross exposure Σ|w_i|.  A sane covariance
# gives a sane portfolio; a broken one gives extreme long/short bets.
println("=== Minimum-variance portfolios (Notes 8.1) ===")
@printf("%-10s %10s %10s %8s %8s %7s\n", "", "predicted", "realized", "max w", "min w", "gross")
report(nm, Σ, wv) = @printf("%-10s %9.2f%% %9.2f%% %7.1f%% %7.1f%% %7.2f\n",
                            nm, annvol(wv' * Σ * wv), std(R * wv) * sqrt(ANN) * 100,
                            maximum(wv) * 100, minimum(wv) * 100, sum(abs.(wv)))

# The sample covariance first.  It has no inverse (rank 59), so use the
# pseudo-inverse with Julia's default tolerance, as in Script 1.  Two
# things to notice:
#   - predicted = realized EXACTLY.  The sample covariance of these 60 days
#     is, by definition, the covariance of these 60 days, so an in-sample
#     check can never catch it out.  Script 1's 30/30 split is the honest test.
#   - this is not even the true minimizer.  pinv discards the null space,
#     and under S the true minimum variance is zero (any null-space
#     portfolio).  The weights are an artifact of the cutoff.
S = cov(R)
u = pinv(S) * ones(n)
report("Sample†", S, u ./ sum(u))

for (nm, Σ) in [("FF3", ff.Σ), ("PCA-3", pc.Σ), ("Cross-sec", xs.Σ)]
    report(nm, Σ, minvar(Σ))
end
println("† pseudo-inverse; Σ⁻¹ does not exist")

# Why FF3 and PCA-3 under-predict: their D is diagonal, so they think the
# residuals of, say, AMAT and LRCX are independent.  The optimizer diversifies
# against that "independent" noise and believes it has removed it.  It has
# not -- the residuals are correlated by industry (Script 8).  The
# cross-sectional model has industry factors, gives that co-movement
# somewhere to live, and its forecast does not understate realized risk.
# It overstates it a little: the OLS->GLS weights are noisy, and the
# industry factors carry some of the residual noise into F.
println("FF3/PCA-3: predicted << realized.  The diagonal-D assumption is failing (Script 8).")
println("Cross-sectional: predicted >= realized.  Industry factors capture the co-movement.")

# ---------------------------------------------------------------------
# Factor-neutral minimum variance  (Notes 8.2)
# ---------------------------------------------------------------------
#   min w'Σw   s.t.  1'w = 1  and  B'w = 0
# Stack the constraints:  A'w = c  with  A = [1  B]  (n x (m+1)) and
# c = [1, 0, 0, 0].  The Lagrangian solution is
#   w* = Σ⁻¹A (A'Σ⁻¹A)⁻¹ c
# a direct generalization of the minimum variance formula, which is the
# special case A = 1, c = 1.
println("\n=== Factor-neutral construction (Notes 8.2) ===")
B, m = ff.B, ff.m
A  = hcat(ones(n), B)
c  = vcat(1.0, zeros(m))
Σi = inv(ff.Σ)                   # fine at n = 100; use Woodbury (Script 5) at scale
wn = Σi * A * ((A' * Σi * A) \ c)

wmv = minvar(ff.Σ)               # unconstrained MVP for comparison
@printf("%-22s %16s %16s\n", "", "unconstrained", "factor-neutral")
@printf("%-22s %15.2f%% %15.2f%%\n", "annualized vol",
        annvol(wmv' * ff.Σ * wmv), annvol(wn' * ff.Σ * wn))
@printf("%-22s %16s %16s\n", "B'w",
        string(round.(B' * wmv, digits = 3)), string(round.(B' * wn, digits = 3)))
@printf("%-22s %16.2f %16.2f\n", "gross exposure", sum(abs.(wmv)), sum(abs.(wn)))

# Confirm the constraints hold to machine precision.
@printf("constraint check: |sum(w) - 1| = %.1e   max |B'w| = %.1e\n",
        abs(sum(wn) - 1), maximum(abs.(B' * wn)))
@printf("cost of neutrality: %.0f bp of annualized volatility\n",
        (annvol(wn' * ff.Σ * wn) - annvol(wmv' * ff.Σ * wmv)) * 100)

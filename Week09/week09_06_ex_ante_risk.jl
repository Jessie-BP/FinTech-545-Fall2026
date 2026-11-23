# =====================================================================
# Week 09 -- Script 6: Exposures, variance decomposition, VaR, tracking error
#
# Notes, Section 6 "Systematic and Idiosyncratic Risk"
#   -> Table "Idiosyncratic risk at the asset level and the portfolio level"
#   -> total vol = sqrt(systematic² + idiosyncratic²)
# Notes, Section 7 "Ex-Ante Risk"
#   -> Table "Equal-weight portfolio exposures"  (7.1)
#   -> VaR and ES under normality                (7.3)
#   -> Table "Tracking error decomposition"      (7.4)
#
# Everything here uses the FF3 model from Script 2.  The key idea: once you
# have Σ = BFB' + D, a 100-stock portfolio is described by m = 3 numbers,
# its factor exposures x = B'w, plus one idiosyncratic number.
#
# Run from anywhere:   julia --project=Week09 Week09/week09_06_ex_ante_risk.jl
# =====================================================================

include("week09_common.jl")

returns, stocks, _ = load_returns()
ff = fit_ff3(returns, stocks)
B, Fcov, Dv, fnames = ff.B, ff.Fcov, ff.Dv, ff.fnames
n = length(stocks)

# ---------------------------------------------------------------------
# Portfolio exposures  x = B'w   (Notes 7.1)
# ---------------------------------------------------------------------
# B is n x m, so B'w is m x 1: the weighted average of each stock's beta.
# This compresses 100 weights into 3 numbers a PM can reason about.
w = fill(1.0 / n, n)             # equal-weighted portfolio
x = B' * w
println("=== Exposures (Notes 7.1) ===")
@printf("exposures: Mkt %+.3f  SMB %+.3f  HML %+.3f\n", x...)

# ---------------------------------------------------------------------
# Systematic vs idiosyncratic variance  (Notes 6)
# ---------------------------------------------------------------------
#   σ_p² = w'Σw = w'BFB'w + w'Dw = x'Fx + Σ_i w_i² d_i
# The second term never needs Σ: D is diagonal, so it is just a weighted sum.
sys   = x' * Fcov * x
idio  = sum(w .^ 2 .* Dv)
tot_v = sys + idio
@assert tot_v ≈ w' * ff.Σ * w    # the decomposition is exact

println("\n=== Variance decomposition (Notes 6.1) ===")
# Vols add in QUADRATURE (square, add, square-root), not linearly.  That
# is why a small idiosyncratic piece barely moves the total.
@printf("ann vol total %.2f%%  = sqrt( systematic %.2f%%² + idio %.2f%%² )\n",
        annvol(tot_v), annvol(sys), annvol(idio))

# The table: idiosyncratic share for a single stock vs the portfolio.
# A single stock's idiosyncratic share is 1 - R².  For the portfolio, each
# d_i is multiplied by w_i² = 1/n², so the term falls like 1/n
# ("diversification eliminates specific risk but not market risk").
iaapl = findfirst(==(:AAPL), stocks)
@printf("idiosyncratic share:  typical stock %.1f%%   AAPL %.1f%%   EW portfolio %.1f%%\n",
        mean(1 .- ff.R2) * 100, (1 - ff.R2[iaapl]) * 100, idio / tot_v * 100)

# ---------------------------------------------------------------------
# Factor risk attribution  (Notes 7.1 / 7.2)
# ---------------------------------------------------------------------
# Euler decomposition, Week 08 applied in factor space.  The share of
# variance from factor k is  x_k (Fx)_k / σ_p².  (Fx)_k includes the
# covariance of factor k with every other factor, so a factor that is
# negatively correlated with the market can REDUCE total risk and get a
# negative share.  The shares (plus idiosyncratic) sum to 100%.
println("\n=== Factor risk attribution ===")
mc = Fcov * x
for (j, nm) in enumerate(fnames)
    @printf("  %-8s share of variance %+7.2f%%\n", nm, x[j] * mc[j] / tot_v * 100)
end
@printf("  %-8s share of variance %+7.2f%%\n", "Idio", idio / tot_v * 100)

# Why SMB and HML come out negative.  The sign of a factor's share is the
# sign of x_k * (Fx)_k, and with a market exposure near 1, (Fx)_k is driven
# mostly by factor k's covariance with the market.
#   - SMB: the portfolio is SHORT SMB (x < 0), and SMB moved WITH the market
#     over this window (corr > 0).  The short SMB tilt offsets market risk.
#   - HML: the portfolio is LONG HML (x > 0), and HML moved AGAINST the
#     market (corr < 0).  Again an offset.
# Either way the tilt acts as a partial hedge on the market exposure.
Fcor = cor(ff.Fr)
@printf("corr(SMB, Mkt) = %+.2f   corr(HML, Mkt) = %+.2f   corr(SMB, HML) = %+.2f\n",
        Fcor[2, 1], Fcor[3, 1], Fcor[3, 2])

# ---------------------------------------------------------------------
# Parametric VaR and ES under normality  (Notes 7.3)
# ---------------------------------------------------------------------
# Model the portfolio's daily return as N(0, σ_p²), with σ_p from the
# factor model, and hand that distribution to the library's VaR and ES
# (library/RiskStats.jl).  Given a distribution rather than a vector of
# returns, they compute
#   VaR = -quantile(d, α)                     = 1.645 σ_p
#   ES  = -(1/α) ∫_{-∞}^{-VaR} x f(x) dx      = φ(z_0.05)/0.05 σ_p
# The only thing new this week is where σ_p came from.  Problem 2 replaces
# the normal with a simulation through the factor structure -- pass the
# simulated P&L vector to the same VaR/ES functions.
println("\n=== 1-day VaR / ES, normal (Notes 7.3) ===")
sd = sqrt(tot_v)
dP = Normal(0.0, sd)
@printf("daily vol %.3f%%   VaR(5%%) = %.3f%%   ES(5%%) = %.3f%%\n",
        sd * 100, VaR(dP; alpha = 0.05) * 100, ES(dP; alpha = 0.05) * 100)

# ---------------------------------------------------------------------
# Tracking error against a benchmark  (Notes 7.4)
# ---------------------------------------------------------------------
# Benchmark: the same 100 stocks, cap weighted (float-adjusted caps from
# data/constituents.csv) -- the S&P 500 restricted to its 100 largest names.
# Our equal-weight portfolio underweights the mega caps and overweights
# everything else.
# Active weights a = w - b.  Everything above applies to a instead of w:
#   TE² = (B'a)'F(B'a)  +  Σ a_i² d_i
#         factor bets       stock selection
cap   = mktcap(stocks)
bench = cap ./ sum(cap)
a  = w .- bench
ax = B' * a                      # active exposures: the factor bets vs the benchmark
te_f = sqrt(ax' * Fcov * ax * ANN)
te_s = sqrt(sum(a .^ 2 .* Dv) * ANN)
println("\n=== Tracking error (Notes 7.4) ===")
@printf("TE %.2f%% = sqrt( factor bets %.2f%%² + stock selection %.2f%%² )\n",
        sqrt(te_f^2 + te_s^2) * 100, te_f * 100, te_s * 100)
@printf("active exposures: Mkt %+.3f  SMB %+.3f  HML %+.3f\n", ax...)
@printf("stock selection is %.0f%% of TE², factor bets %.0f%%\n",
        te_s^2 / (te_f^2 + te_s^2) * 100, te_f^2 / (te_f^2 + te_s^2) * 100)

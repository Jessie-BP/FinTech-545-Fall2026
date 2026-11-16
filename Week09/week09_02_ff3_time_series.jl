# =====================================================================
# Week 09 -- Script 2: A time-series factor model (Fama-French 3)
#
# Notes, Section 4.1 "Time-Series Models"
#   -> Table "FF3 betas, selected stocks, 60 daily observations"
#   -> mean R² and its range across the 100 stocks
#   -> beta standard errors ("Full sample vs rolling betas")
# Notes, Section 5 "The Factor Covariance Matrix"
#   -> rank 100, PD, condition number 343, 406 parameters
#
# In a time-series model the FACTOR RETURNS ARE OBSERVED (we download them
# from Ken French) and the BETAS ARE ESTIMATED.  One regression per stock,
# run over time.
#
# Run from anywhere:   julia --project=Week09 Week09/week09_02_ff3_time_series.jl
# =====================================================================

include("week09_common.jl")

returns, stocks, _ = load_returns()
n = length(stocks)

# ---------------------------------------------------------------------
# Align the stock returns with the factor returns
# ---------------------------------------------------------------------
# innerjoin keeps only dates present in BOTH files, so row t of the stock
# matrix and row t of the factor matrix refer to the same day.  A
# misalignment of even one day destroys the regression.
ff3 = load_ff3()                     # already converted from percent to decimal
d = sort!(innerjoin(returns, ff3, on = :Date), :Date)

fnames = [:Mkt_RF, :SMB, :HML]
m = length(fnames)

# The FF factors are excess returns (Mkt-RF) or long-short spreads (SMB,
# HML), so the left-hand side should be an EXCESS return too: subtract the
# risk-free rate.  Otherwise RF leaks into the intercept α.
Y  = Matrix(d[!, stocks]) .- d.RF    # T x n  excess stock returns
Fr = Matrix(d[!, fnames])            # T x m  factor returns  (the "F" in R = FB' + E)
T  = size(Y, 1)

# ---------------------------------------------------------------------
# Stacked OLS:  Θ = (X'X)⁻¹ X'R
# ---------------------------------------------------------------------
# Every stock is regressed on the SAME right-hand side X = [1  F].  So the
# expensive part, (X'X)⁻¹, is the same for all 100 regressions.  Putting all
# 100 return series side by side as the columns of Y lets one matrix solve
# do all 100 regressions at once.
#
# We use the backslash (X'X) \ (X'Y) rather than inv(X'X) * X'Y; it solves
# the linear system directly and is more accurate.
X = hcat(ones(T), Fr)                # T x (m+1)
Θ = (X' * X) \ (X' * Y)              # (m+1) x n:  row 1 = α',  rows 2..m+1 = B'
α = Θ[1, :]                          # n intercepts
B = Matrix(Θ[2:end, :]')             # n x m  exposures -- one row per stock
E = Y .- X * Θ                       # T x n  residuals ε

# ---------------------------------------------------------------------
# The ingredients of Σ = BFB' + D
# ---------------------------------------------------------------------
# F: covariance of the m factor returns.  3 x 3, estimated from 60 days --
#    plenty of data for 6 numbers.
Fcov = cov(Fr)

# D: the specific (idiosyncratic) variances, one per stock.  Each residual
#    series came from a regression that estimated m+1 = 4 coefficients, so it
#    has T - m - 1 = 56 degrees of freedom, not 60.  Dividing by T would bias
#    every d_i down by 4/60 ≈ 7%, and the whole risk model with it.
Dv = vec(sum(E .^ 2, dims = 1)) ./ (T - m - 1)
@printf("degrees of freedom T-m-1 = %d;  dividing by T instead would understate D by %.1f%%\n",
        T - m - 1, (1 - (T - m - 1) / T) * 100)

# Assemble the factor covariance matrix.  Diagonal(Dv) is a lazy diagonal
# matrix -- Julia never stores the 9,900 zeros.
Σ_ff = B * Fcov * B' + Diagonal(Dv)

# ---------------------------------------------------------------------
# How much does the model explain?  (R² per stock)
# ---------------------------------------------------------------------
# Each stock's variance splits into a systematic piece β_i'Fβ_i and a
# specific piece d_i (Notes 6).  R² is the systematic share.
println("\n=== FF3 time-series model ===")
sysvar = [B[i, :]' * Fcov * B[i, :] for i in 1:n]
R2 = sysvar ./ (sysvar .+ Dv)
@printf("mean R2 = %.3f   min = %.3f (%s)   max = %.3f (%s)\n",
        mean(R2), minimum(R2), stocks[argmin(R2)], maximum(R2), stocks[argmax(R2)])
@printf("Three factors explain %.0f%% of a typical stock's daily variance (median R2 %.2f).\n",
        mean(R2) * 100, median(R2))

# ---------------------------------------------------------------------
# The betas for the stocks in the notes' table
# ---------------------------------------------------------------------
# Read the signs against what you know about each company: growth names
# (NVDA, MSFT) load negatively on HML, XOM and JNJ positively.  Over this
# 60-day window XOM even has a NEGATIVE market beta -- the market was
# driven by technology while energy went its own way.  Sixty days of data
# describe sixty days, not the company.
betas = DataFrame(:Stock => String.(stocks), :Mkt => B[:, 1], :SMB => B[:, 2],
                  :HML => B[:, 3], :R2 => R2, :AnnSD => sqrt.((sysvar .+ Dv) .* ANN))
table_names = ["TSLA", "MSFT", "NVDA", "CSCO", "AAPL", "BRK-B", "JNJ", "XOM"]
sel = betas[indexin(table_names, betas.Stock), :]
println("\nSelected stocks (the table in Notes 4.1):")
for r in eachrow(sel)
    @printf("  %-6s  Mkt %+6.3f  SMB %+6.3f  HML %+6.3f  R2 %.2f  ann sd %5.1f%%\n",
            r.Stock, r.Mkt, r.SMB, r.HML, r.R2, r.AnnSD * 100)
end

# ---------------------------------------------------------------------
# How precise are these betas?  ("Full sample vs rolling betas")
# ---------------------------------------------------------------------
# The OLS covariance of the coefficients for stock i is d_i (X'X)⁻¹.  The
# [2,2] element of (X'X)⁻¹ belongs to the market beta.  This is the exact
# version of the rule of thumb  se(β) ≈ σ_ε / (σ_f √T)  in the notes.
XtXi = inv(X' * X)
se_mkt = sqrt.(Dv .* XtXi[2, 2])
@printf("\nmarket-beta standard error: median %.2f  (range %.2f to %.2f)\n",
        median(se_mkt), minimum(se_mkt), maximum(se_mkt))
@printf("A beta of 1.15 vs 0.85 is %.1f median standard errors apart.\n", 0.30 / median(se_mkt))

# ---------------------------------------------------------------------
# Is the factor covariance matrix well behaved?  (Notes 5.2 and 5.3)
# ---------------------------------------------------------------------
# Compare to Script 1: the sample covariance had rank 59 and λmin ≈ 0.
# Here BFB' has rank 3, but adding a strictly positive D lifts every
# eigenvalue to at least min(d_i).  Full rank, PD, invertible -- for ANY T.
ev_ff = eigvals(Symmetric(Σ_ff))
println("\n=== Properties of Σ = BFB' + D ===")
@printf("rank = %d   min eig = %.3e   min d_i = %.3e   cond = %.1f   isposdef = %s\n",
        rank(Σ_ff), minimum(ev_ff), minimum(Dv), maximum(ev_ff) / minimum(ev_ff),
        isposdef(Symmetric(Σ_ff)))

# Parameter count (Notes 5.1):  nm for B, m(m+1)/2 for F, n for D.
@printf("parameters: %d = %d (B) + %d (F) + %d (D)   vs %d for the sample covariance\n",
        n * m + m * (m + 1) ÷ 2 + n, n * m, m * (m + 1) ÷ 2, n, n * (n + 1) ÷ 2)

# The same fit is packaged as fit_ff3() in week09_common.jl, which the later
# scripts use.  Sanity check that the two agree:
@assert fit_ff3(returns, stocks).Σ ≈ Σ_ff

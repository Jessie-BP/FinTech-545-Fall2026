# =====================================================================
# Week 09 -- Script 4: A cross-sectional (fundamental) factor model
#
# Notes, Section 4.2 "Cross-Sectional (Fundamental) Models"
#   -> The regression  f̂_t = (B'WB)⁻¹ B'W r_t, one per date
#   -> W = D⁻¹ in two steps: OLS -> residuals -> D -> GLS
#   -> Identification: the sum-to-zero constraint and the KKT system
#   -> Fama-MacBeth inference
#   -> Table "Cross-sectional factor returns, 60 days"
#
# This is the time-series model turned around.  The EXPOSURES ARE OBSERVED
# (here: industry membership, a 0/1 dummy) and the FACTOR RETURNS ARE
# ESTIMATED, one cross-sectional regression per day.
#
# Run from anywhere:   julia --project=Week09 Week09/week09_04_cross_sectional.jl
# =====================================================================

include("week09_common.jl")

returns, stocks, _ = load_returns()
n  = length(stocks)
Rx = Matrix(returns[!, stocks])      # raw (not excess) returns: the market factor absorbs RF
T  = size(Rx, 1)

# ---------------------------------------------------------------------
# The exposure matrix B: a market column plus industry dummies
# ---------------------------------------------------------------------
# These exposures are DATA.  No regression is needed to know JPM is a bank.
# SECTORS (in week09_common.jl) maps each ticker to one of 8 groups.
industries = sort(unique(getindex.(Ref(SECTORS), stocks)))
Bx = hcat(ones(n),                                         # market: every stock has exposure 1
          [SECTORS[s] == g ? 1.0 : 0.0 for s in stocks, g in industries])   # n x 8 dummies
xnames = vcat(["Market"], industries)
mx = size(Bx, 2)                                           # 1 + 8 = 9 factors

# ---------------------------------------------------------------------
# Identification (Notes 4.2, "Identification")
# ---------------------------------------------------------------------
# Every stock is in exactly one industry, so the 8 dummy columns add up to
# the market column of ones.  The columns of B are linearly dependent, B is
# rank deficient, and B'WB has no inverse.  The regression cannot tell
# "market up 1%" apart from "every industry up 1%".
@printf("rank(B) = %d with %d columns  ->  B'WB is singular\n", rank(Bx), mx)

# The fix: add the constraint  Σ_k c_k f_k = 0.  The industry factors
# become RELATIVE effects -- each one is measured against a c-weighted
# "average industry" -- and the market factor carries the common move.
# As a row vector C (1 x mx), the constraint is C*f = 0 with a 0 on the
# market and c_k on each industry.  Three common choices of c_k
# (Notes 4.2, "Identification"):
#   equal : c_k = 1     -> every industry counts once, whatever its size
#   count : c_k = n_k   -> every STOCK counts once (big industries count more)
#   cap   : c_k = industry share of market cap  -> every DOLLAR counts once
# Our model uses equal (Notes 4.2).  The industry sizes show how different "equal" and "count" are:
nk = vec(sum(Bx[:, 2:end], dims = 1))
println("stocks per industry: ",
        join([@sprintf("%s %d", g, k) for (g, k) in zip(industries, Int.(nk))], ", "))
C = reshape(vcat(0.0, ones(length(industries))), 1, mx)

# ---------------------------------------------------------------------
# One constrained regression per date, via the KKT system
# ---------------------------------------------------------------------
# For a given weighting matrix W, each date's regression is
#     min (r_t - B f)'W(r_t - B f)   subject to   C f = 0
# and its solution satisfies the KKT system from the notes:
#   [ B'WB  C' ] [ f_t ]   [ B'W r_t ]
#   [ C     0  ] [ λ   ] = [ 0       ]
#
# The left-hand matrix does not depend on t, so invert it ONCE and reuse it
# for all 60 days.  Only the right-hand side changes from day to day.  The
# last element of the solution is the Lagrange multiplier λ, which we drop.
#
# We need this regression twice (OLS, then GLS), so wrap it in a function
# of W.  It returns the KKT inverse too, for the mimicking portfolios below.
function cross_section(W)
    KKTi = inv([Bx' * W * Bx  C'; C  zeros(1, 1)])
    Fhat = zeros(T, mx)              # row t = estimated factor returns on day t
    for t in 1:T
        Fhat[t, :] = (KKTi * vcat(Bx' * W * Rx[t, :], 0.0))[1:mx]
    end
    E = Rx .- Fhat * Bx'             # residuals: what the industry model leaves over
    return Fhat, E, KKTi
end

# ---------------------------------------------------------------------
# Step 1: OLS  (W = I)
# ---------------------------------------------------------------------
# GLS wants W = D⁻¹, but D comes from the residuals, and the residuals come
# from the regression.  Break the circle by running the regression first
# with equal weights.  OLS is unbiased; it is just inefficient, because it
# trusts a noisy stock like SNDK as much as a quiet one like PEP.
Fhat_ols, E_ols, _ = cross_section(I(n))

# Each stock's specific variance from the OLS residuals.  Same formula as
# the final D below.
D_ols = vec(sum(E_ols .^ 2, dims = 1)) ./ (T - mx)

# How unequal are the stocks?  If the specific vols were all similar, GLS
# would barely differ from OLS.  They are not.
sv = sqrt.(D_ols .* ANN) .* 100
@printf("OLS specific vol: min %.1f%% (%s)  median %.1f%%  max %.1f%% (%s)\n",
        minimum(sv), stocks[argmin(sv)], median(sv), maximum(sv), stocks[argmax(sv)])
@printf("  -> GLS gives the quietest stock %.0fx the weight of the noisiest\n",
        maximum(D_ols) / minimum(D_ols))

# ---------------------------------------------------------------------
# Step 2: GLS  (W = D⁻¹ from step 1)
# ---------------------------------------------------------------------
# Weight each stock by the inverse of its specific variance.  A stock whose
# returns are mostly idiosyncratic noise tells us little about what the
# industry did today, so it gets little say.  This is the efficient
# estimator when D is right, and D_ols is a consistent estimate of D.
#
# You could repeat: new residuals -> new D -> GLS again.  In practice one
# pass captures nearly all of the gain.
W = Diagonal(1.0 ./ D_ols)
Fhat, Ex, KKTi = cross_section(W)

# How much did reweighting change the factor returns?
@printf("corr(OLS, GLS) market factor return = %.3f\n", cor(Fhat_ols[:, 1], Fhat[:, 1]))

# Check the constraint really holds every day.
@printf("max |sum of industry factor returns| over all days = %.1e\n",
        maximum(abs.(Fhat[:, 2:end] * ones(length(industries)))))

# ---------------------------------------------------------------------
# Factor returns are portfolio returns
# ---------------------------------------------------------------------
# f̂_t = Ω r_t where Ω (the top-left block of KKT⁻¹ times B'W) does not depend
# on t.  Each row of Ω is a set of portfolio weights -- the factor-mimicking
# portfolio.  Its exposure to its own factor is 1:
Ω = KKTi[1:mx, 1:mx] * Bx' * W
@printf("market mimicking portfolio: weights sum to %.4f (unit market exposure)\n",
        sum(Ω[1, :]))

# ---------------------------------------------------------------------
# Fama-MacBeth inference
# ---------------------------------------------------------------------
# We now have 60 estimates of each factor return.  Within a day the
# residuals are highly correlated, so one regression's standard errors are
# useless.  Across days the estimates are close to independent, so treat
# the 60 numbers as a sample and compute an ordinary t-stat on their mean:
#     t = mean(f̂_k) / ( sd(f̂_k) / √T )
println("\n=== Fama-MacBeth: cross-sectional factor returns, $T days ===")
println("Factor                mean(bp)   ann vol    FM t-stat")
tstats = zeros(mx)
for j in 1:mx
    f = Fhat[:, j]
    tstats[j] = mean(f) / (std(f) / sqrt(T))
    @printf("  %-16s %+8.1f  %8.1f%%  %+9.2f\n",
            xnames[j], mean(f) * 1e4, std(f) * sqrt(ANN) * 100, tstats[j])
end
# Volatilities are estimated far more precisely than means (Notes 9.6).
# With 9 factors tested at 5%, about 0.45 false positives are expected by
# chance alone, so one |t| > 2 is not evidence of a premium.  Use this
# model for risk; do not use these means as expected returns.
@printf("%d of %d factors clear |t| > 2; chance alone gives about %.2f.\n",
        count(abs.(tstats) .> 2), mx, 0.05 * mx)

# ---------------------------------------------------------------------
# The covariance matrix  Σ = B F B' + D
# ---------------------------------------------------------------------
# Same structure as FF3; only where B and F come from has changed.
# Residual degrees of freedom: T - mx (mx factor returns were estimated).
Fcx = cov(Fhat)
Dx  = vec(sum(Ex .^ 2, dims = 1)) ./ (T - mx)
Σ_x = Bx * Fcx * Bx' + Diagonal(Dx)
ev  = eigvals(Symmetric(Σ_x))
@printf("\nΣ_x: rank %d   min eig %.2e   cond %.0f\n",
        rank(Σ_x), minimum(ev), maximum(ev) / minimum(ev))

# Packaged as fit_cross_sectional() in week09_common.jl for later scripts.
@assert fit_cross_sectional(returns, stocks).Σ ≈ Σ_x

# =====================================================================
# What is "the market factor"?
# Notes 4.2, "Choosing the weighting matrix" and "The market factor"
# =====================================================================
# Every stock has a market loading of 1, so the market factor is the
# intercept of each day's regression.  Which portfolio it tracks depends on
# two choices: W (how stocks are weighted inside the regression) and C (how
# the industry factors are constrained).

# ---------------------------------------------------------------------
# The closed form for a market + industry-dummy model
# ---------------------------------------------------------------------
# With only industry dummies, the fit for stock i in industry g is
# f_mkt + f_g, and the best that can do is the W-weighted average return
# of industry g:
#     f_mkt,t + f_g,t = m_g,t = Σ_{i∈g} W_ii r_i,t / Σ_{i∈g} W_ii
# The constraint Σ c_k f_k = 0 then pins down the market factor as the
# c-weighted average of those industry averages:
#     f_mkt,t = Σ_k c̃_k m_k,t        with c̃_k = c_k / Σ c_k
# So the market factor is a portfolio, with weight on stock i of
#     ω_i = c̃_g(i) · W_ii / Σ_{j∈g(i)} W_jj
# C sets the weight ACROSS industries; W sets the weight WITHIN each one.
println("\n=== The market factor (Notes 4.2, 'The market factor') ===")
function market_weights(wdiag, c)
    c̃ = c ./ sum(c)
    ω = zeros(n)
    for (k, g) in enumerate(industries)
        idx = [SECTORS[s] == g for s in stocks]
        ω[idx] .= c̃[k] .* wdiag[idx] ./ sum(wdiag[idx])
    end
    return ω
end
ω_mkt = market_weights(diag(W), vec(C[2:end]))      # our model: W = D⁻¹, equal c
@printf("closed form vs KKT market factor: max |difference| = %.1e\n",
        maximum(abs.(Rx * ω_mkt .- Fhat[:, 1])))
@assert ω_mkt ≈ Ω[1, :]                   # the same mimicking portfolio as above

# ---------------------------------------------------------------------
# Market caps
# ---------------------------------------------------------------------
# Float-adjusted market caps from data/constituents.csv (see data/README.md).
# Their ratios are SPY's weights.  wcap is the cap-weighted portfolio of our
# 100 stocks -- the S&P 500 restricted to its 100 largest names.
cap  = mktcap(stocks)
wcap = cap ./ sum(cap)
rcap = Rx * wcap                          # daily return of the cap-weighted 100
spy  = returns.SPY

# Our choices -- W = D⁻¹ and equal c -- give every industry 1/8 of the
# market factor regardless of how many stocks (or how much cap) it has, and
# inside each industry favor the quiet names.  Nothing like an index.
o = sortperm(ω_mkt, rev = true)
println("largest market-factor weights:  ",
        join([@sprintf("%s %.1f%% (cap %.1f%%)", stocks[i], ω_mkt[i] * 100, wcap[i] * 100)
              for i in o[1:5]], ", "))
println("smallest market-factor weights: ",
        join([@sprintf("%s %.2f%% (cap %.1f%%)", stocks[i], ω_mkt[i] * 100, wcap[i] * 100)
              for i in o[end-2:end]], ", "))
oc = sortperm(wcap, rev = true)
println("largest cap weights:            ",
        join([@sprintf("%s %.1f%% (factor %.1f%%)", stocks[i], wcap[i] * 100, ω_mkt[i] * 100)
              for i in oc[1:5]], ", "))

# ---------------------------------------------------------------------
# How W and C change the observed f_mkt time series
# ---------------------------------------------------------------------
# Re-estimate with each combination and compare the market factor's
# return series to the cap-weighted 100 and to SPY.
#   equal c : c_k = 1             -> each industry counts equally
#   count c : c_k = n_k           -> each STOCK counts equally across industries
#   cap c   : c_k = industry cap  -> each DOLLAR counts equally
# The closed form predicts two exact matches:
#   W = I,   count c  -> f_mkt is the equal-weighted average of the 100
#   W = cap, cap c    -> f_mkt is the cap-weighted average of the 100
indcap  = [sum(cap[[SECTORS[s] == g for s in stocks]]) for g in industries]
C_count = reshape(vcat(0.0, nk), 1, mx)          # nk: stocks per industry, from above
C_cap   = reshape(vcat(0.0, indcap ./ sum(cap)), 1, mx)
W_cap   = Diagonal(cap)
W_sqrt  = Diagonal(sqrt.(cap))

combos = [("I", I(n), "equal", C), ("I", I(n), "count", C_count),
          ("D⁻¹", W, "equal", C), ("D⁻¹", W, "count", C_count), ("D⁻¹", W, "cap", C_cap),
          ("√cap", W_sqrt, "cap", C_cap), ("cap", W_cap, "cap", C_cap)]
println("\nW      C       ann vol  corr(cap)  beta(cap)  TE vs cap  corr(SPY)  TE vs SPY")
Σ_by = Dict{Tuple{String,String},Matrix{Float64}}()
fmkt = Dict{Tuple{String,String},Vector{Float64}}()
for (wn, Wv, cn, Cv) in combos
    Fv, Ev = cs_regression(Bx, Cv, Wv, Rx)
    f = Fv[:, 1]
    fmkt[(wn, cn)] = f
    @printf("%-6s %-6s %7.1f%%  %9.3f  %9.3f  %8.2f%%  %9.3f  %8.2f%%\n", wn, cn,
            std(f) * sqrt(ANN) * 100, cor(f, rcap), cov(f, rcap) / var(rcap),
            std(f .- rcap) * sqrt(ANN) * 100, cor(f, spy), std(f .- spy) * sqrt(ANN) * 100)
    Σ_by[(wn, cn)] = Bx * cov(Fv) * Bx' + Diagonal(vec(sum(Ev .^ 2, dims = 1)) ./ (T - mx))
end
@printf("cap-weighted 100:  ann vol %.1f%%  corr(SPY) %.3f  TE vs SPY %.2f%%\n",
        std(rcap) * sqrt(ANN) * 100, cor(rcap, spy), std(rcap .- spy) * sqrt(ANN) * 100)
@printf("closed-form checks: max |f_mkt - EW 100| = %.1e (W = I, count)   max |f_mkt - cap 100| = %.1e (W = cap, cap)\n",
        maximum(abs.(fmkt[("I", "count")] .- vec(mean(Rx, dims = 2)))),
        maximum(abs.(fmkt[("cap", "cap")] .- rcap)))

# √cap with the cap constraint: cap weighted ACROSS industries, √cap weighted
# WITHIN them.  The mega caps get less than their cap weight inside their
# industry, the small names more.
ω_sqrt = market_weights(sqrt.(cap), indcap)
println("√cap market factor vs cap weight: ",
        join([@sprintf("%s %.1f%% vs %.1f%%", stocks[i], ω_sqrt[i] * 100, wcap[i] * 100)
              for i in oc[1:3]], ", "))

# C changes what we CALL the market factor, not the risk model.  B is rank
# deficient, so every C gives the same fitted values B f̂_t, the same
# residuals, and therefore the same Σ.  W is different: it changes the fit.
@printf("max |ΔΣ| changing C only: %.1e (W = I)  %.1e (W = D⁻¹)\n",
        maximum(abs.(Σ_by[("I", "equal")] .- Σ_by[("I", "count")])),
        maximum(max.(abs.(Σ_by[("D⁻¹", "equal")] .- Σ_by[("D⁻¹", "count")]),
                     abs.(Σ_by[("D⁻¹", "equal")] .- Σ_by[("D⁻¹", "cap")]))))
@printf("max |ΔΣ| changing W only: %.1e (I vs D⁻¹)  %.1e (D⁻¹ vs √cap)   (largest Σ entry %.1e)\n",
        maximum(abs.(Σ_by[("I", "equal")] .- Σ_by[("D⁻¹", "equal")])),
        maximum(abs.(Σ_by[("D⁻¹", "cap")] .- Σ_by[("√cap", "cap")])), maximum(Σ_x))

# ---------------------------------------------------------------------
# Each stock's beta to the cap-weighted market
# ---------------------------------------------------------------------
# Our market factor is not the cap-weighted index, so a market loading of 1
# is not a beta of 1 to the index.  For any market portfolio w_m, the
# model's beta of stock i to it is
#     β_i = (Σ w_m)_i / (w_m'Σ w_m)
#         = [ b_i'F x_m + w_m,i d_i ] / [ x_m'F x_m + Σ_j w_m,j² d_j ],   x_m = B'w_m
# where b_i is stock i's row of B.  Here w_m = wcap, the real cap weights.
model_beta(B, F, d, wm) = (x = B' * wm; (B * (F * x) .+ wm .* d) ./ (x' * F * x + sum(wm .^ 2 .* d)))
β_model = model_beta(Bx, Fcx, Dx, wcap)
@assert β_model ≈ (Σ_x * wcap) ./ (wcap' * Σ_x * wcap)  # same thing, without the factor form
@printf("\ncap-weighted beta of the cap-weighted 100: %.4f (must be 1)\n", wcap' * β_model)

# Compare the beta to our market factor itself:  β_i^f = (F b_i)_mkt / F_mkt,mkt.
# The two agree only when the market factor is the cap-weighted index.
β_f = (Bx * Fcx[:, 1]) ./ Fcx[1, 1]

# Historical beta of each stock to the same cap-weighted portfolio.
β_hist = [cov(Rx[:, i], rcap) / var(rcap) for i in 1:n]

# In this model a stock's market loading is 1 and its only other exposure is
# its industry dummy, so b_i'F x_m is IDENTICAL for every stock in an
# industry, and so is β_i^f.  Model betas inside an industry differ only
# through the own-weight term w_m,i d_i, which matters for a large name with
# high specific variance.  The historical betas inside an industry are
# spread widely.  That is the gap a "Beta" style factor fills.
println("beta to the cap-weighted 100, by industry")
println("industry  to mkt factor   model beta range    historical beta range")
for g in industries
    idx = [SECTORS[s] == g for s in stocks]
    @printf("%-9s  %8.2f       %5.2f to %5.2f      %5.2f to %5.2f\n", g, β_f[idx][1],
            minimum(β_model[idx]), maximum(β_model[idx]),
            minimum(β_hist[idx]), maximum(β_hist[idx]))
end
i_hi = argmax(abs.(β_model .- [mean(β_model[[SECTORS[s] == SECTORS[t] for s in stocks]]) for t in stocks]))
@printf("largest own-weight effect: %s (cap %.1f%%, specific vol %.0f%%), model beta %.2f\n",
        stocks[i_hi], wcap[i_hi] * 100, sqrt(Dx[i_hi] * ANN) * 100, β_model[i_hi])

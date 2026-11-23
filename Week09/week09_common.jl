# =====================================================================
# Week 09 -- Factor Models: shared setup
#
# Every week09_0X_*.jl script starts with
#
#     include("week09_common.jl")
#
# so that it can be run on its own.  This file does three things:
#
#   1. Loads the packages every script needs, plus the course library
#      (../library) -- the same functions you have used since Week 03.
#   2. Loads the data (data/, built by data/get_data.py) the same way every time.
#   3. Packages the three model fits from Section 4 of the notes as
#      functions, so that later scripts (risk, portfolio construction,
#      diagnostics) can rebuild a model in one line.
#
# The step-by-step, heavily commented versions of each fit live in their
# own scripts.  Read those first; these functions are the same code
# with the commentary stripped out.
#
#     fit_ff3              <->  week09_02_ff3_time_series.jl   (Notes 4.1)
#     fit_pca              <->  week09_03_pca_statistical.jl   (Notes 4.3)
#     fit_cross_sectional  <->  week09_04_cross_sectional.jl   (Notes 4.2)
#     (cs_regression is the one-regression-per-date step it calls twice)
# =====================================================================

using CSV
using DataFrames
using Dates
using Distributions
using LinearAlgebra
using QuadGK
using Random
using Statistics
using Printf

# Course library.  Paths are relative to THIS file, so they work from any
# working directory.
#   simulate.jl  -> chol_psd!  (Week 03 PSD Cholesky), near_psd, higham_nearestPSD,
#                   simulateNormal, simulate_pca
#   RiskStats.jl -> VaR, ES    (for a sample of returns or for a distribution)
include("../library/simulate.jl")
include("../library/RiskStats.jl")
include("../library/return_calculate.jl")   # return_calculate: prices -> returns

# The data lives in Week09/data, downloaded by data/get_data.py (see the
# README there).  @__DIR__ is the folder this file is in, so the path works
# no matter where you launch Julia from.
const DATA_DIR = joinpath(@__DIR__, "data")

# Days per year, used to annualize daily variances (var * 255) and
# volatilities (sd * sqrt(255)).  Same convention as earlier weeks.
const ANN = 255

# The estimation window.  60 days with 100 stocks is deliberate: with more
# assets than days the sample covariance is singular (Notes 1.2), which is
# the problem the whole week is about.
const T_WINDOW = 60

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
"""
    load_ff3() -> DataFrame

The Fama-French 3 factor daily file.  Ken French publishes the returns in
PERCENT, so we divide by 100 to put them on the same scale as the stock
returns.  Forgetting this step inflates every factor variance by 10,000.
"""
function load_ff3()
    file = joinpath(DATA_DIR, "ff3_daily.csv")
    isfile(file) || error("$file not found -- run data/get_data.py (see data/README.md)")
    ff3 = CSV.read(file, DataFrame)
    rename!(ff3, Symbol("Mkt-RF") => :Mkt_RF)
    ff3[!, Not(:Date)] = Matrix{Float64}(ff3[!, Not(:Date)]) ./ 100.0
    ff3[!, :Date] = Date.(string.(ff3.Date), dateformat"yyyymmdd")
    return ff3
end

"""
    load_returns(; T = T_WINDOW) -> (returns, stocks, R)

Daily returns for the 100 largest stocks in SPY, plus SPY itself, over the
last `T` days that the Fama-French file also covers.

Returns are computed from adjusted closes in `prices.csv` with the library's
`return_calculate` (Week 04).  The window ends where the Fama-French data
ends, which lags the stock data by a month or two, so that every model in
the week is fit on exactly the same days.

- `returns` : DataFrame with Date, SPY and the 100 stocks, sorted by date
- `stocks`  : the 100 tickers, largest first.  SPY is left out of the
              universe because it is an index, not a stock -- it is (almost)
              a linear combination of the others.
- `R`       : T x n matrix of returns, one row per day, one column per stock
"""
function load_returns(; T = T_WINDOW)
    prices = CSV.read(joinpath(DATA_DIR, "prices.csv"), DataFrame)
    rets = return_calculate(prices; dateColumn = "Date")
    disallowmissing!(rets)                 # return_calculate allows missings; we have none
    ff_dates = Set(load_ff3().Date)
    rets = filter(r -> r.Date in ff_dates, rets)
    sort!(rets, :Date)
    returns = last(rets, T)
    stocks = Symbol.(setdiff(names(returns), ["Date", "SPY"]))
    R = Matrix(returns[!, stocks])
    return returns, stocks, R
end

"""
    load_constituents() -> DataFrame

One row per stock: ticker, name, Yahoo sector, the 8-group industry used in
the notes, SPY weight, and float-adjusted market cap in \$bn.  See
data/get_data.py for how the float-adjusted cap is built.
"""
load_constituents() = CSV.read(joinpath(DATA_DIR, "constituents.csv"), DataFrame)

"""
    mktcap(stocks) -> Vector

Float-adjusted market cap (\$bn) for each ticker in `stocks`, in that order.
Divide by the sum to get cap weights -- these are SPY's own weights,
rescaled to the 100 names.
"""
function mktcap(stocks)
    c = load_constituents()
    lookup = Dict(Symbol(t) => v for (t, v) in zip(c.Ticker, c.FloatAdjMktCap))
    return [lookup[s] for s in stocks]
end

# ---------------------------------------------------------------------
# PSD Cholesky root (Notes 1.2 and the comparison table in 5.3)
# ---------------------------------------------------------------------
"""
    psd_root(A) -> L

Cholesky root of a PSD matrix, A = L*L', using the library's `chol_psd!`
from Week 03.  `chol_psd!` fills a pre-allocated matrix in place (the `!`
convention), so this wrapper just allocates it.
"""
function psd_root(A)
    L = zeros(size(A))
    chol_psd!(L, A)
    return L
end

"""
    nonzero_cols(L; rtol = 1e-6) -> Int

Number of columns of L that carry real variance.

`chol_psd!` sets a pivot to zero only when rounding pushes it slightly
NEGATIVE (between -1e-8 and 0).  A pivot that rounding pushes slightly
POSITIVE, say 4e-19, survives as a column of size ~1e-8 -- about six orders
of magnitude below the genuine columns.  Those are zero in every sense
that matters, so count a column only if its length is at least `rtol`
times the longest column's.
"""
function nonzero_cols(L; rtol = 1e-6)
    cn = [norm(col) for col in eachcol(L)]
    return count(cn .> rtol * maximum(cn))
end

# ---------------------------------------------------------------------
# Model 1: time-series FF3 (Notes 4.1)
# ---------------------------------------------------------------------
"""
    fit_ff3(returns, stocks) -> NamedTuple

Stacked-OLS fit of the Fama-French 3 factor model.  Returns the joined
data `d`, factor names, the pieces `α, B, E, Fcov, Dv`, the factor
covariance matrix `Σ = B*Fcov*B' + D`, and each stock's R².
"""
function fit_ff3(returns, stocks)
    d = sort!(innerjoin(returns, load_ff3(), on = :Date), :Date)
    fnames = [:Mkt_RF, :SMB, :HML]
    m = length(fnames)
    Y  = Matrix(d[!, stocks]) .- d.RF            # excess returns
    Fr = Matrix(d[!, fnames])                    # T x m factor returns
    T  = size(Y, 1)
    X  = hcat(ones(T), Fr)
    Θ  = (X' * X) \ (X' * Y)
    α  = Θ[1, :]
    B  = Matrix(Θ[2:end, :]')                    # n x m
    E  = Y .- X * Θ
    Fcov = cov(Fr)
    Dv   = vec(sum(E .^ 2, dims = 1)) ./ (T - m - 1)
    Σ    = B * Fcov * B' + Diagonal(Dv)
    sysvar = [B[i, :]' * Fcov * B[i, :] for i in 1:size(B, 1)]
    R2     = sysvar ./ (sysvar .+ Dv)
    return (; d, fnames, m, T, Y, Fr, X, α, B, E, Fcov, Dv, Σ, R2)
end

# ---------------------------------------------------------------------
# Model 2: statistical / PCA (Notes 4.3)
# ---------------------------------------------------------------------
"""
    fit_pca(S, k) -> NamedTuple

k-factor statistical model from the covariance matrix `S`.  Loadings are
eigenvectors scaled by sqrt(eigenvalue), so the factors have unit
variance (F = I).  Specific variances are floored at 1e-8 (Heywood case).
"""
function fit_pca(S, k)
    ef  = eigen(Symmetric(S))
    ord = sortperm(ef.values, rev = true)
    λ   = ef.values[ord]
    V   = ef.vectors[:, ord]
    Bp  = V[:, 1:k] * Diagonal(sqrt.(λ[1:k]))
    Dp_raw = diag(S) .- vec(sum(Bp .^ 2, dims = 2))
    Dp  = max.(Dp_raw, 1e-8)
    Σ   = Bp * Bp' + Diagonal(Dp)
    return (; λ, V, B = Bp, Dp_raw, Dv = Dp, Σ)
end

# ---------------------------------------------------------------------
# Model 3: cross-sectional, market + industries (Notes 4.2)
# ---------------------------------------------------------------------
# Industry group for each stock, from data/constituents.csv.  The groups
# are Yahoo sectors, with Energy, Utilities, Materials and Real Estate pooled
# into "Other" so that no group has only one or two stocks.  A production
# model would use GICS industries from a vendor.
const SECTORS = Dict(Symbol(t) => String(g) for (t, g) in
                     zip(load_constituents().Ticker, load_constituents().Group))

"""
    cs_regression(Bx, C, W, Rx) -> (Fhat, E)

One weighted, constrained cross-sectional regression per date:
f̂_t = argmin (r_t - Bx f)'W(r_t - Bx f)  subject to  C f = 0,
solved through the KKT system.  Returns the T x m factor returns and the
T x n residuals.
"""
function cs_regression(Bx, C, W, Rx)
    T, mx = size(Rx, 1), size(Bx, 2)
    KKTi = inv([Bx' * W * Bx  C'; C  zeros(1, 1)])
    Fhat = zeros(T, mx)
    for t in 1:T
        Fhat[t, :] = (KKTi * vcat(Bx' * W * Rx[t, :], 0.0))[1:mx]
    end
    return Fhat, Rx .- Fhat * Bx'
end

"""
    fit_cross_sectional(d, stocks) -> NamedTuple

Market + industry-dummy model with the industry factor returns constrained
to sum to zero, estimated in two steps (Notes 4.2):
  1. OLS (W = I)      -> residuals -> specific variances D
  2. GLS (W = D⁻¹)    -> final factor returns, residuals and Σ
`d` is any DataFrame with one column per stock in `stocks`.
"""
function fit_cross_sectional(d, stocks)
    n = length(stocks)
    industries = sort(unique(getindex.(Ref(SECTORS), stocks)))
    Bx = hcat(ones(n), [SECTORS[s] == g ? 1.0 : 0.0 for s in stocks, g in industries])
    xnames = vcat(["Market"], industries)
    mx = size(Bx, 2)
    C  = reshape(vcat(0.0, ones(length(industries))), 1, mx)
    Rx = Matrix(d[!, stocks])
    T  = size(Rx, 1)
    # Step 1: OLS
    _, E_ols = cs_regression(Bx, C, I(n), Rx)
    D_ols = vec(sum(E_ols .^ 2, dims = 1)) ./ (T - mx)
    # Step 2: GLS with W = D⁻¹ from step 1
    W = Diagonal(1.0 ./ D_ols)
    Fhat, Ex = cs_regression(Bx, C, W, Rx)
    Fcx = cov(Fhat)
    Dx  = vec(sum(Ex .^ 2, dims = 1)) ./ (T - mx)
    Σ   = Bx * Fcx * Bx' + Diagonal(Dx)
    return (; B = Bx, xnames, industries, C, W, D_ols, Fhat, E = Ex, Fcov = Fcx, Dv = Dx, Σ)
end

# ---------------------------------------------------------------------
# Small helpers used by several scripts
# ---------------------------------------------------------------------
"Annualized volatility in percent from a daily variance."
annvol(v) = sqrt(v * ANN) * 100

"Minimum-variance weights subject only to 1'w = 1:  w = Σ⁻¹1 / 1'Σ⁻¹1."
minvar(Σ) = (u = Σ \ ones(size(Σ, 1)); u ./ sum(u))

# Assignment 2 -- worked solution.
#
# Run from this directory:  julia --project=. solution.jl
#
# The numbers printed here are the numbers quoted in the solutions write up.
# Everything that already exists in library/ is used from there rather than
# written again.

using CSV
using DataFrames
using Distributions
using LinearAlgebra
using Random
using StatsBase
using JuMP
using Ipopt
using Printf

# Figures for the answer key. The assignment asks for these plots, so the key
# shows them. Headless GR so this runs without a display.
ENV["GKSwstype"] = "100"
using Plots
gr(size=(900, 500), dpi=150)
default(fontfamily="Computer Modern", legendfontsize=8, guidefontsize=9, titlefontsize=10)
mkpath("figures")

include("../../library/fitted_model.jl")
include("../../library/simulate.jl")
include("../../library/RiskStats.jl")
include("../../library/ewCov.jl")
include("../../library/multivariate_t.jl")
include("../../library/copula.jl")

line(t) = (println(); println(t); println(repeat("-", length(t))))

function printmat(M, nms; fmt="%9.4f")
    f = Printf.Format(fmt)
    @printf("      "); for n in nms; @printf("%9s", n); end; println()
    for i in axes(M, 1)
        @printf("  %-4s", nms[i])
        for j in axes(M, 2); print(Printf.format(f, M[i, j])); end
        println()
    end
end

const NSIM = 100_000

# =========================================================================
# Problem 1 -- correlations from mismatched histories
# =========================================================================
line("Problem 1")

p1 = CSV.read("problem1.csv", DataFrame; missingstring="")
nms1 = ["A", "B", "C", "D", "IDX"]
X = Matrix{Union{Missing,Float64}}(p1[:, nms1])
n1, m1 = size(X)
obs = .!ismissing.(X)

# How many days each pair is jointly observed. The diagonal is each series'
# own history.
N = [sum(obs[:, i] .& obs[:, j]) for i in 1:m1, j in 1:m1]
println("Days jointly observed")
printmat(N, nms1; fmt="%9d")
complete = vec(all(obs, dims=2))
@printf("complete rows: %d of %d (%.0f%%)\n", sum(complete), n1, 100 * mean(complete))

# Method 1, complete cases. Method 2, pairwise.
Ccc = cor(Float64.(X[complete, :]))
Cpw = Matrix{Float64}(I, m1, m1)
for i in 1:m1, j in 1:(i-1)
    r = obs[:, i] .& obs[:, j]
    Cpw[i, j] = Cpw[j, i] = cor(Float64.(X[r, i]), Float64.(X[r, j]))
end
println("Complete case correlation"); printmat(Ccc, nms1)
println("Pairwise correlation"); printmat(Cpw, nms1)
println("Pairwise minus complete case"); printmat(Cpw - Ccc, nms1)

eCC = eigvals(Symmetric(Ccc))
ePW = eigen(Symmetric(Cpw))
@printf("eigenvalues, complete case: %s\n", string(round.(eCC, digits=5)))
@printf("eigenvalues, pairwise:      %s\n", string(round.(ePW.values, digits=5)))
@printf("smallest / largest, pairwise: %.2e\n", ePW.values[1] / ePW.values[end])
println("eigenvector of the negative eigenvalue: ",
        string(round.(ePW.vectors[:, 1], digits=3)))

# Cholesky on each. The PD routine refuses the pairwise matrix; so does
# chol_psd, because the negative pivot is far past floating point error.
for (lab, C) in [("complete case", Ccc), ("pairwise", Cpw)]
    ok = try
        cholesky(Symmetric(C)); "succeeds"
    catch e
        "fails ($(typeof(e)))"
    end
    ok2 = try
        root = zeros(m1, m1); chol_psd!(root, C); "succeeds"
    catch e
        "fails ($(typeof(e)))"
    end
    @printf("  cholesky on %-13s %s   chol_psd %s\n", lab, ok, ok2)
end

# The index weights are given in the problem. A negative eigenvalue means some
# portfolio has negative variance, and the tracking portfolio is the natural
# candidate: long the index, short the basket that builds it.
wIdx = [0.40, 0.30, 0.20, 0.10]
track = vcat(-wIdx, 1.0)
sdFull = [std(skipmissing(X[:, i])) for i in 1:m1]
Dm = diagm(sdFull)
covOf(C) = Dm * C * Dm
@printf("full history sd: %s\n", string(round.(sdFull, digits=5)))
for (lab, C) in [("complete case", Ccc), ("pairwise", Cpw)]
    v = track' * covOf(C) * track
    @printf("  tracking portfolio variance, %-13s %+.3e   (sd %s)\n", lab, v,
            v > 0 ? @sprintf("%.5f", sqrt(v)) : "undefined")
end
# The realised tracking error on the days IDX and all four stocks trade.
te = Float64.(X[complete, :]) * track
@printf("  realised tracking error sd on complete rows: %.5f\n", std(te))

# Repair. near_psd is Rebonato and Jackel; higham_nearestPSD is Higham.
Crj = near_psd(Cpw)
Chi = higham_nearestPSD(Cpw)
frob(A) = sqrt(sum(A .^ 2))
for (lab, C) in [("Rebonato-Jackel", Crj), ("Higham", Chi)]
    @printf("%-16s min eig %+.2e   ||C - pairwise||_F %.5f   max |entry change| %.5f\n",
            lab, minimum(eigvals(Symmetric(C))), frob(C - Cpw), maximum(abs.(C - Cpw)))
    v = track' * covOf(C) * track
    @printf("                 tracking portfolio variance %+.3e\n", v)
end
println("Higham minus pairwise"); printmat(Chi - Cpw, nms1; fmt="%9.5f")
println("Rebonato-Jackel minus pairwise"); printmat(Crj - Cpw, nms1; fmt="%9.5f")
@printf("||Higham - complete case||_F %.4f   ||R-J - complete case||_F %.4f   ||pairwise - complete case||_F %.4f\n",
        frob(Chi - Ccc), frob(Crj - Ccc), frob(Cpw - Ccc))

# Does the size of each move line up with how few days the entry was built on?
pairs = [(i, j) for i in 1:m1 for j in (i+1):m1]
println("entry      days   pairwise   Higham move")
for (i, j) in sort(pairs, by=p -> -abs(Chi[p...] - Cpw[p...]))
    @printf("  %-3s %-3s  %4d   %8.4f   %+9.5f\n", nms1[i], nms1[j], N[i, j], Cpw[i, j], Chi[i, j] - Cpw[i, j])
end
# After the repair the Cholesky runs.
root = zeros(m1, m1); chol_psd!(root, covOf(Chi))
@printf("chol_psd on the Higham covariance: max |LL' - S| = %.2e\n",
        maximum(abs.(root * root' - covOf(Chi))))

# =========================================================================
# Problem 2 -- a volatility estimate after the regime changed
# =========================================================================
line("Problem 2")

p2 = CSV.read("problem2.csv", DataFrame)
px2 = p2.Price
r2 = px2[2:end] ./ px2[1:end-1] .- 1          # arithmetic returns
n2 = length(r2)
@printf("n = %d, mean %.6f\n", n2, mean(r2))
r2 = r2 .- mean(r2)                            # demeaned, as the problem asks
PV2 = 1_000_000.0

@printf("sd %.5f  skew %.3f  exkurt %.3f\n", std(r2), skewness(r2), kurtosis(r2))
@printf("sd first %d: %.5f   sd last 40: %.5f\n", n2 - 40, std(r2[1:(n2-40)]), std(r2[(n2-39):end]))
@printf("annualised (255 days): full %.1f%%  last 40 %.1f%%\n",
        100 * std(r2) * sqrt(255), 100 * std(r2[(n2-39):end]) * sqrt(255))

# Effective sample size and half life.
for lam in (0.94, 0.97)
    w = expW(n2, lam)
    @printf("lambda %.2f: n_eff (formula) %.1f  n_eff (finite sample) %.1f  half life %.1f days  weight on last 40 %.3f\n",
            lam, (1 + lam) / (1 - lam), 1 / sum(w .^ 2), log(0.5) / log(lam), sum(w[(n2-39):end]))
end

z05 = quantile(Normal(), 0.05)
sdEq = std(r2)
sd97 = sqrt(ewCovar(reshape(r2, :, 1), 0.97)[1])
sd94 = sqrt(ewCovar(reshape(r2, :, 1), 0.94)[1])
tfit = fit_general_t(r2)
tdist = tfit.errorModel
var2 = [
    ("Normal, equal weight", -z05 * sdEq),
    ("Normal, EW 0.97", -z05 * sd97),
    ("Normal, EW 0.94", -z05 * sd94),
    ("MLE t", VaR(tdist)),
    ("Historical", VaR(r2)),
]
@printf("EW sd: 0.97 %.5f  0.94 %.5f   t fit: mu %.6f  sigma %.5f  nu %.2f\n",
        sd97, sd94, tdist.μ, tdist.σ, tdist.ρ.ν)
for (lab, v) in sort(var2, by=x -> x[2])
    @printf("  %-22s VaR %.4f  = \$%9.0f\n", lab, v, PV2 * v)
end
# Standard error of the EW vol, sigma / sqrt(2 n_eff).
for (lam, s) in ((0.94, sd94), (0.97, sd97))
    ne = (1 + lam) / (1 - lam)
    @printf("lambda %.2f: sd %.5f  se %.5f (%.1f%%)  VaR +/- 1 se: \$%.0f to \$%.0f\n",
            lam, s, s / sqrt(2ne), 100 / sqrt(2ne),
            -PV2 * z05 * (s - s / sqrt(2ne)), -PV2 * z05 * (s + s / sqrt(2ne)))
end
# How much excess kurtosis does a pure two regime mixture produce? Two normals,
# 460 days at the calm sd and 40 at the stressed one, no fat tails in either.
s1, s2 = std(r2[1:(n2-40)]), std(r2[(n2-39):end])
p = 40 / n2
m2 = (1 - p) * s1^2 + p * s2^2
m4 = 3 * ((1 - p) * s1^4 + p * s2^4)
@printf("two normal mixture excess kurtosis at those sds: %.2f\n", m4 / m2^2 - 3)
@printf("excess kurtosis within regimes: calm %.2f  stressed %.2f\n",
        kurtosis(r2[1:(n2-40)]), kurtosis(r2[(n2-39):end]))

# The plot: returns with the three normal VaR levels, and the EW volatility
# path through the sample so the reader can see how far each has caught up.
function ew_path(r, lam)
    s = zeros(length(r))
    v = var(r[1:60])
    for t in eachindex(r)
        s[t] = sqrt(v)
        v = lam * v + (1 - lam) * r[t]^2
    end
    return s
end
pr = plot(1:n2, r2, lc=:gray60, lw=0.8, label="demeaned return", xlabel="day", ylabel="return",
          title="problem2.csv: arithmetic returns and 5% normal VaR levels")
hline!(pr, [z05 * sdEq], lc=:black, lw=1.5, label="equal weight")
plot!(pr, 1:n2, z05 .* ew_path(r2, 0.97), lc=:steelblue, lw=1.5, label="EW 0.97")
plot!(pr, 1:n2, z05 .* ew_path(r2, 0.94), lc=:crimson, lw=1.5, label="EW 0.94")
savefig(pr, "figures/p2-returns.png")

# =========================================================================
# Problem 3 -- when diversification raises VaR
# =========================================================================
line("Problem 3")

p3 = CSV.read("problem3.csv", DataFrame)
rA, rB = p3.A, p3.B
for (lab, r) in [("A", rA), ("B", rB)]
    @printf("%s: mean %.4f  sd %.4f  skew %.3f  exkurt %.3f  scenarios with loss > 20%%: %d (%.2f%%)\n",
            lab, mean(r), std(r), skewness(r), kurtosis(r), sum(r .< -0.2), 100 * mean(r .< -0.2))
end
@printf("at least one bond loses > 20%%: %.2f%%   both: %.2f%%\n",
        100 * mean((rA .< -0.2) .| (rB .< -0.2)), 100 * mean((rA .< -0.2) .& (rB .< -0.2)))
@printf("correlation A,B: %.4f\n", cor(rA, rB))

# P&L in dollars.
pnl = Dict(
    "1m A" => 1e6 .* rA,
    "1m B" => 1e6 .* rB,
    "2m A" => 2e6 .* rA,
    "2m B" => 2e6 .* rB,
    "1m A + 1m B" => 1e6 .* (rA .+ rB),
)
for alpha in (0.05, 0.01)
    @printf("alpha = %.2f\n", alpha)
    for k in ["1m A", "1m B", "2m A", "2m B", "1m A + 1m B"]
        @printf("  %-12s VaR \$%10.0f   ES \$%10.0f\n", k, VaR(pnl[k]; alpha=alpha), ES(pnl[k]; alpha=alpha))
    end
    vs = VaR(pnl["1m A"]; alpha=alpha) + VaR(pnl["1m B"]; alpha=alpha)
    es = ES(pnl["1m A"]; alpha=alpha) + ES(pnl["1m B"]; alpha=alpha)
    @printf("  sum of parts: VaR \$%.0f vs combined \$%.0f  -> subadditive? %s\n",
            vs, VaR(pnl["1m A + 1m B"]; alpha=alpha), vs >= VaR(pnl["1m A + 1m B"]; alpha=alpha))
    @printf("  sum of parts: ES  \$%.0f vs combined \$%.0f  -> subadditive? %s\n",
            es, ES(pnl["1m A + 1m B"]; alpha=alpha), es >= ES(pnl["1m A + 1m B"]; alpha=alpha))
end
# The normal, fitted to the sample mean and sd -- the absolute convention.
println("Normal VaR at 5%, fitted mean and sd")
for k in ["2m A", "1m A + 1m B", "1m A", "1m B"]
    x = pnl[k]
    @printf("  %-12s \$%10.0f\n", k, -(mean(x) + z05 * std(x)))
end
vsN = -(mean(pnl["1m A"]) + z05 * std(pnl["1m A"])) - (mean(pnl["1m B"]) + z05 * std(pnl["1m B"]))
@printf("  normal: sum of parts \$%.0f vs combined \$%.0f\n", vsN,
        -(mean(pnl["1m A + 1m B"]) + z05 * std(pnl["1m A + 1m B"])))

# The plot: the two ways of spending $2m, on one axis.
bins = range(floor(minimum(pnl["2m A"]), digits=-5), 0.35e6, length=110)
# Step outlines rather than filled bars: fill does not survive a log axis, and
# the log axis is what makes the 4% default mass visible next to the 96%.
ph = stephist(pnl["2m A"], bins=bins, normalize=:probability, lc=:steelblue, lw=1.8,
              label="2m in A", xlabel="P&L (USD millions)", ylabel="fraction of scenarios",
              title="problem3.csv: 2m in one bond against 1m in each", yscale=:log10,
              ylims=(1e-4, 1.2), xformatter=x -> @sprintf("%.1f", x / 1e6))
stephist!(ph, pnl["1m A + 1m B"], bins=bins, normalize=:probability, lc=:darkorange, lw=1.8,
          label="1m in A, 1m in B")
savefig(ph, "figures/p3-hist.png")

# =========================================================================
# Problem 4 -- Gaussian or t copula
# =========================================================================
line("Problem 4")

p4 = CSV.read("problem4.csv", DataFrame)
nms4 = names(p4)
X4 = Matrix(p4)
n4 = size(X4, 1)
for i in 1:3
    x = X4[:, i]
    @printf("%s: mean %.5f  sd %.5f  skew %.3f  exkurt %.3f\n", nms4[i], mean(x), std(x), skewness(x), kurtosis(x))
end

# Sample moments are fragile under fat tails. Refit the moments of X2 without
# its single most extreme day.
x2 = X4[:, 2]
kmin = argmin(x2)
@printf("X2 most extreme day %.4f (%.1f sd). Without it: skew %.3f  exkurt %.3f
",
        x2[kmin], (x2[kmin] - mean(x2)) / std(x2), skewness(x2[1:end .!= kmin]), kurtosis(x2[1:end .!= kmin]))

# Margins: a normal and a t for each, chosen by AICc.
margins = UnivariateDistribution[]
for i in 1:3
    x = X4[:, i]
    fn = fit_normal(x)
    ft = fit_general_t(x)
    an, at = aicc(fn, x), aicc(ft, x)
    pick = at < an ? ft.errorModel : fn.errorModel
    push!(margins, pick)
    @printf("%s: AICc normal %.2f  t %.2f (nu %.2f)  -> %s\n", nms4[i], an, at, ft.errorModel.ρ.ν,
            at < an ? "t" : "normal")
end

# The plot the predict step asks for: ranks of each pair, scaled to (0,1).
Ur = hcat([tiedrank(X4[:, i]) ./ (n4 + 1) for i in 1:3]...)
panels = Plots.Plot[]
for (i, j) in [(1, 2), (1, 3), (2, 3)]
    push!(panels, scatter(Ur[:, i], Ur[:, j], ms=1.8, msw=0, mc=:steelblue, legend=false,
                          xlabel="rank of $(nms4[i])", ylabel="rank of $(nms4[j])",
                          title="$(nms4[i]) vs $(nms4[j])", aspect_ratio=1, xlims=(0, 1), ylims=(0, 1)))
end
savefig(plot(panels..., layout=(1, 3), size=(1200, 420), left_margin=5Plots.mm, bottom_margin=5Plots.mm),
        "figures/p4-ranks.png")

# Joint tail counts in the data, on the ranks. A pair is counted when both
# series sit in their own worst 2.5% on the same day, and separately when both
# sit in their best 2.5%. Both copulas here are symmetric, so both tails count.
const QT = 0.025
const PAIRS = [(1, 2), (1, 3), (2, 3)]
function tail_counts(U, q)
    lo = [sum((U[:, i] .< q) .& (U[:, j] .< q)) for (i, j) in PAIRS]
    hi = [sum((U[:, i] .> 1 - q) .& (U[:, j] .> 1 - q)) for (i, j) in PAIRS]
    return lo, hi
end
loD, hiD = tail_counts(Ur, QT)
@printf("data, pairs jointly in worst 2.5%%: %s   best 2.5%%: %s   total %d   (independence: %.2f per pair per tail, %.1f total)
",
        string(loD), string(hiD), sum(loD) + sum(hiD), n4 * QT^2, 6 * n4 * QT^2)

# The copulas. U comes from the fitted margins.
U4 = hcat([cdf.(margins[i], X4[:, i]) for i in 1:3]...)
Rg, llg = fit_gaussian_copula(U4)
Rt, nut, llt = fit_t_copula(U4)
println("R from Kendall's tau"); printmat(Rg, nms4)
@printf("min eig of R: %.4f\n", minimum(eigvals(Rg)))
@printf("Gaussian copula: ll %.2f  AICc %.2f  BIC %.2f\n", llg, copula_aicc(llg, 0, n4), copula_bic(llg, 0, n4))
@printf("t copula:        ll %.2f  AICc %.2f  BIC %.2f  nu %.2f\n", llt, copula_aicc(llt, 1, n4), copula_bic(llt, 1, n4), nut)
@printf("delta BIC = 2(ll_t - ll_g) - ln(m) = %.2f - %.2f = %.2f\n", 2 * (llt - llg), log(n4), 2 * (llt - llg) - log(n4))
for nu in (4, 6, 8, 12, 20, 40)
    @printf("  profile: nu %3d  ll %.2f\n", nu, t_copula_ll(U4, Rt, nu))
end
for (i, j) in [(1, 2), (1, 3), (2, 3)]
    @printf("tail dependence %s,%s: rho %.3f  lambda at fitted nu %.3f   Gaussian 0\n",
            nms4[i], nms4[j], Rt[i, j], tail_dependence_t(Rt[i, j], nut))
end

# Simulate each copula through the fitted margins, $1m in each asset.
Ug = simulate_gaussian_copula(Rg, NSIM; seed=4)
Ut = simulate_t_copula(Rt, nut, NSIM; seed=4)
Xg = hcat([quantile.(margins[i], Ug[:, i]) for i in 1:3]...)
Xt = hcat([quantile.(margins[i], Ut[:, i]) for i in 1:3]...)
w4 = [1e6, 1e6, 1e6]
println("Portfolio of \$1m in each")
for (lab, S) in [("Historical", X4), ("Gaussian copula", Xg), ("t copula", Xt)]
    pl = S * w4
    @printf("  %-16s VaR5 \$%7.0f  ES5 \$%7.0f  VaR1 \$%7.0f  ES1 \$%7.0f\n", lab,
            VaR(pl; alpha=0.05), ES(pl; alpha=0.05), VaR(pl; alpha=0.01), ES(pl; alpha=0.01))
end
# Normal delta VaR from the sample covariance, for reference.
pvN = sqrt(w4' * cov(X4) * w4)
@printf("  %-16s VaR5 \$%7.0f  ES5 \$%7.0f  VaR1 \$%7.0f  ES1 \$%7.0f\n", "Normal (delta)",
        -z05 * pvN, pvN * pdf(Normal(), z05) / 0.05, -quantile(Normal(), 0.01) * pvN,
        pvN * pdf(Normal(), quantile(Normal(), 0.01)) / 0.01)
# Joint tail counts each copula implies over the same 1000 days, on its own
# simulated uniforms.
for (lab, U) in [("Gaussian", Ug), ("t", Ut)]
    lo, hi = tail_counts(U, QT)
    sc = n4 / NSIM
    @printf("  %-8s implied per %d days: worst %s  best %s  total %.1f
", lab, n4,
            string(round.(lo .* sc, digits=1)), string(round.(hi .* sc, digits=1)), (sum(lo) + sum(hi)) * sc)
end

# Where does the likelihood find its evidence? Each day's contribution to
# ll_t - ll_Gauss, split by whether any series sits in its outer 5% that day.
Zg = quantile.(Normal(), U4)
Tt = quantile.(TDist(nut), U4)
dll = [logpdf(MvTDist(nut, Matrix(Rt)), Tt[k, :]) - sum(logpdf.(TDist(nut), Tt[k, :])) -
       (logpdf(MvNormal(Matrix(Rg)), Zg[k, :]) - sum(logpdf.(Normal(), Zg[k, :]))) for k in 1:n4]
outer = vec(any((U4 .< 0.05) .| (U4 .> 0.95), dims=2))
@printf("ll_t - ll_G = %.1f:  %d days with a series in its outer 5%% contribute %.1f;  the other %d contribute %.1f
",
        sum(dll), sum(outer), sum(dll[outer]), sum(.!outer), sum(dll[.!outer]))
@printf("  per day: outer %.3f   middle %.3f
", mean(dll[outer]), mean(dll[.!outer]))

# =========================================================================
# Problem 5 -- model based simulation with correlated residuals
# =========================================================================
line("Problem 5")

p5 = CSV.read("problem5.csv", DataFrame)
ret(p) = p[2:end] ./ p[1:end-1] .- 1
m = ret(p5.MKT)
ra = ret(p5.A)
rb = ret(p5.B)
n5 = length(m)
Xm = hcat(ones(n5), m)

fits = Dict{String,Any}()
for (lab, r) in [("A", ra), ("B", rb)]
    b = Xm \ r
    e = r - Xm * b
    s = sqrt(sum(e .^ 2) / (n5 - 2))
    se = sqrt.(diag(s^2 .* inv(Xm' * Xm)))
    fits[lab] = (b=b, e=e, s=s)
    @printf("%s: alpha %.5f (se %.5f)  beta %.4f (se %.4f)  resid sd %.5f  R2 %.3f\n",
            lab, b[1], se[1], b[2], se[2], s, 1 - var(e) / var(r))
end
eA, eB = fits["A"].e, fits["B"].e
rhoE = cor(eA, eB)
@printf("residual correlation %.4f   corr of residuals with MKT: %.1e %.1e\n", rhoE, cor(eA, m), cor(eB, m))
@printf("MKT sd %.5f   raw return corr A,B %.4f\n", std(m), cor(ra, rb))

bA, bB = fits["A"].b[2], fits["B"].b[2]
sA, sB = std(eA), std(eB)
Se_full = cov(hcat(eA, eB))
Se_diag = diagm(diag(Se_full))
sm = std(m)

# Holdings in dollars: P1 long both, P2 long A short B.
H = Dict("P1: +1m A, +1m B" => [1e6, 1e6], "P2: +1m A, -1m B" => [1e6, -1e6])
@printf("net market beta: P1 %.3f   P2 %.3f  (per \$1m)\n", bA + bB, bA - bB)

# Simulate the market and the residuals, zero expected return, then build the
# two stock returns through the fitted betas.
Random.seed!(5)
zm = rand(Normal(0, sm), NSIM)
simE(S) = (cholesky(Symmetric(S)).L * rand(Normal(), 2, NSIM))'
Ef = simE(Se_full)
Ed = simE(Se_diag)
Rf5 = hcat(bA .* zm .+ Ef[:, 1], bB .* zm .+ Ef[:, 2])
Rd5 = hcat(bA .* zm .+ Ed[:, 1], bB .* zm .+ Ed[:, 2])
Sig = cov(hcat(ra, rb))
for k in sort(collect(keys(H)))
    h = H[k]
    vF = VaR(Rf5 * h); vD = VaR(Rd5 * h)
    vN = -z05 * sqrt(h' * Sig * h)
    # the analytic versions of both models, to separate model from simulation error
    beta_p = h[1] * bA + h[2] * bB
    aF = -z05 * sqrt(beta_p^2 * sm^2 + h' * Se_full * h)
    aD = -z05 * sqrt(beta_p^2 * sm^2 + h' * Se_diag * h)
    @printf("%s\n  simulated: full resid cov \$%.0f   diagonal resid cov \$%.0f   (diag/full %.2f)\n", k, vF, vD, vD / vF)
    @printf("  analytic:  full \$%.0f   diagonal \$%.0f\n", aF, aD)
    @printf("  delta normal on the sample covariance of A and B: \$%.0f\n", vN)
    @printf("  variance split (full): market %.1f%%  residual %.1f%%\n",
            100 * beta_p^2 * sm^2 / (beta_p^2 * sm^2 + h' * Se_full * h),
            100 * (h' * Se_full * h) / (beta_p^2 * sm^2 + h' * Se_full * h))
end

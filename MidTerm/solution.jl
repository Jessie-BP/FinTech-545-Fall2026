# Midterm -- worked solution.
#
# Run from this directory:  julia --project=. solution.jl
#
# The numbers printed here are the numbers quoted in the answer key. Everything
# that already exists in library/ is used from there rather than written again.
#
# fit_NIG_mle calls scipy through PyCall. See README.md for the Python setup.
#
# Problem 3(c) runs 1,000 replications of a 5,000 draw simulation to put a 99%
# band around every simulated VaR and ES. That takes a few minutes.

using CSV
using DataFrames
using Distributions
using LinearAlgebra
using Random
using StatsBase
using Statistics
using JuMP
using Ipopt
using QuadGK
using Interpolations
using PyCall
using SpecialFunctions
using Printf

# skewNormal.jl carries the NIG cdf and quantile, so it goes first.
include("../library/skewNormal.jl")
include("../library/fitted_model.jl")
include("../library/missing_cov.jl")
include("../library/return_calculate.jl")
include("../library/RiskStats.jl")
include("../library/simulate.jl")
include("../library/multivariate_t.jl")
include("../library/copula.jl")

line(t) = (println(); println(t); println(repeat("-", length(t))))

function printmat(M, nms; fmt="%10.4f")
    f = Printf.Format(fmt)
    @printf("      "); for n in nms; @printf("%10s", n); end; println()
    for i in axes(M, 1)
        @printf("  %-4s", nms[i])
        for j in axes(M, 2); print(Printf.format(f, M[i, j])); end
        println()
    end
end

# =========================================================================
# Problem 1 -- pairwise correlation, full sample variance, Higham
# =========================================================================
line("Problem 1")

df1 = CSV.read("problem1.csv", DataFrame; missingstring="")
nms1 = names(df1)
X1 = Matrix{Union{Missing,Float64}}(df1)
n1, m1 = size(X1)
obs = .!ismissing.(X1)
println("Days jointly observed")
printmat([sum(obs[:, i] .& obs[:, j]) for i in 1:m1, j in 1:m1], nms1; fmt="%10d")
@printf("complete rows: %d of %d\n", sum(all(obs, dims=2)), n1)

# (a) pairwise correlation
C1 = missing_cov(X1; skipMiss=false, fun=cor)
println("(a) Pairwise correlation"); printmat(C1, nms1)

# (b) variance of each series on every observation it has
v1 = [var(skipmissing(X1[:, j])) for j in 1:m1]
println("(b) Full sample variance: ", join([@sprintf("%.6e", v) for v in v1], "  "))
println("    Full sample sd:       ", join([@sprintf("%.5f", sqrt(v)) for v in v1], "  "))

# (c) definiteness
e1 = eigvals(Symmetric(C1))
println("(c) Eigenvalues of the pairwise correlation: ", round.(e1, digits=5))
println("    Eigenvector of the negative eigenvalue: ", round.(eigen(Symmetric(C1)).vectors[:, 1], digits=3))
println("    Complete case eigenvalues, for comparison: ",
        round.(eigvals(Symmetric(missing_cov(X1; skipMiss=true, fun=cor))), digits=4))
Sig_raw = diagm(sqrt.(v1)) * C1 * diagm(sqrt.(v1))
println("    Eigenvalues of the covariance built from it: ", [@sprintf("%.4e", e) for e in eigvals(Symmetric(Sig_raw))])

# (d) Higham on the correlation matrix, W = I
H1 = higham_nearestPSD(C1)
eH = eigvals(Symmetric(H1))
println("(d) Higham nearest PSD correlation"); printmat(H1, nms1)
println("    Eigenvalues after Higham: ", round.(eH, digits=5))
println("    Change, Higham minus pairwise"); printmat(H1 - C1, nms1)
@printf("    Frobenius distance: %.5f\n", norm(H1 - C1))
@assert e1[1] < -0.02 "Problem 1 plant failed: the pairwise matrix is PSD"
@assert eH[1] > -1e-8 "Higham did not return a PSD matrix"

# (e) covariance
D1 = diagm(sqrt.(v1))
Sig1 = D1 * H1 * D1
println("(e) Covariance, D R D"); printmat(Sig1, nms1; fmt="%12.4e")
println("    Eigenvalues: ", [@sprintf("%.4e", e) for e in eigvals(Symmetric(Sig1))])

# =========================================================================
# Problem 2 -- the regression is not needed
# =========================================================================
line("Problem 2")

df2 = CSV.read("problem2.csv", DataFrame)
r2 = return_calculate(df2; dateColumn="Date")
x = r2.XYZ
mu2 = mean(x)
sd2 = std(x)
@printf("n = %d returns, mean XYZ = %.3e, sd XYZ = %.6f\n", length(x), mu2, sd2)
@assert abs(mu2) < 1e-12 "Problem 2 plant failed: XYZ mean is not zero"

d2 = Normal(mu2, sd2)
for a in (0.05, 0.01)
    @printf("  alpha %.2f   VaR %.4f%%   ES %.4f%%\n", a, 100 * VaR(d2; alpha=a), 100 * ES(d2; alpha=a))
end
@printf("  ES / VaR at 5%%: %.4f, at 1%%: %.4f\n", ES(d2; alpha=0.05) / VaR(d2; alpha=0.05), ES(d2; alpha=0.01) / VaR(d2; alpha=0.01))
p_last = df2.XYZ[end]
@printf("  For reference, 100 shares at %.2f = %.2f.  VaR95 = %.2f, VaR99 = %.2f, ES95 = %.2f, ES99 = %.2f\n",
        p_last, 100p_last, 100p_last * VaR(d2; alpha=0.05), 100p_last * VaR(d2; alpha=0.01),
        100p_last * ES(d2; alpha=0.05), 100p_last * ES(d2; alpha=0.01))

# The modelled route, for comparison only. OLS on the same sample decomposes the
# sample variance exactly: var(XYZ) = b^2 var(SPY) + SSE/(n-1). Using the usual
# n-2 residual variance gives a hair more.
Xr = hcat(ones(length(x)), r2.SPY)
b = Xr \ x
res = x - Xr * b
n2 = length(x)
s_e = sqrt(sum(res .^ 2) / (n2 - 2))
sd_model = sqrt(b[2]^2 * var(r2.SPY) + s_e^2)
mu_model = b[1] + b[2] * mean(r2.SPY)
@printf("  OLS: alpha %.6f  beta %.4f  resid sd %.6f  R^2 %.4f\n", b[1], b[2], s_e, 1 - sum(res .^ 2) / sum((x .- mu2) .^ 2))
@printf("  Modelled: mean %.3e  sd %.6f  (sample sd %.6f, ratio %.6f)\n", mu_model, sd_model, sd2, sd_model / sd2)
dm = Normal(mu_model, sd_model)
for a in (0.05, 0.01)
    @printf("  modelled alpha %.2f   VaR %.4f%%   ES %.4f%%\n", a, 100 * VaR(dm; alpha=a), 100 * ES(dm; alpha=a))
end

# =========================================================================
# Problem 3 -- margins, copula, VaR and ES by stock, book, and total
# =========================================================================
line("Problem 3")

df3 = CSV.read("problem3.csv", DataFrame)
hold = CSV.read("problem3_holdings.csv", DataFrame)
hold.Stock = String.(hold.Stock)
hold.Book = String.(hold.Book)
r3 = return_calculate(df3; dateColumn="Date")
stocks = hold.Stock
n3 = nrow(r3)
@printf("%d returns per stock\n", n3)

# Distributions' NIG logpdf multiplies by besselk(1, .), which underflows to zero
# when a fit drifts to the normal limit (alpha in the thousands). besselkx is the
# exponentially scaled Bessel function, so the log is taken before anything can
# underflow. Same density, written to survive that case.
function nig_logpdf(d::NormalInverseGaussian, x)
    mu, a, b, dl = d.μ, d.α, d.β, d.δ
    g = sqrt(a^2 - b^2)
    q = sqrt(dl^2 + (x - mu)^2)
    return log(a * dl / pi) + dl * g + b * (x - mu) + log(besselkx(1, a * q)) - a * q - log(q)
end
loglik(d::NormalInverseGaussian, x) = sum(nig_logpdf.(Ref(d), x))
loglik(d, x) = sum(logpdf.(d, x))
nparams(d) = length(Distributions.params(d))

# Criteria from the log likelihood. ic_aicc is the library aicc written on ll so
# that the NIG can use the stable density above; for the normal and the t it
# matches aicc() exactly, which the assert below checks.
ic_aicc(ll, k, n) = -2ll + 2k + 2k * (k + 1) / (n - k - 1)
ic_bic(ll, k, n) = k * log(n) - 2ll

families = ["Normal", "T", "NIG"]
fitted = Dict{String,Vector{FittedModel}}()
crit = DataFrame(Stock=String[], Family=String[], k=Int[], LL=Float64[], AICc=Float64[], BIC=Float64[])
for s in stocks
    d = r3[!, s]
    fs = [fit_normal(d), fit_general_t(d), fit_NIG_mle(d)]
    fitted[s] = fs
    for (f, fam) in zip(fs, families)
        ll = loglik(f.errorModel, d)
        k = nparams(f.errorModel)
        push!(crit, (s, fam, k, ll, ic_aicc(ll, k, n3), ic_bic(ll, k, n3)))
    end
    @assert isapprox(aicc(fs[1], d), ic_aicc(loglik(fs[1].errorModel, d), 2, n3))
    @assert isapprox(aicc(fs[2], d), ic_aicc(loglik(fs[2].errorModel, d), 3, n3))
end

println("(a) Sample moments")
for s in stocks
    d = r3[!, s]
    @printf("  %-4s mean %.6f  sd %.6f  skew %+.3f  excess kurtosis %.3f\n", s, mean(d), std(d), skewness(d), kurtosis(d))
end

println("(a) Fitted parameters")
for s in stocks
    nfit, tfit, gfit = (f.errorModel for f in fitted[s])
    @printf("  %-4s Normal mu %.6f sigma %.6f | T mu %.6f sigma %.6f nu %.3f | NIG mu %.6f alpha %.3f beta %.3f delta %.6f\n",
            s, nfit.μ, nfit.σ, tfit.μ, tfit.σ, tfit.ρ.ν, gfit.μ, gfit.α, gfit.β, gfit.δ)
end

println("(a) Log likelihood and criteria, with the gap to the best in each column")
choice = DataFrame(Stock=String[], AICc=String[], BIC=String[])
for s in stocks
    c = crit[crit.Stock .== s, :]
    ia = argmin(c.AICc); ib = argmin(c.BIC)
    for r in eachrow(c)
        @printf("  %-4s %-7s LL %10.3f  AICc %11.3f (+%7.2f)  BIC %11.3f (+%7.2f)\n", s, r.Family, r.LL,
                r.AICc, r.AICc - c.AICc[ia], r.BIC, r.BIC - c.BIC[ib])
    end
    push!(choice, (s, c.Family[ia], c.Family[ib]))
end
println(choice)
agree = all(choice.AICc .== choice.BIC)
println("AICc and BIC agree on every stock: ", agree)
for fam in families
    @assert any(choice.AICc .== fam) "Problem 3 plant failed: no stock picks $fam"
end

# (b) copulas, fit on the chosen margins' U, same Kendall R for both
function copula_step(pick)
    best = Dict(s => fitted[s][findfirst(==(pick[s]), families)] for s in stocks)
    U = hcat([best[s].u for s in stocks]...)
    RG, llG = fit_gaussian_copula(U)
    RT, nu, llT = fit_t_copula(U)
    return best, U, RG, llG, nu, llT
end
pickA = Dict(r.Stock => r.AICc for r in eachrow(choice))
best, U3, R3, llG, nu3, llT = copula_step(pickA)
m3 = size(U3, 1)
println("(b) Kendall based R"); printmat(R3, stocks; fmt="%7.3f")
println("    Smallest eigenvalue of R: ", round(minimum(eigvals(Symmetric(R3))), digits=4))
@printf("    Gaussian: LL %.3f  k 0  AICc %.3f  BIC %.3f\n", llG, copula_aicc(llG, 0, m3), copula_bic(llG, 0, m3))
@printf("    t:        LL %.3f  k 1  nu %.2f  AICc %.3f  BIC %.3f\n", llT, nu3, copula_aicc(llT, 1, m3), copula_bic(llT, 1, m3))
dAICc = copula_aicc(llT, 1, m3) - copula_aicc(llG, 0, m3)
dBIC = copula_bic(llT, 1, m3) - copula_bic(llG, 0, m3)
@printf("    t minus Gaussian: AICc %+.3f  BIC %+.3f   (positive favours the Gaussian)\n", dAICc, dBIC)
# the copula log likelihood at a few nu, to show it is still rising at the bound
for nu in (5.0, 10.0, 20.0, 50.0, 100.0)
    @printf("      nu %5.0f  t copula LL %.3f\n", nu, t_copula_ll(U3, R3, nu))
end
@assert dAICc > 0 && dBIC > 0 "Problem 3 plant failed: the Gaussian copula does not win"
Rs = corspearman(U3)
@printf("    Largest |Spearman - Kendall R| entry: %.4f\n", maximum(abs.(Rs - R3)))

# (c) simulation
prices = Dict(s => df3[end, s] for s in stocks)
hold.Price = [prices[s] for s in stocks]
hold.Value = hold.Holding .* hold.Price
println("(c) Current positions"); println(hold)
@printf("    Book A %.2f   Book B %.2f   Total %.2f\n", sum(hold.Value[hold.Book .== "A"]),
        sum(hold.Value[hold.Book .== "B"]), sum(hold.Value))

const NSIM = 5000
const NREP = parse(Int, get(ENV, "NREP", "1000"))

function risk_run(R, nsim, seed)
    simU = simulate_gaussian_copula(R, nsim; seed=seed)
    simR = DataFrame([best[s].eval(simU[:, j]) for (j, s) in enumerate(stocks)], stocks)
    values = crossjoin(hold[:, [:Stock, :Book, :Holding, :Price]], DataFrame(iteration=1:nsim))
    values.currentValue = values.Holding .* values.Price
    values.simulatedValue = [values.currentValue[i] * (1 + simR[values.iteration[i], values.Stock[i]]) for i in 1:nrow(values)]
    values.pnl = values.simulatedValue .- values.currentValue
    out = aggRisk(values, [:Book, :Stock])
    out.Label = [r.Stock != "Total" ? r.Stock : (r.Book != "Total" ? "Book " * r.Book : "Total") for r in eachrow(out)]
    return out[:, [:Label, :currentValue, :VaR95, :ES95]]
end

# The key quotes the median of the replications, with the 0.5% and 99.5%
# quantiles as the band. A student running 5,000 draws once should land inside
# the band 99% of the time on any single line.
t0 = time()
reps = [risk_run(R3, NSIM, s) for s in 1:NREP]
@printf("    %d replications in %.0f s\n", NREP, time() - t0)
labels = reps[1].Label
V = hcat([r.VaR95 for r in reps]...)
E = hcat([r.ES95 for r in reps]...)
q(v, p) = quantile(v, p)
bands = DataFrame(Label=labels, Value=reps[1].currentValue,
                  VaR95=[median(V[i, :]) for i in eachindex(labels)],
                  VaR_lo=[q(V[i, :], 0.005) for i in eachindex(labels)], VaR_hi=[q(V[i, :], 0.995) for i in eachindex(labels)],
                  ES95=[median(E[i, :]) for i in eachindex(labels)],
                  ES_lo=[q(E[i, :], 0.005) for i in eachindex(labels)], ES_hi=[q(E[i, :], 0.995) for i in eachindex(labels)])
println("(c) Median of $(NREP) runs of $(NSIM) draws, with the 99% band")
for r in eachrow(bands)
    @printf("  %-7s value %11.2f  VaR95 %9.2f [%9.2f, %9.2f]   ES95 %9.2f [%9.2f, %9.2f]\n",
            r.Label, r.Value, r.VaR95, r.VaR_lo, r.VaR_hi, r.ES95, r.ES_lo, r.ES_hi)
end
CSV.write("q3_bands.csv", bands)

# Diversification, from the medians
sv(l) = bands.VaR95[bands.Label .== l][1]
se(l) = bands.ES95[bands.Label .== l][1]
sumA = sum(sv(s) for s in stocks[hold.Book .== "A"]); sumB = sum(sv(s) for s in stocks[hold.Book .== "B"])
@printf("    Sum of stock VaR, A %.2f vs book %.2f; B %.2f vs book %.2f; books %.2f vs total %.2f; all stocks %.2f\n",
        sumA, sv("Book A"), sumB, sv("Book B"), sv("Book A") + sv("Book B"), sv("Total"), sumA + sumB)
seA = sum(se(s) for s in stocks[hold.Book .== "A"]); seB = sum(se(s) for s in stocks[hold.Book .== "B"])
@printf("    Sum of stock ES,  A %.2f vs book %.2f; B %.2f vs book %.2f; books %.2f vs total %.2f; all stocks %.2f\n",
        seA, se("Book A"), seB, se("Book B"), se("Book A") + se("Book B"), se("Total"), seA + seB)

# Spearman R in place of Kendall, same draws. Within the band means the choice
# of rank correlation does not matter at this precision.
sp = risk_run(Rs, NSIM, 1234)
inside = (bands.VaR_lo .<= sp.VaR95 .<= bands.VaR_hi) .& (bands.ES_lo .<= sp.ES95 .<= bands.ES_hi)
println("    Spearman R run inside the Kendall bands on every line: ", all(inside))
for (r, ok) in zip(eachrow(sp), inside)
    @printf("      %-7s VaR95 %9.2f  ES95 %9.2f  %s\n", r.Label, r.VaR95, r.ES95, ok ? "" : "OUTSIDE")
end

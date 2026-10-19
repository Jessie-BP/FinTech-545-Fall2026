# Assignment 1 -- worked solution.
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

line(t) = (println(); println(t); println(repeat("-", length(t))))

# =========================================================================
# Problem 1 -- reading the shape of a sample
# =========================================================================
line("Problem 1")

p1 = CSV.read("problem1.csv", DataFrame)
x = p1.x
n = length(x)

# The population (biased) convention, written out, and what StatsBase returns.
xc = x .- mean(x)
cm2 = sum(xc .^ 2) / n
m1 = mean(x)
m2 = cm2
m3 = sum(xc .^ 3) / n / sqrt(cm2^3)
m4 = sum(xc .^ 4) / n / cm2^2 - 3

@printf("n = %d\n", n)
@printf("  mean            %10.6f\n", m1)
@printf("  variance (pop)  %10.6f   (sample) %10.6f\n", m2, var(x))
@printf("  sd              %10.6f\n", std(x))
@printf("  skewness        %10.4f   (StatsBase %.4f)\n", m3, skewness(x))
@printf("  excess kurtosis %10.4f   (StatsBase %.4f)\n", m4, kurtosis(x))

# What a normal would have said about the tail, against what the sample shows.
nd = fit_normal(x)
for q in [0.01, 0.05]
    emp = quantile(x, q)
    nrm = quantile(nd.errorModel, q)
    @printf("  %.0f%% quantile: empirical %8.5f   normal %8.5f   ratio %.2f\n",
            100q, emp, nrm, emp / nrm)
end
@printf("  observations past the normal 1%% quantile: %d (a normal expects %.0f)\n",
        sum(x .< quantile(nd.errorModel, 0.01)), 0.01n)

# =========================================================================
# Problem 2 -- a regression whose errors are not normal
# =========================================================================
line("Problem 2")

p2 = CSV.read("problem2.csv", DataFrame)
y = p2.y
X = hcat(ones(size(p2, 1)), p2.x)
n2 = size(p2, 1)

# OLS by the normal equations, with the textbook standard errors.
beta_ols = X \ y
resid = y - X * beta_ols
s2 = sum(resid .^ 2) / (n2 - 2)
se_ols = sqrt.(diag(s2 .* inv(X' * X)))

@printf("OLS      alpha %7.4f (se %.4f)   beta %7.4f (se %.4f)   sigma %.4f\n",
        beta_ols[1], se_ols[1], beta_ols[2], se_ols[2], sqrt(s2))
@printf("  largest |residual| %.2f, which is %.1f sigma\n",
        maximum(abs.(resid)), maximum(abs.(resid)) / sqrt(s2))

# Normal MLE. Same betas as OLS by construction; sigma divides by n, not n-2.
beta_n = beta_ols
sigma_n = sqrt(sum(resid .^ 2) / n2)
ll_n = sum(logpdf.(Normal(0, sigma_n), resid))
k_n = 3                              # alpha, beta, sigma
aicc_n = -2 * ll_n + 2 * k_n + 2 * k_n * (k_n + 1) / (n2 - k_n - 1)

@printf("Normal MLE  alpha %7.4f   beta %7.4f   sigma %.4f   ll %8.2f   AICc %8.2f\n",
        beta_n[1], beta_n[2], sigma_n, ll_n, aicc_n)

# t MLE.
fm_t = fit_regression_t(y, p2.x)
beta_t = fm_t.beta
nu_t = fm_t.errorModel.ρ.ν
sigma_t = fm_t.errorModel.σ
ll_t = sum(logpdf.(fm_t.errorModel, fm_t.errors))
k_t = 4                              # alpha, beta, sigma, nu
aicc_t = -2 * ll_t + 2 * k_t + 2 * k_t * (k_t + 1) / (n2 - k_t - 1)

@printf("t MLE       alpha %7.4f   beta %7.4f   sigma %.4f   nu %.2f   ll %8.2f   AICc %8.2f\n",
        beta_t[1], beta_t[2], sigma_t, nu_t, ll_t, aicc_t)
@printf("  delta AICc = %.2f in favour of the %s error model\n",
        aicc_n - aicc_t, aicc_t < aicc_n ? "t" : "normal")

# The betas barely move. What moves is the shape of the error distribution:
# the normal has to inflate its scale to reach the outliers, which makes it too
# wide through the middle and still too thin in the tail.
@printf("  error scale: normal %.4f   t %.4f  (the normal is %.0f%% wider)\n",
        sigma_n, sigma_t, 100 * (sigma_n / sigma_t - 1))
for q in [0.95, 0.99, 0.995]
    hn = quantile(Normal(0, sigma_n), q)
    ht = quantile(fm_t.errorModel, q) - fm_t.errorModel.μ
    @printf("  %.1f%% quantile of the error: normal %6.3f   t %6.3f\n", 100q, hn, ht)
end

# The plot the question asks for, before any fitting: y against x. The outliers
# are the point of it. Flag the ones past 3 residual sigma so they are countable.
big = abs.(resid) .> 3 * sqrt(s2)
xs = range(minimum(p2.x), maximum(p2.x), length=100)
plt2 = scatter(p2.x[.!big], y[.!big], label="data", ms=3, mc=:steelblue, msw=0,
               xlabel="x", ylabel="y", title="problem2.csv: y against x")
scatter!(plt2, p2.x[big], y[big], label="|residual| > 3 sigma", ms=5, mc=:crimson, msw=0)
plot!(plt2, xs, beta_ols[1] .+ beta_ols[2] .* xs, label="OLS", lc=:black, lw=2)
plot!(plt2, xs, beta_t[1] .+ beta_t[2] .* xs, label="t MLE", lc=:darkorange, lw=2, ls=:dash)
savefig(plt2, "figures/p2-scatter.png")
@printf("  points past 3 residual sigma: %d of %d\n", sum(big), n2)

# =========================================================================
# Problem 3 -- Pearson against Spearman
# =========================================================================
line("Problem 3")

p3 = CSV.read("problem3.csv", DataFrame)
M = Matrix(p3)
P = cor(M)
S = corspearman(M)
nms = names(p3)

println("Pearson")
for i in 1:4
    @printf("  %-4s", nms[i]); for j in 1:4; @printf("%9.3f", P[i, j]); end; println()
end
println("Spearman")
for i in 1:4
    @printf("  %-4s", nms[i]); for j in 1:4; @printf("%9.3f", S[i, j]); end; println()
end
println("Largest gaps |Spearman - Pearson|")
for i in 1:4, j in (i+1):4
    @printf("  %s,%s  %.3f\n", nms[i], nms[j], abs(S[i, j] - P[i, j]))
end

# Every pair, as the question asks. Four series give six pairs. The panel with
# the largest Pearson/Spearman gap is coloured differently so it stands out.
gaps = [(i, j, abs(S[i, j] - P[i, j])) for i in 1:4 for j in (i+1):4]
worst = argmax([g[3] for g in gaps])
panels = Plots.Plot[]
for (k, (i, j, _)) in enumerate(gaps)
    push!(panels,
        scatter(M[:, i], M[:, j], ms=2, msw=0, legend=false,
                mc=(k == worst ? :crimson : :steelblue),
                xlabel=nms[i], ylabel=nms[j],
                title=@sprintf("%s vs %s: P %.3f  S %.3f", nms[i], nms[j], P[i, j], S[i, j])))
end
savefig(plot(panels..., layout=(2, 3), size=(1200, 720),
             left_margin=6Plots.mm, bottom_margin=6Plots.mm, top_margin=2Plots.mm),
        "figures/p3-pairs.png")
# =========================================================================
# Problem 4 -- the conditional distribution
# =========================================================================
line("Problem 4")

p4 = CSV.read("problem4.csv", DataFrame)
x1 = p4.x1
x2 = p4.x2
mu4 = [mean(x1), mean(x2)]
C = cov(Matrix(p4))
rho = C[1, 2] / sqrt(C[1, 1] * C[2, 2])

b = C[1, 2] / C[1, 1]
condVar = C[2, 2] - b * C[1, 2]

# The blocks of the covariance matrix, which is the form the notes give the
# conditional distribution in. With N = 2 every block is a scalar.
@printf("covariance matrix\n")
@printf("  S11 %.4f   S12 %.4f\n  S21 %.4f   S22 %.4f\n", C[1,1], C[1,2], C[2,1], C[2,2])
@printf("corr %.4f\n", rho)
@printf("S21*inv(S11)*S12         %.4f\n", C[2,1] * inv(C[1,1]) * C[1,2])
@printf("unconditional var of x2  %.4f   sd %.4f\n", C[2, 2], sqrt(C[2, 2]))
@printf("conditional var of x2|x1 %.4f   sd %.4f\n", condVar, sqrt(condVar))
@printf("variance ratio           %.4f   sd ratio %.4f  (= sqrt(1-rho^2) = %.4f)\n",
        condVar / C[2, 2], sqrt(condVar / C[2, 2]), sqrt(1 - rho^2))
@printf("regression coefficient S21*inv(S11) = %.4f\n", C[2, 1] * inv(C[1, 1]))

expt = mu4[2] .+ b .* (x1 .- mu4[1])
lcl = expt .+ quantile(Normal(0, sqrt(condVar)), 0.025)
ucl = expt .+ quantile(Normal(0, sqrt(condVar)), 0.975)
inside = (x2 .>= lcl) .& (x2 .<= ucl)
@printf("overall coverage of the 95%% band: %.1f%%\n", 100 * mean(inside))

# Split on how far x1 sits from its mean. Under a normal the coverage would be
# flat across these buckets. Under an elliptical t it is not.
d = abs.(x1 .- mu4[1]) ./ std(x1)
for (lab, sel) in [("|x1| < 1 sd", d .< 1), ("1 to 2 sd", (d .>= 1) .& (d .< 2)), ("beyond 2 sd", d .>= 2)]
    @printf("  %-12s n = %4d   coverage %.1f%%\n", lab, sum(sel), 100 * mean(inside[sel]))
end

# The plot the question asks for: the conditional expectation with a 95% band,
# over the scatter. Points outside the band are marked so the reader can see
# where they concentrate.
ord = sortperm(x1)
plt4 = plot(x1[ord], lcl[ord], fillrange=ucl[ord], fillalpha=0.18, fc=:steelblue,
            lc=:transparent, label="95% band", xlabel="x1", ylabel="x2",
            title="Conditional expectation of x2 given x1, with a constant width 95% band")
scatter!(plt4, x1[inside], x2[inside], ms=2.5, mc=:steelblue, msw=0, label="inside")
scatter!(plt4, x1[.!inside], x2[.!inside], ms=3.5, mc=:crimson, msw=0, label="outside")
plot!(plt4, x1[ord], expt[ord], lc=:black, lw=2, label="E[x2|x1]")
savefig(plt4, "figures/p4-band.png")

# =========================================================================
# Problem 5 -- identifying the order
# =========================================================================
line("Problem 5")

p5 = CSV.read("problem5.csv", DataFrame)
s = p5.x
ns = length(s)

@printf("ACF  1:6  %s\n", string(round.(autocor(s, 1:6), digits=3)))
@printf("PACF 1:6  %s\n", string(round.(pacf(s, 1:6), digits=3)))
@printf("2/sqrt(n) band = %.3f\n", 2 / sqrt(ns))

# The three plots the question asks for: the series, its ACF, and its PACF.
band = 2 / sqrt(ns)
lags = 1:12
ps = plot(1:ns, s, legend=false, lc=:steelblue, lw=0.7,
          xlabel="t", ylabel="x", title="problem5.csv")
pacf_v = pacf(s, lags)
acf_v = autocor(s, lags)
pa5 = bar(lags, acf_v, legend=false, fc=:steelblue, lc=:steelblue,
          xlabel="lag", title="ACF", ylims=(-0.4, 0.7))
hline!(pa5, [band, -band], lc=:crimson, ls=:dash)
pp5 = bar(lags, pacf_v, legend=false, fc=:seagreen, lc=:seagreen,
          xlabel="lag", title="PACF", ylims=(-0.4, 0.7))
hline!(pp5, [band, -band], lc=:crimson, ls=:dash)
savefig(plot(ps, plot(pa5, pp5, layout=(1, 2)), layout=(2, 1), size=(950, 620)),
        "figures/p5-acf-pacf.png")

# AR(p) by conditional least squares, then the Gaussian AICc on the same
# conditioning set so the orders are comparable to each other.
function fit_ar(y, p; pmax=3)
    T = length(y)
    rows = (pmax+1):T
    Z = hcat(ones(length(rows)), [y[rows.-j] for j in 1:p]...)
    b = Z \ y[rows]
    r = y[rows] - Z * b
    sig = sqrt(sum(r .^ 2) / length(r))
    ll = sum(logpdf.(Normal(0, sig), r))
    k = p + 2
    return b, sig, ll, -2 * ll + 2 * k + 2 * k * (k + 1) / (length(r) - k - 1)
end

# MA(q) by conditional MLE: run the residual recursion forward from zeros.
function fit_ma(y, q; pmax=3)
    T = length(y)
    rows = (pmax+1):T
    function negll(par)
        c = par[1]; th = par[2:(q+1)]; sig = par[q+2]
        sig <= 0 && return 1e10
        e = zeros(T)
        for t in 1:T
            m = c
            for j in 1:q
                t - j >= 1 && (m += th[j] * e[t-j])
            end
            e[t] = y[t] - m
        end
        return -sum(logpdf.(Normal(0, sig), e[rows]))
    end
    # Nelder-Mead by hand is overkill; a coarse coordinate search is enough
    # here because the surface is smooth and low dimensional.
    par = vcat(mean(y), zeros(q), std(y))
    step = vcat(0.5, fill(0.4, q), 0.4)
    best = negll(par)
    for _ in 1:400
        improved = false
        for i in eachindex(par)
            for d in (-step[i], step[i])
                trial = copy(par); trial[i] += d
                v = negll(trial)
                if v < best - 1e-10
                    best = v; par = trial; improved = true
                end
            end
        end
        improved || (step ./= 2.0)
        maximum(step) < 1e-7 && break
    end
    ll = -best
    k = q + 2
    nEff = length(rows)
    return par, ll, -2 * ll + 2 * k + 2 * k * (k + 1) / (nEff - k - 1)
end

println("model     AICc        parameters")
results = Tuple{String,Float64}[]
for p in 1:3
    b, sig, ll, a = fit_ar(s, p)
    @printf("AR(%d)  %10.2f     c %.3f  phi %s  sigma %.3f\n",
            p, a, b[1], string(round.(b[2:end], digits=3)), sig)
    push!(results, ("AR($p)", a))
end
for q in 1:3
    par, ll, a = fit_ma(s, q)
    @printf("MA(%d)  %10.2f     c %.3f  theta %s  sigma %.3f\n",
            q, a, par[1], string(round.(par[2:(q+1)], digits=3)), par[q+2])
    push!(results, ("MA($q)", a))
end
bestModel = results[argmin([r[2] for r in results])]
@printf("\nlowest AICc: %s at %.2f\n", bestModel[1], bestModel[2])

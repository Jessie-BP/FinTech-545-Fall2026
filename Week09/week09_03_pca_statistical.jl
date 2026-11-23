# =====================================================================
# Week 09 -- Script 3: A statistical factor model (PCA)
#
# Notes, Section 4.3 "Statistical Models"
#   -> Rotation indeterminacy: is PC1 the market?
#   -> Table "Variance explained by principal components" and the scree plot
#   -> Asymptotic PCA when n > T
#   -> PCA on correlation vs covariance
# Notes, Section 5.2/5.3
#   -> minimum d_i (the Heywood check at m = 3)
#   -> Table "The three covariance estimates compared"
#
# In a statistical model NEITHER the factors NOR the betas are observed.
# Both come out of the eigendecomposition of the sample covariance.  This is
# Week 03's PCA, used to build a covariance model instead of to simulate.
#
# Run from anywhere:   julia --project=Week09 Week09/week09_03_pca_statistical.jl
# =====================================================================

include("week09_common.jl")
using Plots
ENV["GKSwstype"] = "100"             # headless plotting: write the file, no window

returns, stocks, R = load_returns()
T, n = size(R)
S = cov(R)

# ---------------------------------------------------------------------
# Eigendecomposition  S = V Λ V'
# ---------------------------------------------------------------------
# eigen() returns eigenvalues in ASCENDING order.  PCA wants the biggest
# first, so sort descending and reorder the eigenvectors to match.
ef  = eigen(Symmetric(S))
ord = sortperm(ef.values, rev = true)
λ   = ef.values[ord]
V   = ef.vectors[:, ord]

# ---------------------------------------------------------------------
# Choosing m: how much variance does each component explain?
# ---------------------------------------------------------------------
# The total variance is the trace of S = the sum of the eigenvalues.  Each
# eigenvalue is the variance of one principal component, so λ_k / Σλ is
# that component's share.
println("=== Variance explained (Notes 4.3, 'Choosing m') ===")
tot = sum(λ)
cumvar = cumsum(λ) ./ tot
for k in [1, 2, 3, 5, 10, 20, 59]
    @printf("  k = %2d   cumulative variance explained = %5.1f%%\n", k, cumvar[k] * 100)
end
# Past component 59 there is nothing left: the rank problem from Script 1.
@printf("  eigenvalue 60 = %.1e   <- everything past 59 is zero\n", λ[60])

# Scree plot.  Look for the "elbow" where the bars flatten into the noise
# floor.  For equity returns there is usually no clean elbow -- you see one
# big market component and then a slow decay.
scree = plot(1:30, λ[1:30] ./ tot .* 100, seriestype = :bar, legend = false,
             xlabel = "Component", ylabel = "% of variance", title = "Scree")
cumpl = plot(1:30, cumvar[1:30] .* 100, marker = :circle, legend = false,
             xlabel = "Number of components", ylabel = "Cumulative %",
             title = "Cumulative variance", ylims = (0, 100))
figfile = joinpath(@__DIR__, "figures", "scree_julia.png")
savefig(plot(scree, cumpl, layout = (1, 2), size = (900, 340)), figfile)
println("  scree plot written to ", figfile)

# ---------------------------------------------------------------------
# Build the m = 3 model:  B = V_m Λ_m^{1/2},  F = I
# ---------------------------------------------------------------------
# Scaling each eigenvector by sqrt(eigenvalue) gives the factors unit
# variance, so F is the identity and BFB' = BB'.
k  = 3
Bp = V[:, 1:k] * Diagonal(sqrt.(λ[1:k]))

# Specific variance = what the k components did NOT explain about each
# stock:  d_i = S_ii - Σ_k B_ik².  This can go negative if the components
# "over-explain" a stock (a Heywood case, Notes 5.2).  A negative variance
# would break PSD, so floor it at a small positive number.
Dp = diag(S) .- vec(sum(Bp .^ 2, dims = 2))
println("\n=== PCA-$k model ===")
@printf("minimum specific variance = %.3e  (%s)\n", minimum(Dp),
        minimum(Dp) > 0 ? "positive, no flooring needed" : "NEGATIVE => Heywood case, flooring")
Dp = max.(Dp, 1e-8)
Σ_pca = Bp * Bp' + Diagonal(Dp)

# ---------------------------------------------------------------------
# Rotation indeterminacy: is PC1 "the market"?
# ---------------------------------------------------------------------
# For any orthogonal Q, (BQ)(BQ)' = BB'.  The model fits equally well with
# rotated loadings, so it has no opinion on what the factors MEAN.  Check
# whether PC1 resembles the market rather than assuming it.
#
# Compare to the FF3 market betas from Script 2.  Eigenvector signs are
# arbitrary (v and -v are both eigenvectors), so a correlation of -0.95 is
# just as strong as +0.95.
#
# Three checks per component:
#   - correlation of its loadings with the FF3 market betas
#   - share of loadings with the majority sign (a market factor moves
#     nearly every stock the same way, so this should be close to 1)
#   - correlation of its factor-mimicking return (R * loading) with SPY
ff = fit_ff3(returns, stocks)
for j in 1:2
    b = Bp[:, j]
    pos = mean(b .> 0)
    @printf("PC%d: corr(loading, FF3 market beta) = %+.3f   same-sign share %.2f   corr(R*loading, SPY) = %+.3f\n",
            j, cor(b, ff.B[:, 1]), max(pos, 1 - pos), cor(R * b, returns.SPY))
end
# Which stocks sit at the two ends of PC1?
o1 = sortperm(Bp[:, 1] .* sign(sum(Bp[:, 1])))
println("PC1 ends: ", join(String.(stocks[o1[1:5]]), " "), "  <->  ",
        join(String.(stocks[o1[end-4:end]]), " "))
# Here PC1 is a semiconductor/hardware versus software-and-defensives
# spread, the largest source of variance in this window.  The market-like
# component is PC2.  PCA ranks components by variance; it never promised
# that the biggest one is the one with a name.

# ---------------------------------------------------------------------
# PCA on covariance vs correlation
# ---------------------------------------------------------------------
# Covariance PCA weights each stock by its variance.  A stock with 4.8x the
# volatility has ~23x the variance and so ~23x the pull on the components.
i_tsla, i_jnj = findfirst(==(:TSLA), stocks), findfirst(==(:JNJ), stocks)
@printf("\nvariance ratio TSLA / JNJ = %.0f   <- TSLA's extra influence in covariance PCA\n",
        S[i_tsla, i_tsla] / S[i_jnj, i_jnj])

# ---------------------------------------------------------------------
# Asymptotic PCA (Connor-Korajczyk):  work with the T x T matrix
# ---------------------------------------------------------------------
# With n > T, R̃'R̃ (n x n) and R̃R̃' (T x T) share the same non-zero
# eigenvalues.  The T x T problem is smaller and gives the same answer.
Rt = R .- mean(R, dims = 1)                     # demeaned returns, T x n
ev_small = sort(eigvals(Symmetric(Rt * Rt' ./ (T - 1))), rev = true)   # T x T
ev_big   = λ                                                            # n x n
@printf("asymptotic PCA: max |difference| in the top %d eigenvalues = %.1e\n",
        T - 1, maximum(abs.(ev_small[1:T-1] .- ev_big[1:T-1])))

# ---------------------------------------------------------------------
# The three covariance estimates compared  (Notes 5.3 table)
# ---------------------------------------------------------------------
# Sample vs FF3 (Script 2) vs PCA-3 (this script).  Both factor models are
# full rank and well conditioned; the sample matrix is neither.
println("\n=== The three covariance estimates compared ===")
@printf("%-12s %6s %12s %12s %16s\n", "", "rank", "min eig", "cond", "nonzero cols L")
for (nm, Σ) in [("Sample", S), ("FF3", ff.Σ), ("PCA-3", Σ_pca)]
    ev = eigvals(Symmetric(Σ))
    cnd = minimum(ev) > 1e-14 ? @sprintf("%.0f", maximum(ev) / minimum(ev)) : "undefined"
    @printf("%-12s %6d %12.2e %12s %12d of %d\n",
            nm, rank(Σ), minimum(ev), cnd, nonzero_cols(psd_root(Σ)), n)
end

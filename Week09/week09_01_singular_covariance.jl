# =====================================================================
# Week 09 -- Script 1: Why the sample covariance matrix stops working
#
# Notes, Section 1 "Why We Need Factor Models"
#   1.1 The Parameter Counting Problem
#   1.2 What Actually Breaks                       -> Table "Sample covariance of 100 stocks..."
#   1.3 What the Factorization Cannot Tell Us      -> Table "A portfolio the sample covariance calls riskless"
#
# The story this script tells:
#   (a) With n = 100 stocks and T = 60 days, the sample covariance has
#       rank 59.  That is guaranteed, not bad luck.
#   (b) The Week 03 PSD Cholesky still "works" -- and that is the problem,
#       because everything it simulates lives in a 59-dim subspace.
#   (c) The missing directions are reported as having ZERO risk.  We build
#       a portfolio in one of those directions and watch it lose money.
#
# Run from anywhere:   julia --project=Week09 Week09/week09_01_singular_covariance.jl
# =====================================================================

include("week09_common.jl")

returns, stocks, R = load_returns()
T, n = size(R)

# ---------------------------------------------------------------------
# 1.1  Count the parameters
# ---------------------------------------------------------------------
# A symmetric n x n matrix has n diagonal entries plus n(n-1)/2 distinct
# off-diagonal entries: n(n+1)/2 free numbers.  We have T*n observations to
# estimate them with.  When those two numbers are about the same size we are
# asking each data point to pin down roughly one parameter -- hopeless.
println("=== 1.1 The parameter counting problem ===")
@printf("T = %d observations, n = %d assets\n", T, n)
@printf("Parameters in a full covariance matrix: %d\n", n * (n + 1) ÷ 2)
@printf("Data points available:                  %d\n", T * n)

# ---------------------------------------------------------------------
# 1.2  What actually breaks
# ---------------------------------------------------------------------
println("\n=== 1.2 What actually breaks ===")
S   = cov(R)                       # the sample covariance we would "normally" use
evS = eigvals(Symmetric(S))        # Symmetric() tells LAPACK to use the symmetric solver

# cov() demeans each column, then forms (1/(T-1)) Σ_t r_t r_t'.  That is a
# sum of T rank-one matrices, and demeaning removes one more degree of
# freedom, so the rank can be at most T-1 = 59.  With n = 100 > 59, at least
# 41 eigenvalues must be zero.
@printf("rank(S)             = %d   (= min(T-1, n) = %d)\n", rank(S), min(T - 1, n))

# The "zero" eigenvalues come out as tiny numbers of either sign, around
# 1e-18.  Those are floating-point rounding, not information.  A negative
# value here is why the PD Cholesky refuses the matrix.
@printf("smallest eigenvalue = %.3e   <- zero, up to rounding\n", minimum(evS))

# The condition number λmax/λmin should be infinite.  Whatever finite number
# prints is a report on rounding error (the notes call it "undefined").
@printf("condition number    = %.3e   <- meaningless; λmin is really 0\n",
        maximum(evS) / max(minimum(evS), eps()))

# isposdef attempts the standard (PD) Cholesky.  It fails because the matrix
# is only positive SEMI-definite.
println("isposdef(S)         = ", isposdef(Symmetric(S)), "   <- the PD Cholesky fails")

# Week 03's PSD Cholesky, chol_psd! from the course library
# (library/simulate.jl), handles a zero pivot by writing a column of zeros
# and moving on.  It succeeds -- and the number of columns that carry real
# variance is exactly the rank.
#
# chol_psd!(root, a) writes the root into a matrix you allocate; it does not
# return one.  The `!` in the name is Julia's convention for "modifies its
# argument".
L = zeros(n, n)
chol_psd!(L, S)

# Counting the zero columns takes a little care.  chol_psd! zeroes a pivot
# only when rounding pushes it slightly negative.  When rounding pushes it
# slightly positive instead, the column survives at ~1e-8 -- six orders of
# magnitude smaller than a real column.  Count both kinds as zero; see
# nonzero_cols() in week09_common.jl.
exact0 = count(j -> all(iszero, L[:, j]), 1:n)
nz = nonzero_cols(L)
@printf("PSD Cholesky SUCCEEDS: %d non-zero columns, %d zero columns\n", nz, n - nz)
@printf("  (%d exactly zero, %d at the ~1e-8 rounding level)\n", exact0, n - nz - exact0)

# The root really is a root: L*L' reproduces S to machine precision.
@printf("reconstruction error max|LL' - S| = %.1e\n", maximum(abs.(L * L' - S)))

# Why success is not reassuring: a simulated scenario is r = L*z.  Only 59
# columns of L are non-zero, so every r is a combination of 59 vectors, and
# those 59 vectors span the same space as the 60 demeaned historical days.
# No simulation from S can ever produce a move in the other 41 directions.
println("So we can factor S and simulate from it.  That is exactly the problem:")
@printf("every simulated scenario lies in the %d-dim span of the %d historical days.\n",
        nz, T)

# ---------------------------------------------------------------------
# 1.3  What the missing directions cost, measured out of sample
# ---------------------------------------------------------------------
# Split the 60 days in half.  Estimate on the first 30 (rank 29), so the
# null space -- the directions S1 says have zero variance -- is 100-29 = 71
# dimensional.  Then hold a portfolio in that null space over the NEXT 30
# days and see what really happens.
println("\n=== 1.3 A portfolio the sample covariance calls riskless ===")
est_days = 30
h1, h2 = R[1:est_days, :], R[est_days+1:end, :]   # estimation / test windows
S1 = cov(h1)

# The eigenvectors of S1 with (numerically) zero eigenvalue form a basis for
# its null space.  Any w = nullsp * c satisfies S1 * w = 0, i.e. w'S1w = 0.
e1 = eigen(Symmetric(S1))
nullsp = e1.vectors[:, e1.values .< 1e-14]
@printf("dimension of the null space of S1 = %d\n", size(nullsp, 2))

# We want a fully invested portfolio, 1'w = 1.  Writing w = nullsp * c turns
# that into one linear equation in c:  (1'nullsp) c = 1.  That is one
# equation in 71 unknowns; the backslash returns the minimum-norm solution,
# which keeps the weights from being absurd.
a  = reshape(ones(n)' * nullsp, 1, :)
wn = nullsp * (a \ [1.0])

# This is an ordinary-looking portfolio -- nothing about the weights screams
# "pathological".  That is what makes it dangerous.
@printf("weights: sum = %.4f   max |w| = %.1f%%   gross = %.1f\n",
        sum(wn), maximum(abs.(wn)) * 100, sum(abs.(wn)))

# Forecast vs realized.  Three numbers, matching the first three rows of the
# table in Notes 1.3.
#   - in-sample forecast  : sqrt(w'S1w), annualized.  Zero up to rounding.
#   - realized, same days : the actual P&L stdev over the 30 estimation days.
#                           Also zero, because S1 IS the covariance of those
#                           days -- the estimate is internally consistent.
#   - realized, next days : what actually happened out of sample.
@printf("  forecast vol (in-sample)      = %.2e %%   <- zero to numerical precision\n",
        annvol(max(wn' * S1 * wn, 0)))
@printf("  realized vol, same 30 days    = %.2e %%\n", std(h1 * wn) * sqrt(ANN) * 100)
@printf("  realized vol, NEXT 30 days    = %.2f%%     <- the estimate called this riskless\n",
        std(h2 * wn) * sqrt(ANN) * 100)

# The last row of the table: what would a factor model have said?  The FF3
# model is built step by step in week09_02_ff3_time_series.jl; here we just
# call the packaged version.  Because every stock gets a strictly positive
# specific variance d_i, w'Σw >= Σ w_i² d_i > 0 -- a factor model can never
# call a portfolio riskless (Notes 5.3).
ff = fit_ff3(returns, stocks)
@printf("  FF3 factor model forecast     = %.2f%%     <- not perfect, but honest\n",
        annvol(wn' * ff.Σ * wn))

# ---------------------------------------------------------------------
# The optimizer finds the null space for you
# ---------------------------------------------------------------------
# Notes 1.3, last paragraph.  Minimum variance is w = Σ⁻¹1 / 1'Σ⁻¹1, but S
# has no inverse.  The usual workaround is the pseudo-inverse pinv(S), which
# inverts the eigenvalues it judges to be non-zero and sets the rest to zero.
# "Judges" is the key word: pinv needs a tolerance below which an eigenvalue
# counts as zero.  That tolerance is a numerical choice, not information,
# and the answer depends heavily on it.
#
#   - A sensible tolerance (Julia's default) drops the 41 zero eigenvalues.
#     The resulting weights avoid the null space entirely, so they are not
#     the true minimizer -- the true minimum variance under S is exactly zero,
#     achieved by any null-space portfolio like the one above.
#   - A tolerance that is too tight keeps some of the ~1e-18 rounding-noise
#     eigenvalues, inverts them into ~1e18, and the optimizer piles into
#     those directions.  Weights explode.
#
# Either way the optimizer is not finding low risk.  It is finding whatever
# the estimate (or the cutoff you picked) says is low risk -- which is the
# point of Notes 1.4: patching zero eigenvalues turns "I know nothing here"
# into a specific, arbitrary number.
println()
for (label, rt) in [("default tolerance", nothing), ("rtol = 1e-16      ", 1e-16)]
    Sinv = rt === nothing ? pinv(S) : pinv(S; rtol = rt)
    u = Sinv * ones(n)
    w = u ./ sum(u)
    @printf("pinv min-variance, %s: largest long %6.1f%%  largest short %6.1f%%  gross %5.2f\n",
            label, maximum(w) * 100, minimum(w) * 100, sum(abs.(w)))
end

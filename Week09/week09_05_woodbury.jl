# =====================================================================
# Week 09 -- Script 5: Fast inversion with the Woodbury identity
#
# Notes, Section 5.4 "Fast Inversion, the Woodbury Identity"
#   -> max relative difference ~1e-15 ("it is the same answer")
#   -> Table "Cost of solving with and without the factor structure"
#
#   (D + BFB')⁻¹ = D⁻¹ - D⁻¹B (F⁻¹ + B'D⁻¹B)⁻¹ B'D⁻¹
#
# The only real inverse on the right is m x m.  Everything else is either
# a diagonal (n divisions) or a matrix-vector product.  Cost O(nm²) instead
# of O(n³).
#
# Run from anywhere:   julia --project=Week09 Week09/week09_05_woodbury.jl
# =====================================================================

include("week09_common.jl")

"""
    woodbury_solve(B, F, dv, x)

Return Σ⁻¹x for Σ = B*F*B' + Diagonal(dv) WITHOUT forming Σ.

Read it right to left, matching the identity:
    Di .* x                        D⁻¹x                 (n divisions)
    B' * (Di .* x)                 B'D⁻¹x               (m x 1)
    inner * (...)                  (F⁻¹+B'D⁻¹B)⁻¹ B'D⁻¹x (m x m times m x 1)
    Di .* (B * (...))              D⁻¹B(...)            (back to n x 1)
"""
function woodbury_solve(B, F, dv, x)
    Di = 1.0 ./ dv                                  # D⁻¹ is just reciprocals
    # B' .* Di' scales column i of B' by 1/d_i, i.e. forms B'D⁻¹ without
    # ever building an n x n diagonal matrix.
    inner = inv(inv(F) + (B' .* Di') * B)           # the ONLY inverse: m x m
    return Di .* x .- Di .* (B * (inner * (B' * (Di .* x))))
end

# ---------------------------------------------------------------------
# 1. Same answer, on the real FF3 model
# ---------------------------------------------------------------------
println("=== Woodbury vs direct solve on the FF3 model ===")
returns, stocks, _ = load_returns()
ff = fit_ff3(returns, stocks)
n = length(stocks)

b = ones(n)                       # Σ⁻¹1 is what minimum variance needs (Notes 8.1)
direct = ff.Σ \ b                 # dense LU on the full 100 x 100 matrix
wood   = woodbury_solve(ff.B, ff.Fcov, ff.Dv, b)
@printf("max relative difference: %.1e   (same answer, to rounding)\n",
        maximum(abs.(direct .- wood)) / maximum(abs.(direct)))

# ---------------------------------------------------------------------
# 2. How the cost grows with n
# ---------------------------------------------------------------------
# Our data only has 100 stocks, so build random factor models of larger
# size.  The values do not matter for timing -- only the shapes do.
#
# Timing notes:
#   - Every timed call is run once first, so we do not time compilation.
#   - We take the minimum over several repetitions; the minimum is the
#     least contaminated by whatever else the machine is doing.
#   - Your numbers will differ from the notes'.  The SHAPE should not: the
#     direct column grows roughly like n³, the Woodbury column barely moves.
println("\n=== Timing (ms): direct dense solve vs Woodbury, m = 3 ===")
Random.seed!(545)
mfac = 3
besttime(f, reps) = minimum(@elapsed(f()) for _ in 1:reps) * 1000

@printf("%8s %12s %12s %10s\n", "n", "direct", "Woodbury", "speedup")
for nn in [100, 500, 1000, 2000, 5000]
    Bn = randn(nn, mfac)
    Fn = Matrix(ff.Fcov)                       # reuse the real 3x3 factor covariance
    dn = rand(nn) .* 1e-4 .+ 1e-5              # positive specific variances
    xn = ones(nn)
    Σn = Bn * Fn * Bn' + Diagonal(dn)          # only the direct method needs this

    direct_solve() = Σn \ xn
    wood_solve()   = woodbury_solve(Bn, Fn, dn, xn)
    direct_solve(); wood_solve()               # warm-up (compile)

    reps = nn <= 1000 ? 20 : 3                 # big direct solves are slow; fewer reps
    td = besttime(direct_solve, reps)
    tw = besttime(wood_solve, 20)
    @printf("%8d %12.3f %12.3f %9.0fx\n", nn, td, tw, td / tw)
end
println("Direct is O(n³).  Woodbury is O(nm²) -- with m = 3, effectively O(n).")

# Memory, the other reason vendors keep Σ in factor form (Notes 5.4):
nbig = 8000
@printf("\nn = %d: dense Σ needs %.0f MB;  B, F, D need %.2f MB\n",
        nbig, nbig^2 * 8 / 1e6, (nbig * mfac + mfac^2 + nbig) * 8 / 1e6)

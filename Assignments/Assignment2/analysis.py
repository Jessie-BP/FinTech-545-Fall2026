"""FinTech 545 Assignment 2: complete reproducible analysis.

Run this file from the Assignment2 directory.  It reads problem1.csv through
problem5.csv, prints every reported numerical result, saves figures in figures/,
and writes machine-readable results to results.json for the report builder.

Conventions
-----------
* Arithmetic returns are used throughout.
* VaR and ES are positive dollar loss numbers.
* Sample moments use the usual bias-corrected estimators and excess kurtosis.
* Monte Carlo uses fixed seeds so every reported number is reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
ALPHA = 0.05
RNG_SEED = 5452026


def py(value):
    """Convert NumPy/Pandas objects to JSON-safe built-in objects."""
    if isinstance(value, dict):
        return {str(k): py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [py(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def moments(x: np.ndarray | pd.Series) -> dict[str, float]:
    """First four sample moments; kurtosis is excess kurtosis."""
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "variance": float(np.var(x, ddof=1)),
        "skewness": float(stats.skew(x, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(x, fisher=True, bias=False)),
    }


def historical_var_es(pnl: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    """Historical VaR and ES from a P&L sample, both reported as losses.

    VaR uses the linear sample quantile.  ES averages observations at or below
    that P&L quantile, matching the numerical definition used in Week 5.
    """
    pnl = np.asarray(pnl, dtype=float)
    cutoff = float(np.quantile(pnl, alpha, method="linear"))
    return -cutoff, -float(np.mean(pnl[pnl <= cutoff]))


def aicc(log_likelihood: float, k: int, n: int) -> float:
    base = 2 * k - 2 * log_likelihood
    return base + (2 * k * (k + 1)) / (n - k - 1) if k else base


def bic(log_likelihood: float, k: int, n: int) -> float:
    return k * np.log(n) - 2 * log_likelihood


def fit_t(x: np.ndarray, minimum_df: float = 2.0001) -> dict[str, float]:
    """MLE fit of a location-scale Student t with finite variance."""
    x = np.asarray(x, dtype=float)
    mu0, sd0 = float(np.mean(x)), float(np.std(x, ddof=0))

    def nll(theta):
        df = minimum_df + np.exp(theta[0])
        scale = np.exp(theta[2])
        return -float(np.sum(stats.t.logpdf(x, df=df, loc=theta[1], scale=scale)))

    starts = [
        np.array([np.log(6 - minimum_df), mu0, np.log(sd0)]),
        np.array([np.log(12 - minimum_df), mu0, np.log(sd0)]),
        np.array([np.log(30 - minimum_df), mu0, np.log(sd0)]),
    ]
    fits = [optimize.minimize(nll, s, method="L-BFGS-B") for s in starts]
    fit = min(fits, key=lambda z: z.fun)
    df = minimum_df + np.exp(fit.x[0])
    return {
        "df": float(df),
        "loc": float(fit.x[1]),
        "scale": float(np.exp(fit.x[2])),
        "log_likelihood": float(-fit.fun),
        "success": bool(fit.success),
    }


def rebonato_jackel(corr: np.ndarray) -> np.ndarray:
    """Spectral Rebonato-Jaeckel repair followed by diagonal rescaling."""
    vals, vecs = np.linalg.eigh((corr + corr.T) / 2)
    vals = np.maximum(vals, 0.0)
    repaired = (vecs * vals) @ vecs.T
    scale = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(scale, scale)
    np.fill_diagonal(repaired, 1.0)
    return (repaired + repaired.T) / 2


def higham_nearest_correlation(corr: np.ndarray, tolerance: float = 1e-12,
                               max_iterations: int = 10000) -> tuple[np.ndarray, int]:
    """Higham's alternating projections algorithm with Dykstra correction."""
    y = np.array(corr, dtype=float, copy=True)
    delta_s = np.zeros_like(y)
    for iteration in range(1, max_iterations + 1):
        r = y - delta_s
        vals, vecs = np.linalg.eigh((r + r.T) / 2)
        x = (vecs * np.maximum(vals, 0.0)) @ vecs.T
        delta_s = x - r
        old_y = y
        y = x.copy()
        np.fill_diagonal(y, 1.0)
        y = (y + y.T) / 2
        if np.linalg.norm(y - old_y, "fro") <= tolerance * max(1.0, np.linalg.norm(y, "fro")):
            return y, iteration
    raise RuntimeError("Higham algorithm did not converge")


def covariance_from_corr(corr: np.ndarray, standard_deviations: np.ndarray) -> np.ndarray:
    return np.diag(standard_deviations) @ corr @ np.diag(standard_deviations)


def problem1() -> dict:
    df = pd.read_csv(ROOT / "problem1.csv")
    columns = ["A", "B", "C", "D", "IDX"]
    x = df[columns]

    joint_counts = pd.DataFrame(index=columns, columns=columns, dtype=int)
    for a in columns:
        for b in columns:
            joint_counts.loc[a, b] = int(df[[a, b]].dropna().shape[0])
    all_five = int(x.dropna().shape[0])

    complete = x.dropna().corr().to_numpy()
    pairwise = x.corr(min_periods=2).to_numpy()
    eig_complete = np.linalg.eigvalsh(complete)
    eig_pairwise = np.linalg.eigvalsh(pairwise)

    def cholesky_status(matrix):
        try:
            np.linalg.cholesky(matrix)
            return "succeeded"
        except np.linalg.LinAlgError as error:
            return f"failed: {error}"

    standard_deviations = x.std(ddof=1).to_numpy()
    portfolio = np.array([-0.40, -0.30, -0.20, -0.10, 1.0])

    def tracking_variance(corr):
        cov = covariance_from_corr(corr, standard_deviations)
        return float(portfolio @ cov @ portfolio)

    rj = rebonato_jackel(pairwise)
    higham, higham_iterations = higham_nearest_correlation(pairwise)

    delta = np.abs(higham - pairwise)
    changes = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            changes.append({
                "pair": f"{columns[i]}-{columns[j]}",
                "before": float(pairwise[i, j]),
                "after": float(higham[i, j]),
                "absolute_change": float(delta[i, j]),
                "joint_days": int(joint_counts.iloc[i, j]),
            })
    changes.sort(key=lambda z: z["absolute_change"], reverse=True)

    result = {
        "joint_counts": joint_counts.astype(int).to_dict(orient="index"),
        "all_five_days": all_five,
        "complete_correlation": dict(zip(columns, complete.tolist())),
        "pairwise_correlation": dict(zip(columns, pairwise.tolist())),
        "complete_eigenvalues": eig_complete,
        "pairwise_eigenvalues": eig_pairwise,
        "complete_cholesky": cholesky_status(complete),
        "pairwise_cholesky": cholesky_status(pairwise),
        "standard_deviations": dict(zip(columns, standard_deviations)),
        "tracking_variance_pairwise": tracking_variance(pairwise),
        "rj_min_eigenvalue": float(np.min(np.linalg.eigvalsh(rj))),
        "rj_frobenius_distance": float(np.linalg.norm(rj - pairwise, "fro")),
        "rj_tracking_variance": tracking_variance(rj),
        "higham_min_eigenvalue": float(np.min(np.linalg.eigvalsh(higham))),
        "higham_frobenius_distance": float(np.linalg.norm(higham - pairwise, "fro")),
        "higham_tracking_variance": tracking_variance(higham),
        "higham_iterations": higham_iterations,
        "largest_higham_changes": changes[:5],
        "complete_pairwise_frobenius_gap": float(np.linalg.norm(complete - pairwise, "fro")),
    }
    return result


def recent_regime_length(returns: np.ndarray, min_segment: int = 20) -> int:
    """Length of the recent high-volatility stretch.

    Looks for the single break point that maximises the Gaussian likelihood of
    a two-variance model (one variance before the break, one after).  This is
    only used to size the "recent regime" the prompt asks us to eyeball from
    the plot, not as a formal test.
    """
    n = len(returns)
    best_k, best_ll = None, -np.inf
    for k in range(min_segment, n - min_segment):
        ll = -0.5 * (k * np.log(np.var(returns[:k])) + (n - k) * np.log(np.var(returns[k:])))
        if ll > best_ll:
            best_k, best_ll = k, ll
    return n - best_k


def problem2() -> dict:
    df = pd.read_csv(ROOT / "problem2.csv")
    raw_returns = df["Price"].pct_change().dropna().to_numpy()
    returns = raw_returns - np.mean(raw_returns)
    n = len(returns)
    recent = recent_regime_length(returns)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(np.arange(1, n + 1), 100 * returns, lw=0.8, color="#214e80")
    ax.axvline(n - recent, color="#bb3e03", ls="--", lw=1.2, label=f"last {recent} days")
    ax.set(xlabel="Return day", ylabel="Demeaned arithmetic return (%)",
           title="Problem 2: volatility jumps near the end of the sample")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "problem2_returns.png", dpi=180)
    plt.close(fig)

    equal_sd = float(np.std(returns, ddof=1))
    normal_equal = -1_000_000 * stats.norm.ppf(ALPHA, loc=0, scale=equal_sd)

    ew = {}
    for lam in [0.94, 0.97]:
        # Oldest to newest, with the newest observation receiving the most weight.
        raw_weights = lam ** np.arange(n - 1, -1, -1)
        weights = raw_weights / raw_weights.sum()
        ew_variance = float(np.sum(weights * returns**2))
        sd = np.sqrt(ew_variance)
        regime_weight = float(np.sum(weights[-recent:]))
        ew[str(lam)] = {
            "sd": sd,
            "var": float(-1_000_000 * stats.norm.ppf(ALPHA, loc=0, scale=sd)),
            "effective_sample_size_finite": float(1 / np.sum(weights**2)),
            "effective_sample_size_infinite": float((1 + lam) / (1 - lam)),
            "half_life": float(np.log(0.5) / np.log(lam)),
            "weight_recent": regime_weight,
            "volatility_relative_se": float(1 / np.sqrt(2 * (1 / np.sum(weights**2)))),
        }

    t_fit = fit_t(returns)
    t_var = -1_000_000 * stats.t.ppf(
        ALPHA, df=t_fit["df"], loc=t_fit["loc"], scale=t_fit["scale"]
    )
    historical_var, _ = historical_var_es(1_000_000 * returns)

    first_sd = float(np.std(returns[:-recent], ddof=1))
    recent_sd = float(np.std(returns[-recent:], ddof=1))
    sampling_noise = {}
    for lam in [0.94, 0.97]:
        info = ew[str(lam)]
        sigma_se = info["sd"] / np.sqrt(2 * info["effective_sample_size_finite"])
        sampling_noise[str(lam)] = {
            "sigma_standard_error": float(sigma_se),
            "var_standard_error_approx": float(abs(stats.norm.ppf(ALPHA)) * 1_000_000 * sigma_se),
        }

    return {
        "observations": n,
        "moments": moments(returns),
        "equal_weight_sd": equal_sd,
        "normal_equal_var": float(normal_equal),
        "ewma": ew,
        "t_fit": t_fit,
        "t_var": float(t_var),
        "historical_var": float(historical_var),
        "regime_days": recent,
        "old_regime_sd": first_sd,
        "recent_regime_sd": recent_sd,
        "ewma_var_gap": float(abs(ew["0.94"]["var"] - ew["0.97"]["var"])),
        "sampling_noise": sampling_noise,
    }


def problem3() -> dict:
    df = pd.read_csv(ROOT / "problem3.csv")
    a, b = df["A"].to_numpy(), df["B"].to_numpy()
    pnl = {
        "A_1m": 1_000_000 * a,
        "B_1m": 1_000_000 * b,
        "A_2m": 2_000_000 * a,
        "diversified": 1_000_000 * (a + b),
    }

    fig, ax = plt.subplots(figsize=(9, 4.3))
    bins = np.linspace(min(pnl["A_2m"].min(), pnl["diversified"].min()),
                       max(pnl["A_2m"].max(), pnl["diversified"].max()), 80)
    ax.hist(pnl["A_2m"] / 1000, bins=bins / 1000, alpha=0.58, density=True,
            label="$2m in A")
    ax.hist(pnl["diversified"] / 1000, bins=bins / 1000, alpha=0.58, density=True,
            label="$1m in each")
    ax.set(xlabel="One-year P&L ($000)", ylabel="Density",
           title="Problem 3: concentrated and diversified bond P&L")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "problem3_pnl.png", dpi=180)
    plt.close(fig)

    historical_5 = {}
    historical_1 = {}
    normal_5 = {}
    for name, values in pnl.items():
        historical_5[name] = dict(zip(["var", "es"], historical_var_es(values, 0.05)))
        historical_1[name] = dict(zip(["var", "es"], historical_var_es(values, 0.01)))
        mu, sd = float(np.mean(values)), float(np.std(values, ddof=1))
        normal_5[name] = float(-stats.norm.ppf(0.05, loc=mu, scale=sd))

    loss_a = a < -0.20
    loss_b = b < -0.20
    return {
        "moments": {"A": moments(a), "B": moments(b)},
        "loss_count_A": int(np.sum(loss_a)),
        "loss_count_B": int(np.sum(loss_b)),
        "loss_count_either": int(np.sum(loss_a | loss_b)),
        "loss_count_both": int(np.sum(loss_a & loss_b)),
        "historical_5pct": historical_5,
        "normal_5pct": normal_5,
        "historical_1pct": historical_1,
        "var_subadditivity_lhs": float(historical_5["A_1m"]["var"] + historical_5["B_1m"]["var"]),
        "var_subadditivity_rhs": float(historical_5["diversified"]["var"]),
        "es_subadditivity_lhs": float(historical_5["A_1m"]["es"] + historical_5["B_1m"]["es"]),
        "es_subadditivity_rhs": float(historical_5["diversified"]["es"]),
    }


def fit_margin(x: np.ndarray) -> dict:
    """Fit normal and t margins and select by AICc."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    normal = {"loc": float(np.mean(x)), "scale": float(np.std(x, ddof=0))}
    normal["log_likelihood"] = float(np.sum(stats.norm.logpdf(x, **normal)))
    normal["aicc"] = float(aicc(normal["log_likelihood"], 2, n))
    t_model = fit_t(x)
    t_model["aicc"] = float(aicc(t_model["log_likelihood"], 3, n))
    choice = "t" if t_model["aicc"] < normal["aicc"] else "normal"
    return {"normal": normal, "t": t_model, "choice": choice}


def margin_cdf(x: np.ndarray, model: dict) -> np.ndarray:
    if model["choice"] == "normal":
        p = model["normal"]
        u = stats.norm.cdf(x, loc=p["loc"], scale=p["scale"])
    else:
        p = model["t"]
        u = stats.t.cdf(x, df=p["df"], loc=p["loc"], scale=p["scale"])
    return np.clip(u, 1e-10, 1 - 1e-10)


def margin_ppf(u: np.ndarray, model: dict) -> np.ndarray:
    u = np.clip(u, 1e-10, 1 - 1e-10)
    if model["choice"] == "normal":
        p = model["normal"]
        return stats.norm.ppf(u, loc=p["loc"], scale=p["scale"])
    p = model["t"]
    return stats.t.ppf(u, df=p["df"], loc=p["loc"], scale=p["scale"])


def gaussian_copula_log_terms(u: np.ndarray, corr: np.ndarray) -> np.ndarray:
    z = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    return stats.multivariate_normal.logpdf(z, mean=np.zeros(z.shape[1]), cov=corr) - np.sum(
        stats.norm.logpdf(z), axis=1
    )


def t_copula_log_terms(u: np.ndarray, corr: np.ndarray, df: float) -> np.ndarray:
    z = stats.t.ppf(np.clip(u, 1e-10, 1 - 1e-10), df=df)
    return stats.multivariate_t.logpdf(z, loc=np.zeros(z.shape[1]), shape=corr, df=df) - np.sum(
        stats.t.logpdf(z, df=df), axis=1
    )


def tail_counts(u: np.ndarray, threshold: float = 0.025) -> dict[str, dict[str, int]]:
    names = ["X1-X2", "X1-X3", "X2-X3"]
    pairs = [(0, 1), (0, 2), (1, 2)]
    out = {}
    for name, (i, j) in zip(names, pairs):
        out[name] = {
            "lower": int(np.sum((u[:, i] < threshold) & (u[:, j] < threshold))),
            "upper": int(np.sum((u[:, i] > 1 - threshold) & (u[:, j] > 1 - threshold))),
        }
    return out


def problem4() -> dict:
    df = pd.read_csv(ROOT / "problem4.csv")
    names = ["X1", "X2", "X3"]
    x = df[names].to_numpy()
    n, d = x.shape

    # Prediction plots and counts use ranks, not fitted margins.
    ranks = np.column_stack([stats.rankdata(x[:, j]) / (n + 1) for j in range(d)])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        ax.scatter(ranks[:, i], ranks[:, j], s=8, alpha=0.45, color="#214e80")
        ax.axvline(0.025, color="#bb3e03", lw=0.8, ls="--")
        ax.axhline(0.025, color="#bb3e03", lw=0.8, ls="--")
        ax.axvline(0.975, color="#bb3e03", lw=0.8, ls="--")
        ax.axhline(0.975, color="#bb3e03", lw=0.8, ls="--")
        ax.set(xlabel=names[i], ylabel=names[j], xlim=(0, 1), ylim=(0, 1))
    fig.suptitle("Problem 4: rank-scaled pair plots")
    fig.tight_layout()
    fig.savefig(FIGURES / "problem4_rank_pairs.png", dpi=180)
    plt.close(fig)

    margin_fits = {name: fit_margin(x[:, i]) for i, name in enumerate(names)}
    u = np.column_stack([margin_cdf(x[:, i], margin_fits[name]) for i, name in enumerate(names)])

    tau = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau[i, j] = tau[j, i] = stats.kendalltau(x[:, i], x[:, j]).statistic
    corr = np.sin(np.pi * tau / 2)
    np.fill_diagonal(corr, 1.0)
    if np.min(np.linalg.eigvalsh(corr)) < 0:
        corr, _ = higham_nearest_correlation(corr)

    gaussian_terms = gaussian_copula_log_terms(u, corr)
    gaussian_ll = float(np.sum(gaussian_terms))

    # Search theta=1/nu on nu in [2.01, 100], as in the Week 5 notes.
    def objective_theta(theta):
        nu = 1 / float(theta)
        return -float(np.sum(t_copula_log_terms(u, corr, nu)))

    fit = optimize.minimize_scalar(
        objective_theta, bounds=(1 / 100, 1 / 2.01), method="bounded",
        options={"xatol": 1e-10}
    )
    nu = float(1 / fit.x)
    t_terms = t_copula_log_terms(u, corr, nu)
    t_ll = float(np.sum(t_terms))

    # Both copulas share the same normal draws (and the t copula adds one chi-square
    # mixing draw per day), so differences between them are not Monte Carlo noise.
    rng = np.random.default_rng(RNG_SEED)
    z = rng.multivariate_normal(np.zeros(d), corr, size=100_000)
    chi = rng.chisquare(nu, size=100_000)
    simulations = {}
    for copula in ["gaussian", "t"]:
        if copula == "gaussian":
            sim_u = stats.norm.cdf(z)
        else:
            t_draw = z / np.sqrt(chi[:, None] / nu)
            sim_u = stats.t.cdf(t_draw, df=nu)
        simulated_returns = np.column_stack([
            margin_ppf(sim_u[:, i], margin_fits[name]) for i, name in enumerate(names)
        ])
        portfolio_pnl = 1_000_000 * np.sum(simulated_returns, axis=1)
        v5, e5 = historical_var_es(portfolio_pnl, 0.05)
        v1, e1 = historical_var_es(portfolio_pnl, 0.01)
        counts = tail_counts(sim_u)
        # Scale 100,000 simulated observations to an expected count in 1,000 days.
        implied_counts = {
            pair: {tail: value / 100.0 for tail, value in tails.items()}
            for pair, tails in counts.items()
        }
        simulations[copula] = {
            "var_5": v5, "es_5": e5, "var_1": v1, "es_1": e1,
            "implied_tail_counts_per_1000": implied_counts,
        }

    historical_pnl = 1_000_000 * np.sum(x, axis=1)
    hist_v5, hist_e5 = historical_var_es(historical_pnl, 0.05)
    hist_v1, hist_e1 = historical_var_es(historical_pnl, 0.01)

    difference = t_terms - gaussian_terms
    middle = np.all((u >= 0.05) & (u <= 0.95), axis=1)
    middle_contribution = float(np.sum(difference[middle]))
    total_contribution = float(np.sum(difference))

    off_diagonal = np.triu(corr, 1)
    most_correlated = np.unravel_index(np.argmax(off_diagonal), off_diagonal.shape)
    rho = float(corr[most_correlated])
    tail_dependence = float(2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1))

    # X2's very large negative observation is the conspicuous single point in
    # the moment table.  Recompute its shape after deleting only that point.
    x2 = x[:, 1]
    x2_without_min = np.delete(x2, np.argmin(x2))
    return {
        "moments": {name: moments(x[:, i]) for i, name in enumerate(names)},
        "x2_skewness_without_minimum": float(stats.skew(x2_without_min, bias=False)),
        "x2_excess_kurtosis_without_minimum": float(stats.kurtosis(x2_without_min, fisher=True, bias=False)),
        "rank_tail_counts": tail_counts(ranks),
        "independent_expected_pair_count": float(n * 0.025**2),
        "margin_fits": margin_fits,
        "kendall_tau": dict(zip(names, tau.tolist())),
        "copula_correlation": dict(zip(names, corr.tolist())),
        "copula_min_eigenvalue": float(np.min(np.linalg.eigvalsh(corr))),
        "gaussian": {
            "log_likelihood": gaussian_ll,
            "aicc": float(aicc(gaussian_ll, 0, n)),
            "bic": float(bic(gaussian_ll, 0, n)),
        },
        "t": {
            "df": nu,
            "log_likelihood": t_ll,
            "aicc": float(aicc(t_ll, 1, n)),
            "bic": float(bic(t_ll, 1, n)),
            "optimization_success": bool(fit.success),
        },
        "delta_bic_in_favor_of_t": float(bic(gaussian_ll, 0, n) - bic(t_ll, 1, n)),
        "simulation": simulations,
        "historical": {"var_5": hist_v5, "es_5": hist_e5, "var_1": hist_v1, "es_1": hist_e1},
        "middle_days": int(np.sum(middle)),
        "middle_loglik_contribution": middle_contribution,
        "total_loglik_improvement": total_contribution,
        "middle_share_of_total": float(middle_contribution / total_contribution),
        "most_correlated_pair": f"{names[most_correlated[0]]}-{names[most_correlated[1]]}",
        "most_correlated_rho": rho,
        "t_tail_dependence": tail_dependence,
        "gaussian_tail_dependence": 0.0,
    }


def problem5() -> dict:
    prices = pd.read_csv(ROOT / "problem5.csv")
    # Regress on the raw returns so alpha is a real estimate; zero expected
    # returns are imposed afterwards (alpha is dropped when simulating, and
    # cov() below demeans automatically).
    returns = prices[["MKT", "A", "B"]].pct_change().dropna()
    market = returns["MKT"].to_numpy()
    n = len(market)
    design = np.column_stack([np.ones(n), market])

    coefficients, residuals = {}, {}
    for stock in ["A", "B"]:
        y = returns[stock].to_numpy()
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        error = y - design @ beta
        coefficients[stock] = {"alpha": float(beta[0]), "beta": float(beta[1])}
        residuals[stock] = error

    residual_matrix = np.column_stack([residuals["A"], residuals["B"]])
    residual_cov = np.cov(residual_matrix, rowvar=False, ddof=1)
    residual_corr = float(np.corrcoef(residual_matrix, rowvar=False)[0, 1])
    residual_sd = np.sqrt(np.diag(residual_cov))

    rng = np.random.default_rng(RNG_SEED + 1)
    market_sim = rng.normal(0, np.std(market, ddof=1), size=100_000)
    # Same standard normal draws for both residual assumptions, so the gap
    # between them comes only from the residual correlation.
    std_normal = rng.standard_normal((100_000, 2))
    simulated = {}
    for label, covariance in [
        ("full_residual_covariance", residual_cov),
        ("independent_residuals", np.diag(np.diag(residual_cov))),
    ]:
        eps = std_normal @ np.linalg.cholesky(covariance).T
        a_sim = coefficients["A"]["beta"] * market_sim + eps[:, 0]
        b_sim = coefficients["B"]["beta"] * market_sim + eps[:, 1]
        p1 = 1_000_000 * (a_sim + b_sim)
        p2 = 1_000_000 * (a_sim - b_sim)
        simulated[label] = {
            "P1_var": float(historical_var_es(p1, 0.05)[0]),
            "P2_var": float(historical_var_es(p2, 0.05)[0]),
        }

    stock_cov = returns[["A", "B"]].cov().to_numpy()
    z = -stats.norm.ppf(0.05)
    delta_normal = {}
    for label, weights in [("P1", np.array([1_000_000, 1_000_000])),
                           ("P2", np.array([1_000_000, -1_000_000]))]:
        delta_normal[label] = float(z * np.sqrt(weights @ stock_cov @ weights))

    return {
        "observations": n,
        "coefficients": coefficients,
        "residual_standard_deviations": {"A": float(residual_sd[0]), "B": float(residual_sd[1])},
        "residual_correlation": residual_corr,
        "residual_covariance": residual_cov,
        "simulation": simulated,
        "delta_normal_var": delta_normal,
    }


def main():
    results = {
        "metadata": {
            "alpha": ALPHA,
            "simulation_draws": 100_000,
            "random_seed": RNG_SEED,
        },
        "problem1": problem1(),
        "problem2": problem2(),
        "problem3": problem3(),
        "problem4": problem4(),
        "problem5": problem5(),
    }
    output = ROOT / "results.json"
    output.write_text(json.dumps(py(results), indent=2), encoding="utf-8")
    print(json.dumps(py(results), indent=2))
    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()

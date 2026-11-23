"""Solutions and checks for Functional Test 2 (Tests 8.1 through 13.9).

Run from any directory: python3 FunctionalTest2/functional_test2.py
The random simulation answers are compared statistically because Julia and
NumPy use different random streams.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "FunctionalTest1"))
from functional_test1 import fit_general_t, fit_normal


def empirical_var_es(x, alpha=0.05):
    """Match the course's midpoint order statistic and inclusive ES tail."""
    values = np.sort(np.asarray(x, dtype=float))
    n = len(values)
    lo = max(1, int(np.floor(n * alpha))) - 1
    hi = max(1, int(np.ceil(n * alpha))) - 1
    threshold = (values[lo] + values[hi]) / 2
    return -threshold, -values[values <= threshold].mean()


def distribution_risk(mu, sigma, nu=None, alpha=0.05):
    if nu is None:
        q = stats.norm.ppf(alpha)
        es_centered = sigma * stats.norm.pdf(q) / alpha
    else:
        q = stats.t.ppf(alpha, nu)
        es_centered = sigma * (nu + q * q) / ((nu - 1) * alpha) * stats.t.pdf(q, nu)
    return -mu - sigma * q, -sigma * q, es_centered - mu, es_centered


def risk_table(returns, values):
    rows = []
    for name, current_value, column in zip(values.Stock, values.currentValue, returns.T):
        var, es = empirical_var_es(current_value * column)
        rows.append((name, var, es, var / current_value, es / current_value))
    total_value = values.currentValue.sum()
    var, es = empirical_var_es(returns @ values.currentValue.to_numpy())
    rows.append(("Total", var, es, var / total_value, es / total_value))
    return pd.DataFrame(rows, columns=["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"])


def risk_parity(covariance, budget):
    covariance = np.asarray(covariance)
    budget = np.asarray(budget, dtype=float)
    target = budget / budget.sum()

    def objective(w):
        contribution = w * (covariance @ w)
        share = contribution / contribution.sum()
        return np.sum((share - target) ** 2)

    n = len(budget)
    result = optimize.minimize(objective, np.full(n, 1 / n), method="SLSQP",
                               bounds=[(1e-8, 1)] * n,
                               constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                               options={"ftol": 1e-14, "maxiter": 5000})
    if not result.success and result.fun > 1e-10:
        raise RuntimeError(result.message)
    return result.x


def max_sharpe(covariance, means, rf, bounds):
    excess = np.asarray(means) - rf

    def objective(w):
        return -(w @ excess) / np.sqrt(w @ covariance @ w)

    n = len(excess)
    result = optimize.minimize(objective, np.full(n, 1 / n), method="SLSQP",
                               bounds=bounds,
                               constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                               options={"ftol": 1e-13, "maxiter": 5000})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def expost_attribution(weights, stock_returns, factor_returns, betas):
    """Dynamic weights, Carino return attribution and regression SD attribution."""
    stocks = stock_returns.to_numpy(dtype=float)
    factors = factor_returns.to_numpy(dtype=float)
    w = np.asarray(weights, dtype=float).copy()
    betas = np.asarray(betas, dtype=float)
    portfolio = np.empty(len(stocks))
    factor_weights = np.empty((len(stocks), factors.shape[1]))
    residual = np.empty(len(stocks))
    for i, row in enumerate(stocks):
        factor_weights[i] = w @ betas
        new_w = w * (1 + row)
        portfolio[i] = new_w.sum() - 1
        w = new_w / new_w.sum()
        residual[i] = portfolio[i] - factor_weights[i] @ factors[i]
    components = np.column_stack((factor_weights * factors, residual))
    total_return = np.expm1(np.log1p(portfolio).sum())
    k = np.log1p(total_return) / total_return
    carino = (np.log1p(portfolio) / portfolio) / k
    component_names = list(factor_returns.columns) + ["Alpha"]
    series = np.column_stack((factors, residual, portfolio))
    totals = np.expm1(np.log1p(series).sum(axis=0))
    attribution = np.r_[np.sum(components * carino[:, None], axis=0), total_return]
    design = np.column_stack((np.ones(len(portfolio)), portfolio))
    beta = np.linalg.lstsq(design, components, rcond=None)[0][1]
    volatility = np.r_[beta * np.std(portfolio, ddof=1), np.std(portfolio, ddof=1)]
    return pd.DataFrame([totals, attribution, volatility],
                        columns=component_names + ["Portfolio"]).assign(
                            Value=["TotalReturn", "Return Attribution", "Vol Attribution"]
                        )[["Value"] + component_names + ["Portfolio"]]


def gbsm(call, underlying, strike, t, rf, carry, vol):
    d1 = (np.log(underlying / strike) + (carry + vol * vol / 2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    sign = 1 if call else -1
    discount_stock = np.exp((carry - rf) * t)
    discount_strike = np.exp(-rf * t)
    value = sign * (underlying * discount_stock * stats.norm.cdf(sign * d1)
                    - strike * discount_strike * stats.norm.cdf(sign * d2))
    delta = sign * discount_stock * stats.norm.cdf(sign * d1)
    gamma = discount_stock * stats.norm.pdf(d1) / (underlying * vol * np.sqrt(t))
    vega = underlying * discount_stock * stats.norm.pdf(d1) * np.sqrt(t)
    rho = sign * t * strike * discount_strike * stats.norm.cdf(sign * d2)
    if call:
        theta = (-underlying * discount_stock * stats.norm.pdf(d1) * vol / (2 * np.sqrt(t))
                 - (carry - rf) * underlying * discount_stock * stats.norm.cdf(d1)
                 - rf * strike * discount_strike * stats.norm.cdf(d2))
    else:
        theta = (-underlying * discount_stock * stats.norm.pdf(d1) * vol / (2 * np.sqrt(t))
                 + (carry - rf) * underlying * discount_stock * stats.norm.cdf(-d1)
                 + rf * strike * discount_strike * stats.norm.cdf(-d2))
    return value, delta, gamma, vega, rho, theta


def american_tree(call, underlying, strike, t, rf, carry, vol, steps):
    """Recombining CRR tree; accepts an array of initial spot prices."""
    spots = np.asarray(underlying, dtype=float)
    dt = t / steps
    up = np.exp(vol * np.sqrt(dt))
    down = 1 / up
    p = (np.exp(carry * dt) - down) / (up - down)
    discount = np.exp(-rf * dt)
    sign = 1 if call else -1
    powers = up ** np.arange(steps + 1) * down ** np.arange(steps, -1, -1)
    option = np.maximum(sign * (spots[..., None] * powers - strike), 0)
    for j in range(steps - 1, -1, -1):
        option = discount * (p * option[..., 1:] + (1 - p) * option[..., :-1])
        powers = powers[:-1] / down
        option = np.maximum(option, np.maximum(sign * (spots[..., None] * powers - strike), 0))
    return option[..., 0]


def american_discrete_dividends(call, spot, strike, t, rf, vol, steps, dates, amounts):
    """Backward valuation at cash-dividend dates on the CRR lattice."""
    if not dates or dates[0] > steps:
        return float(american_tree(call, spot, strike, t, rf, rf, vol, steps))
    dt = t / steps
    up = np.exp(vol * np.sqrt(dt))
    down = 1 / up
    p = (np.exp(rf * dt) - down) / (up - down)
    discount = np.exp(-rf * dt)
    sign = 1 if call else -1
    first = dates[0]
    prices = spot * up ** np.arange(first + 1) * down ** np.arange(first, -1, -1)
    if len(dates) == 1:
        continuation = american_tree(call, prices - amounts[0], strike,
                                     t - first * dt, rf, rf, vol, steps - first)
    else:
        continuation = np.array([
            american_discrete_dividends(call, float(s - amounts[0]), strike,
                                        t - first * dt, rf, vol, steps - first,
                                        [d - first for d in dates[1:]], amounts[1:])
            for s in prices
        ])
    option = np.maximum(continuation, np.maximum(sign * (prices - strike), 0))
    for j in range(first - 1, -1, -1):
        option = discount * (p * option[1:] + (1 - p) * option[:-1])
        prices = prices[:-1] / down
        option = np.maximum(option, np.maximum(sign * (prices - strike), 0))
    return float(option[0])


def kendall_correlation(x):
    x = np.asarray(x)
    n = x.shape[1]
    correlation = np.eye(n)
    for i in range(n):
        for j in range(i):
            correlation[i, j] = correlation[j, i] = np.sin(
                np.pi * stats.kendalltau(x[:, i], x[:, j]).statistic / 2)
    if np.linalg.eigvalsh(correlation).min() < 1e-8:
        correlation = (correlation + 1e-8 * np.eye(n)) / (1 + 1e-8)
    return correlation


def profile_nu(log_likelihood):
    grid = np.linspace(0.01, 0.49, 200)
    index = int(np.argmax([log_likelihood(1 / theta) for theta in grid]))
    fine = np.linspace(grid[max(index - 1, 0)], grid[min(index + 1, 199)], 200)
    values = np.array([log_likelihood(1 / theta) for theta in fine])
    index = int(np.argmax(values))
    return 1 / fine[index], values[index]


def multivariate_t_fit(x, correlation):
    x = np.asarray(x)
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)

    def scale(nu):
        return np.outer(sd, sd) * correlation * (nu - 2) / nu

    def log_likelihood(nu):
        return stats.multivariate_t.logpdf(x, loc=mu, shape=scale(nu), df=nu).sum()

    nu, ll = profile_nu(log_likelihood)
    return mu, scale(nu), nu, ll


def copula_fit(uniforms, correlation):
    n, dimension = uniforms.shape
    z = stats.norm.ppf(uniforms)
    inverse = np.linalg.inv(correlation)
    logdet = np.linalg.slogdet(correlation)[1]
    ll_gaussian = -n * logdet / 2 - np.einsum("ni,ij,nj->", z, inverse - np.eye(dimension), z) / 2

    def t_log_likelihood(nu):
        t = stats.t.ppf(uniforms, nu)
        joint = stats.multivariate_t.logpdf(t, shape=correlation, df=nu).sum()
        margins = stats.t.logpdf(t, nu).sum()
        return joint - margins

    nu, ll_t = profile_nu(t_log_likelihood)
    return ll_gaussian, nu, ll_t


def simulate_t_copula(correlation, nu, count, seed=13):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(len(correlation)), correlation, size=count)
    chi_square = np.random.default_rng(seed + 1).chisquare(nu, size=count)
    return stats.t.cdf(z * np.sqrt(nu / chi_square)[:, None], nu)


def matrix_frame(array):
    return pd.DataFrame(array, columns=[f"x{i + 1}" for i in range(np.shape(array)[1])])


def run_functional_tests(data_directory):
    d = Path(data_directory)
    outputs = {}

    normal_data = pd.read_csv(d / "test7_1.csv").iloc[:, 0].to_numpy()
    t_data = pd.read_csv(d / "test7_2.csv").iloc[:, 0].to_numpy()
    normal = fit_normal(normal_data)
    t_fit = fit_general_t(t_data)
    normal_risk = distribution_risk(normal["mu"], normal["sigma"])
    t_risk = distribution_risk(t_fit["mu"], t_fit["sigma"], t_fit["nu"])
    for number, values in (("8.1", normal_risk), ("8.2", t_risk)):
        outputs[number] = pd.DataFrame([values[:2]], columns=["VaR Absolute", "VaR Diff from Mean"])
    rng = np.random.default_rng(8)
    for number in ("8.3", "8.6"):
        sim = stats.t.ppf(rng.random(10000), t_fit["nu"]) * t_fit["sigma"] + t_fit["mu"]
        raw = empirical_var_es(sim)
        centered = empirical_var_es(sim - sim.mean())
        metric = 0 if number == "8.3" else 1
        title = "VaR" if metric == 0 else "ES"
        outputs[number] = pd.DataFrame([[raw[metric], centered[metric]]],
                                       columns=[f"{title} Absolute", f"{title} Diff from Mean"])
    for number, values in (("8.4", normal_risk), ("8.5", t_risk)):
        outputs[number] = pd.DataFrame([values[2:]], columns=["ES Absolute", "ES Diff from Mean"])

    returns9 = pd.read_csv(d / "test9_1_returns.csv")
    portfolio9 = pd.read_csv(d / "test9_1_portfolio.csv", encoding="utf-8-sig")
    portfolio9 = pd.DataFrame({"Stock": portfolio9["Stock"],
                               "currentValue": portfolio9["Holding"] * portfolio9["Starting Price"]})
    normal9 = fit_normal(returns9.A)
    t9 = fit_general_t(returns9.B)
    u9 = np.column_stack((stats.norm.cdf(returns9.A, normal9["mu"], normal9["sigma"]),
                          stats.t.cdf(returns9.B, t9["nu"], t9["mu"], t9["sigma"])))
    rho9 = stats.spearmanr(u9).statistic
    z9 = np.random.default_rng(9).multivariate_normal([0, 0], [[1, rho9], [rho9, 1]], 100000)
    sim_u9 = stats.norm.cdf(z9)
    sim9 = np.column_stack((stats.norm.ppf(sim_u9[:, 0], normal9["mu"], normal9["sigma"]),
                            stats.t.ppf(sim_u9[:, 1], t9["nu"], t9["mu"], t9["sigma"])))
    outputs["9.1"] = risk_table(sim9, portfolio9)

    cov52 = pd.read_csv(d / "test5_2.csv").to_numpy()
    cov53 = pd.read_csv(d / "test5_3.csv").to_numpy()
    for number, budget in (("10.1", [1] * 5), ("10.2", [1, 1, 1, 1, 0.5])):
        outputs[number] = pd.DataFrame({"W": risk_parity(cov52, budget)})
    means = pd.read_csv(d / "test10_3_means.csv").iloc[:, 0].to_numpy()
    for number, bounds in (("10.3", [(0, 1)] * 5), ("10.4", [(0.1, 0.5)] * 5)):
        outputs[number] = pd.DataFrame({"W": max_sharpe(cov53, means, 0.04, bounds)})

    returns11 = pd.read_csv(d / "test11_1_returns.csv")
    weights11 = pd.read_csv(d / "test11_1_weights.csv").iloc[:, 0]
    outputs["11.1"] = expost_attribution(weights11, returns11, returns11, np.eye(3)).drop(columns="Alpha")
    factors11 = pd.read_csv(d / "test11_2_factor_returns.csv")
    stocks11 = pd.read_csv(d / "test11_2_stock_returns.csv")
    weights11 = pd.read_csv(d / "test11_2_weights.csv").iloc[:, 0]
    betas11 = pd.read_csv(d / "test11_2_beta.csv").iloc[:, 1:]
    outputs["11.2"] = expost_attribution(weights11, stocks11, factors11, betas11)

    options = pd.read_csv(d / "test12_1.csv").dropna(subset=["ID"])
    european, american = [], []
    for row in options.itertuples(index=False):
        option_id, kind, spot, strike, days, year, rf, dividend, vol = row
        call = kind == "Call"
        t = days / year
        european.append((int(option_id), *gbsm(call, spot, strike, t, rf, rf - dividend, vol)))
        def price(parameters):
            s, k, maturity, rate, carry, sigma = parameters
            return float(american_tree(call, s, k, maturity, rate, carry, sigma, 500))
        parameters = np.array([spot, strike, t, rf, rf - dividend, vol], dtype=float)
        value = price(parameters)
        gradients = []
        for index in (0, 3, 5, 2):
            step = 1e-5 * max(abs(parameters[index]), 1)
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += step
            minus[index] -= step
            gradients.append((price(plus) - price(minus)) / (2 * step))
        gamma_step = 1.5
        plus, minus = parameters.copy(), parameters.copy()
        plus[0] += gamma_step
        minus[0] -= gamma_step
        gamma = (price(plus) + price(minus) - 2 * value) / (gamma_step * gamma_step)
        american.append((int(option_id), value, gradients[0], gamma, gradients[2], gradients[1], gradients[3]))
    option_columns = ["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]
    outputs["12.1"] = pd.DataFrame(european, columns=option_columns)
    outputs["12.2"] = pd.DataFrame(american, columns=option_columns)
    discrete = pd.read_csv(d / "test12_3.csv")
    discrete_results = []
    for row in discrete.itertuples(index=False):
        option_id, kind, spot, strike, days, year, rf, vol, dates, amounts = row
        discrete_results.append((option_id, american_discrete_dividends(
            kind == "Call", spot, strike, days / year, rf, vol, int(days * 2),
            [int(x) * 2 for x in dates.split(",")], [float(x) for x in amounts.split(",")])))
    outputs["12.3"] = pd.DataFrame(discrete_results, columns=["ID", "Value"])

    returns13 = pd.read_csv(d / "test13_returns.csv")
    x = returns13.to_numpy()
    correlation = kendall_correlation(x)
    outputs["13.1"] = matrix_frame(correlation)
    mu, scale, nu, ll = multivariate_t_fit(x, correlation)
    outputs["13.2"] = pd.DataFrame({"mu": mu})
    outputs["13.3"] = matrix_frame(scale)
    outputs["13.4"] = pd.DataFrame({"nu": [nu], "ll": [ll]})
    fitted = [fit_general_t(x[:, i]) for i in range(x.shape[1])]
    u = np.column_stack([stats.t.cdf(x[:, i], f["nu"], f["mu"], f["sigma"])
                         for i, f in enumerate(fitted)])
    ll_gaussian, nu_copula, ll_t = copula_fit(u, correlation)
    outputs["13.5"] = pd.DataFrame({"ll": [ll_gaussian]})
    outputs["13.6"] = pd.DataFrame({"nu": [nu_copula], "ll": [ll_t]})
    n = len(x)
    outputs["13.7"] = pd.DataFrame({
        "Copula": ["Gaussian", "T"], "LL": [ll_gaussian, ll_t], "K": [0, 1],
        "AICC": [-2 * ll_gaussian, -2 * ll_t + 2 + 4 / (n - 2)],
        "BIC": [-2 * ll_gaussian, np.log(n) - 2 * ll_t]})
    tails = []
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            rho = correlation[i, j]
            lam = 2 * stats.t.cdf(-np.sqrt((nu_copula + 1) * (1 - rho) / (1 + rho)),
                                  nu_copula + 1)
            tails.append((i + 1, j + 1, rho, lam))
    outputs["13.8"] = pd.DataFrame(tails, columns=["I", "J", "Rho", "Lambda"])
    sim_u = simulate_t_copula(correlation, nu_copula, 100000)
    sim_returns = np.column_stack([
        stats.t.ppf(sim_u[:, i], f["nu"]) * f["sigma"] + f["mu"]
        for i, f in enumerate(fitted)])
    outputs["13.9"] = risk_table(sim_returns, pd.read_csv(d / "test13_portfolio.csv"))
    return outputs


def expected_path(directory, number):
    return Path(directory) / f"testout{number.replace('.', '_')}.csv"


def validate_outputs(outputs, directory):
    failures = []
    for number, actual in outputs.items():
        expected = pd.read_csv(expected_path(directory, number))
        if list(actual.columns) != list(expected.columns) or actual.shape != expected.shape:
            failures.append(number)
            print(f"Test {number:>4}: FAIL - output shape or column names differ")
            continue
        text_columns = actual.select_dtypes(exclude=[np.number]).columns
        text_ok = all(actual[c].tolist() == expected[c].tolist() for c in text_columns)
        numeric = actual.select_dtypes(include=[np.number]).columns
        difference = np.abs(actual[numeric].to_numpy() - expected[numeric].to_numpy())
        # Simulation results differ by RNG; Greeks from a binomial tree have
        # finite difference noise. All deterministic fits use tighter limits.
        if number in {"8.3", "8.6", "9.1", "13.9"}:
            atol, rtol = 0.005, 0.08
        elif number == "12.2":
            atol, rtol = 0.005, 0.02
        elif number == "12.3":
            atol, rtol = 1e-3, 1e-3
        elif number in {"10.1", "10.2", "10.3", "10.4"}:
            atol, rtol = 1e-4, 1e-3
        elif number.startswith("13."):
            atol, rtol = 3e-4, 3e-3
        else:
            atol, rtol = 1e-5, 2e-4
        passed = text_ok and np.all(difference <= atol + rtol * np.abs(expected[numeric].to_numpy()))
        print(f"Test {number:>4}: {'PASS' if passed else 'FAIL'} - max absolute difference {difference.max():.6g}")
        if not passed:
            failures.append(number)
    return failures


def main():
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "testfiles" / "data"
    outputs = run_functional_tests(directory)
    failures = validate_outputs(outputs, directory)
    if failures:
        raise SystemExit("Failed tests: " + ", ".join(failures))
    print(f"All {len(outputs)} Functional Test 2 cases passed.")


if __name__ == "__main__":
    main()

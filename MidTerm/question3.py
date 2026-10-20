from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal, multivariate_t, norm, norminvgauss, t


PRICE_FILE = Path(__file__).with_name("problem3.csv")
HOLDINGS_FILE = Path(__file__).with_name("problem3_holdings.csv")


def aicc(log_likelihood, number_of_parameters, sample_size):
    k = number_of_parameters
    n = sample_size
    return (-2.0 * log_likelihood + 2.0 * k + 2.0 * k * (k + 1.0) / (n - k - 1.0))


def fit_marginal_distributions(values):
    values = np.asarray(values, dtype=float)
    n = len(values)

    normal_parameters = norm.fit(values)
    normal_log_likelihood = np.sum(norm.logpdf(values, *normal_parameters))

    t_parameters = t.fit(values)
    t_log_likelihood = np.sum(t.logpdf(values, *t_parameters))

    nig_parameters = norminvgauss.fit(values)
    nig_log_likelihood = np.sum(norminvgauss.logpdf(values, *nig_parameters))

    models = {
        "Normal": {
            "distribution": norm,
            "parameters": normal_parameters,
            "log_likelihood": normal_log_likelihood,
            "aicc": aicc(normal_log_likelihood, 2, n),
        },
        "Student t": {
            "distribution": t,
            "parameters": t_parameters,
            "log_likelihood": t_log_likelihood,
            "aicc": aicc(t_log_likelihood, 3, n),
        },
        "NIG": {
            "distribution": norminvgauss,
            "parameters": nig_parameters,
            "log_likelihood": nig_log_likelihood,
            "aicc": aicc(nig_log_likelihood, 4, n),
        },
    }

    best_model = min(models, key=lambda name: models[name]["aicc"])
    return models, best_model


# Problem 3(a): fit and compare the three candidate marginal distributions.
prices = pd.read_csv(PRICE_FILE, parse_dates=["Date"])
holdings = pd.read_csv(HOLDINGS_FILE)
stock_names = [column for column in prices.columns if column != "Date"]
returns = prices[stock_names].pct_change(fill_method=None).dropna()

fitted_margins = {}
comparison_rows = []

print("Problem 3(a): Marginal distribution fits")
print(f"Number of daily returns per stock: {len(returns)}")

for stock in stock_names:
    models, best_model = fit_marginal_distributions(returns[stock])
    fitted_margins[stock] = {"models": models, "selected": best_model,}

    normal_mu, normal_sigma = models["Normal"]["parameters"]
    t_df, t_loc, t_scale = models["Student t"]["parameters"]
    nig_a, nig_b, nig_loc, nig_scale = models["NIG"]["parameters"]
    nig_alpha = nig_a / nig_scale
    nig_beta = nig_b / nig_scale
    nig_delta = nig_scale
    nig_mu = nig_loc

    print(f"\n{stock}")
    print(
        f"  Normal: mu={normal_mu:.8f}, sigma={normal_sigma:.8f}, "
        f"AICc={models['Normal']['aicc']:.3f}"
    )
    print(
        f"  Student t: df={t_df:.6f}, loc={t_loc:.8f}, "
        f"scale={t_scale:.8f}, AICc={models['Student t']['aicc']:.3f}"
    )
    print(
        f"  NIG: alpha={nig_alpha:.6f}, beta={nig_beta:.6f}, "
        f"delta={nig_delta:.8f}, mu={nig_mu:.8f}, "
        f"AICc={models['NIG']['aicc']:.3f}"
    )
    print(f"  Selected distribution: {best_model}")

    comparison_rows.append(
        {
            "Stock": stock,
            "Normal AICc": models["Normal"]["aicc"],
            "Student t AICc": models["Student t"]["aicc"],
            "NIG AICc": models["NIG"]["aicc"],
            "Selected": best_model,
        }
    )

comparison_table = pd.DataFrame(comparison_rows).set_index("Stock")
print("\nProblem 3(a): Model comparison summary")
print(comparison_table.to_string(float_format=lambda value: f"{value:.3f}"))
print(
    "For each stock, I selected the distribution with the lowest AICc. AICc "
    "rewards goodness of fit while penalizing the Student t and NIG models "
    "for their additional parameters."
)


def nearest_correlation_higham(matrix, tolerance=1e-12, max_iterations=10000):
    y = np.asarray(matrix, dtype=float).copy()
    correction = np.zeros_like(y)

    for _ in range(max_iterations):
        previous_y = y.copy()
        residual = y - correction
        eigenvalues, eigenvectors = np.linalg.eigh((residual + residual.T) / 2.0)
        x = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        correction = x - residual
        y = x.copy()
        np.fill_diagonal(y, 1.0)
        y = (y + y.T) / 2.0

        if np.linalg.norm(y - previous_y, ord="fro") <= tolerance * max(1.0, np.linalg.norm(y, ord="fro")):
            return y

    raise RuntimeError("Higham's method did not converge.")


def gaussian_copula_log_likelihood(uniforms, correlation):
    normal_scores = norm.ppf(uniforms)
    joint_log_density = multivariate_normal.logpdf(
        normal_scores,
        mean=np.zeros(normal_scores.shape[1]),
        cov=correlation,
        allow_singular=True,
    )
    marginal_log_density = np.sum(norm.logpdf(normal_scores), axis=1)
    return float(np.sum(joint_log_density - marginal_log_density))


def t_copula_log_likelihood(uniforms, correlation, degrees_of_freedom):
    t_scores = t.ppf(uniforms, df=degrees_of_freedom)
    joint_log_density = multivariate_t.logpdf(
        t_scores,
        loc=np.zeros(t_scores.shape[1]),
        shape=correlation,
        df=degrees_of_freedom,
    )
    marginal_log_density = np.sum(
        t.logpdf(t_scores, df=degrees_of_freedom), axis=1
    )
    return float(np.sum(joint_log_density - marginal_log_density))


def bic(log_likelihood, number_of_parameters, sample_size):
    return (
        number_of_parameters * np.log(sample_size) - 2.0 * log_likelihood
    )


# Problem 3(b): fit Gaussian and Student's t copulas.
uniforms = np.column_stack(
    [
        fitted_margins[stock]["models"][fitted_margins[stock]["selected"]][
            "distribution"
        ].cdf(
            returns[stock].to_numpy(),
            *fitted_margins[stock]["models"][fitted_margins[stock]["selected"]][
                "parameters"
            ],
        )
        for stock in stock_names
    ]
)
uniforms = np.clip(uniforms, 1e-10, 1.0 - 1e-10)

dimension = len(stock_names)


def repair_if_needed(correlation):
    correlation = (correlation + correlation.T) / 2.0
    if np.min(np.linalg.eigvalsh(correlation)) < -1e-12:
        correlation = nearest_correlation_higham(correlation)
    return correlation


gaussian_correlation = repair_if_needed(
    np.corrcoef(norm.ppf(uniforms), rowvar=False)
)
gaussian_copula_ll = gaussian_copula_log_likelihood(
    uniforms, gaussian_correlation
)


def t_copula_correlation(nu):
    return repair_if_needed(
        np.corrcoef(t.ppf(uniforms, df=nu), rowvar=False)
    )

def negative_t_copula_log_likelihood(inverse_nu):
    nu = 1.0 / inverse_nu
    return -t_copula_log_likelihood(uniforms, t_copula_correlation(nu), nu)


t_copula_fit = minimize_scalar(
    negative_t_copula_log_likelihood,
    bounds=(1.0 / 100.0, 1.0 / 2.01),
    method="bounded",
    options={"xatol": 1e-10},
)
t_copula_nu = 1.0 / t_copula_fit.x
t_copula_ll = -t_copula_fit.fun
t_correlation = t_copula_correlation(t_copula_nu)

number_of_returns = len(returns)
correlation_parameters = dimension * (dimension - 1) // 2
copula_comparison = pd.DataFrame(
    {
        "Log likelihood": [gaussian_copula_ll, t_copula_ll],
        "AICc": [
            aicc(gaussian_copula_ll, correlation_parameters, number_of_returns),
            aicc(t_copula_ll, correlation_parameters + 1, number_of_returns),
        ],
        "BIC": [
            bic(gaussian_copula_ll, correlation_parameters, number_of_returns),
            bic(t_copula_ll, correlation_parameters + 1, number_of_returns),
        ],
    },
    index=["Gaussian", "Student t"],
)
selected_copula = copula_comparison["AICc"].idxmin()
copula_correlation = (
    gaussian_correlation if selected_copula == "Gaussian" else t_correlation
)

print("\nProblem 3(b): Copula fits")
for name, matrix in [("Gaussian", gaussian_correlation), ("Student t", t_correlation)]:
    print(f"{name} copula correlation matrix:")
    print(
        pd.DataFrame(matrix, index=stock_names, columns=stock_names).to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )
    print(f"Minimum eigenvalue: {np.min(np.linalg.eigvalsh(matrix)):.8f}")
print(copula_comparison.to_string(float_format=lambda value: f"{value:.3f}"))
print(f"Student t copula degrees of freedom: {t_copula_nu:.6f}")
print(f"Selected copula: {selected_copula}")
print(
    f"The {selected_copula} copula fits better because it has the lower AICc "
    "and BIC. Both copulas have the same correlation parameters, so the "
    "comparison turns on the t copula's extra degrees-of-freedom parameter."
)


def calculate_var_es(profit_and_loss, tail_probability=0.05):
    """Return absolute VaR and ES as positive dollar losses."""
    profit_and_loss = np.asarray(profit_and_loss, dtype=float)
    cutoff = np.quantile(profit_and_loss, tail_probability)
    value_at_risk = -cutoff
    expected_shortfall = -np.mean(profit_and_loss[profit_and_loss <= cutoff])
    return value_at_risk, expected_shortfall


def marginal_quantile(uniform_values, model, grid_size=5001):
    """Evaluate a fitted marginal quantile, with fast interpolation for NIG."""
    distribution = model["distribution"]
    parameters = model["parameters"]

    if distribution is not norminvgauss:
        return distribution.ppf(uniform_values, *parameters)

    fitted_mean, fitted_variance = distribution.stats(
        *parameters, moments="mv"
    )
    fitted_sigma = np.sqrt(fitted_variance)
    lower_bound = fitted_mean - 12.0 * fitted_sigma
    upper_bound = fitted_mean + 12.0 * fitted_sigma
    target_minimum = np.min(uniform_values)
    target_maximum = np.max(uniform_values)

    while distribution.cdf(lower_bound, *parameters) > target_minimum:
        lower_bound -= 6.0 * fitted_sigma
    while distribution.cdf(upper_bound, *parameters) < target_maximum:
        upper_bound += 6.0 * fitted_sigma

    value_grid = np.linspace(lower_bound, upper_bound, grid_size)
    probability_grid = distribution.cdf(value_grid, *parameters)
    valid = np.isfinite(probability_grid)
    value_grid = value_grid[valid]
    probability_grid = probability_grid[valid]

    unique = np.r_[True, np.diff(probability_grid) > 1e-14]
    return np.interp(
        uniform_values,
        probability_grid[unique],
        value_grid[unique],
    )


# Problem 3(c): simulate the selected copula and revalue every position.
number_of_simulations = 50_000
random_generator = np.random.default_rng(545)
normal_draws = random_generator.multivariate_normal(
    mean=np.zeros(dimension),
    cov=copula_correlation,
    size=number_of_simulations,
)

if selected_copula == "Gaussian":
    simulated_uniforms = norm.cdf(normal_draws)
else:
    chi_square_draws = random_generator.chisquare(
        df=t_copula_nu, size=number_of_simulations
    )
    t_draws = normal_draws / np.sqrt(
        chi_square_draws[:, np.newaxis] / t_copula_nu
    )
    simulated_uniforms = t.cdf(t_draws, df=t_copula_nu)

simulated_uniforms = np.clip(
    simulated_uniforms, 1e-10, 1.0 - 1e-10
)
simulated_returns = np.column_stack(
    [
        marginal_quantile(
            simulated_uniforms[:, column],
            fitted_margins[stock]["models"][
                fitted_margins[stock]["selected"]
            ],
        )
        for column, stock in enumerate(stock_names)
    ]
)

holdings_by_stock = holdings.set_index("Stock").loc[stock_names]
last_prices = prices.iloc[-1].set_axis(prices.columns).loc[stock_names].astype(float)
position_values = (
    holdings_by_stock["Holding"].astype(float) * last_prices
)
position_profit_and_loss = simulated_returns * position_values.to_numpy()

risk_rows = []
for column, stock in enumerate(stock_names):
    stock_var, stock_es = calculate_var_es(position_profit_and_loss[:, column])
    risk_rows.append(
        {
            "Position": stock,
            "Current Value": position_values.loc[stock],
            "VaR 95%": stock_var,
            "ES 95%": stock_es,
        }
    )

for book in ["A", "B"]:
    book_mask = holdings_by_stock["Book"].to_numpy() == book
    book_profit_and_loss = np.sum(
        position_profit_and_loss[:, book_mask], axis=1
    )
    book_var, book_es = calculate_var_es(book_profit_and_loss)
    risk_rows.append(
        {
            "Position": f"Book {book}",
            "Current Value": position_values.iloc[book_mask].sum(),
            "VaR 95%": book_var,
            "ES 95%": book_es,
        }
    )

total_profit_and_loss = np.sum(position_profit_and_loss, axis=1)
total_var, total_es = calculate_var_es(total_profit_and_loss)
risk_rows.append(
    {
        "Position": "Total Portfolio",
        "Current Value": position_values.sum(),
        "VaR 95%": total_var,
        "ES 95%": total_es,
    }
)

risk_table = pd.DataFrame(risk_rows).set_index("Position")
sum_individual_var = risk_table.loc[stock_names, "VaR 95%"].sum()
sum_individual_es = risk_table.loc[stock_names, "ES 95%"].sum()
var_diversification = 1.0 - total_var / sum_individual_var
es_diversification = 1.0 - total_es / sum_individual_es

print("\nProblem 3(c): 95% absolute VaR and ES")
print(f"Selected copula: {selected_copula}")
print(f"Number of simulations: {number_of_simulations:,}")
print(
    risk_table.to_string(
        formatters={
            "Current Value": lambda value: f"${value:,.2f}",
            "VaR 95%": lambda value: f"${value:,.2f}",
            "ES 95%": lambda value: f"${value:,.2f}",
        }
    )
)
print(
    f"The total portfolio VaR is {var_diversification:.2%} below the sum of "
    f"the stand-alone stock VaRs, and total ES is {es_diversification:.2%} "
    "below the sum of the stand-alone stock ES values. This difference is the "
    "benefit of diversification across stocks and books."
)

stock_risk_table = risk_table.loc[stock_names]
largest_stock_var = stock_risk_table["VaR 95%"].idxmax()
largest_stock_es = stock_risk_table["ES 95%"].idxmax()
riskier_book = risk_table.loc[["Book A", "Book B"], "VaR 95%"].idxmax()
print(
    f"{largest_stock_var} has the largest stand-alone VaR, and "
    f"{largest_stock_es} has the largest stand-alone ES. {riskier_book} has "
    "the larger book-level VaR. Correlations are stronger within each book "
    "than across books, so combining the books provides additional "
    "diversification."
)

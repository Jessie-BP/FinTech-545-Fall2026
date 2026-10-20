from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm


DATA_FILE = Path(__file__).with_name("problem2.csv")

# Problem 2(a): 95% and 99% VaR for XYZ, reported as percentages.
prices = pd.read_csv(DATA_FILE, parse_dates=["Date"])
returns = prices[["SPY", "XYZ"]].pct_change(fill_method=None).dropna()

spy_returns = returns["SPY"].to_numpy()
xyz_returns = returns["XYZ"].to_numpy()

design_matrix = np.column_stack([np.ones(len(spy_returns)), spy_returns])
alpha, beta = np.linalg.lstsq(design_matrix, xyz_returns, rcond=None)[0]
residuals = xyz_returns - (alpha + beta * spy_returns)

spy_mean = np.mean(spy_returns)
spy_sigma = np.std(spy_returns, ddof=1)
residual_sigma = np.std(residuals, ddof=1)

xyz_mean = alpha + beta * spy_mean
xyz_sigma = np.sqrt(beta**2 * spy_sigma**2 + residual_sigma**2)

confidence_levels = [0.95, 0.99]
var_results = {}
for confidence_level in confidence_levels:
    lower_return_quantile = norm.ppf(1.0 - confidence_level, loc=xyz_mean, scale=xyz_sigma)
    var_results[confidence_level] = -lower_return_quantile

print("Problem 2(a): Normal market-model VaR")
print(f"Number of daily returns: {len(returns)}")
print(f"alpha: {alpha:.8f}")
print(f"beta: {beta:.8f}")
print(f"SPY mean: {spy_mean:.8f}")
print(f"SPY sample sigma: {spy_sigma:.8f}")
print(f"Residual sample sigma: {residual_sigma:.8f}")
print(f"Implied XYZ mean: {xyz_mean:.8f}")
print(f"Implied XYZ sigma: {xyz_sigma:.8f}")
for confidence_level, value_at_risk in var_results.items():
    print(f"{confidence_level:.0%} VaR: {value_at_risk:.4%}")

print(
    "I estimated the market model using OLS. Because the SPY return and the residual returns are assumed "
    "to be independent normal variables, the implied "
    "XYZ return is also normal. VaR is reported as the positive loss at the "
    "lower-tail return quantile."
)


# Problem 2(b): 95% and 99% expected shortfall, reported as percentages.
es_results = {}
for confidence_level in confidence_levels:
    tail_probability = 1.0 - confidence_level
    lower_standard_normal_quantile = norm.ppf(tail_probability)
    expected_shortfall = (-xyz_mean + xyz_sigma * norm.pdf(lower_standard_normal_quantile) / tail_probability)
    es_results[confidence_level] = expected_shortfall

print("\nProblem 2(b): Normal market-model ES")
for confidence_level, expected_shortfall in es_results.items():
    print(f"{confidence_level:.0%} ES: {expected_shortfall:.4%}")

print(
    "Expected shortfall is the positive average loss conditional on the XYZ "
    "return being below the corresponding VaR cutoff."
)

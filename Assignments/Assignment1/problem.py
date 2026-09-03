
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import stats
from scipy.optimize import minimize
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

## problem 1
data = pd.read_csv('problem1.csv')
x = data['x']

print(data.head())
print("Number of observations:", len(data))

mean_x = x.mean()
variance_x = x.var()
skewness_x = x.skew()
excess_kurt_x = x.kurt()

print(f"Mean: {mean_x:.8f}")
print(f"Variance: {variance_x:.8f}")
print(f"Skewness: {skewness_x:.4f}")
print(f"Excess kurtosis: {excess_kurt_x:.4f}")

plt.figure(figsize=(8,5))
plt.hist(x,bins=40,density=True, alpha=0.7, edgecolor="black")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("problem 1: distribution")
plt.tight_layout()
plt.savefig("problem1_hist.png", dpi=300)
plt.close()

mu_hat = mean_x
sigma_hat = variance_x ** 0.5

normal_q01 = norm.ppf(0.01, loc=mu_hat, scale=sigma_hat)

actual_below = (x < normal_q01).sum()
expected_below = len(x) * 0.01

print(f"Fitted mean: {mu_hat:.8f}")
print(f"Fitted standard deviation: {sigma_hat:.8f}")
print(f"Normal 1% quantile: {normal_q01:.8f}")
print(f"Actual observations below quantile: {actual_below}")
print(f"Expected observations below quantile: {expected_below:.1f}")


## problem 2
problem2 = pd.read_csv("problem2.csv")

x2 = problem2["x"]
y2 = problem2["y"]

print("\nProblem 2")
print(problem2.head())
print(problem2.describe())

plt.figure(figsize=(8,5))
plt.scatter(x2, y2, alpha=0.7, edgecolor="black")
plt.xlabel("x")
plt.ylabel("y")
plt.title("problem2: scatter plot of y against x")
plt.tight_layout()
plt.savefig("problem2_scatter.png", dpi=300)
plt.close()


ols = stats.linregress(x2, y2)

alpha_ols = ols.intercept
beta_ols = ols.slope
slope_se_ols = ols.stderr
intercept_se_ols = ols.intercept_stderr

print(f"Alpha standard error: {intercept_se_ols:.6f}")
print(f"Beta standard error: {slope_se_ols:.6f}")

residuals_ols = y2 - (alpha_ols + beta_ols * x2)
sigma_ols = np.sqrt(np.sum(residuals_ols**2)/(len(y2)-2))

print("\nOLS")
print(f"Alpha: {alpha_ols:.6f}")
print(f"Beta: {beta_ols:.6f}")
print(f"Residual standard error: {sigma_ols:.6f}")
print(f"Beta standard error: {slope_se_ols:.6f}")

def normal_negative_log(parameters, x, y):
    alpha, beta, log_sigma = parameters
    sigma = np.exp(log_sigma)
    
    residuals = y - (alpha + beta * x)
    
    return -np.sum(stats.norm.logpdf(residuals, loc=0, scale=sigma))

normal_start=(alpha_ols, beta_ols, np.log(sigma_ols))
normal_result = minimize(normal_negative_log, normal_start, args=(x2,y2), method="Nelder-Mead")

alpha_normal, beta_normal, log_sigma_normal = normal_result.x
sigma_normal = np.exp(log_sigma_normal)
normal_nll = normal_result.fun

print("\nNormal MLE")
print(f"Alpha: {alpha_normal:.6f}")
print(f"Beta: {beta_normal:.6f}")
print(f"Error sigma: {sigma_normal:.6f}")
print(f"Optimization successful: {normal_result.success}")


def t_negative_log(parameters, x, y):
    alpha, beta, log_scale, log_nu_minus_2 = parameters
    
    scale = np.exp(log_scale)
    nu = 2 + np.exp(log_nu_minus_2)

    residuals = y - (alpha + beta * x)
    
    return -np.sum(stats.t.logpdf(residuals, df=nu, loc=0, scale=scale))


t_start = [alpha_ols, beta_ols, np.log(sigma_ols), np.log(5)]

t_result = minimize(t_negative_log, t_start, args=(x2, y2), method="BFGS")

alpha_t, beta_t, log_scale_t, log_nu_minus_2 = t_result.x
scale_t = np.exp(log_scale_t)
nu_t = 2 + np.exp(log_nu_minus_2)
t_nll = t_result.fun

print("\nStudent's t MLE")
print(f"Alpha: {alpha_t:.6f}")
print(f"Beta: {beta_t:.6f}")
print(f"Error scale: {scale_t:.6f}")
print(f"Degrees of freedom: {nu_t:.6f}")
print(f"Optimization successful: {t_result.success}")


def calculate_aicc(nll, number_of_parameters, sample_size):
      aic = 2 * number_of_parameters + 2 * nll

      correction = (2 * number_of_parameters * (number_of_parameters + 1) / (sample_size - number_of_parameters - 1))

      return aic + correction


n = len(y2)

normal_aicc = calculate_aicc(normal_nll, 3, n)

t_aicc = calculate_aicc(t_nll, 4, n)

print("\nAICc comparison")
print(f"Normal AICc: {normal_aicc:.4f}")
print(f"Student's t AICc: {t_aicc:.4f}")

normal_q95 = stats.norm.ppf(0.95, loc=0, scale=sigma_normal)
normal_q995 = stats.norm.ppf(0.995, loc=0, scale=sigma_normal)

t_q95 = stats.t.ppf(0.95, df=nu_t, loc=0, scale=scale_t)
t_q995 = stats.t.ppf(0.995, df=nu_t, loc=0, scale=scale_t)

print("\nFitted error quantiles")
print(f"Normal 95%: {normal_q95:.6f}")
print(f"Student's t 95%: {t_q95:.6f}")
print(f"Normal 99.5%: {normal_q995:.6f}")
print(f"Student's t 99.5%: {t_q995:.6f}")


## problem 3

problem3 = pd.read_csv("problem3.csv")

print("\nProblem 3")
print(problem3.head())

pd.plotting.scatter_matrix(problem3, figsize=(10, 10), alpha=0.6, diagonal="hist")

plt.suptitle("Problem 3: Pairwise Relationships", y=1.02)
plt.tight_layout()
plt.savefig("problem3_pairs.png", dpi=300)
plt.close()

pearson_corr = problem3.corr(method="pearson")
spearman_corr = problem3.corr(method="spearman")

print("\nPearson correlation")
print(pearson_corr)

print("\nSpearman correlation")
print(spearman_corr)

correlation_gap = (pearson_corr - spearman_corr).abs()

upper_triangle = np.triu(np.ones(correlation_gap.shape, dtype=bool), k=1)

largest_pair = correlation_gap.where(upper_triangle).stack().idxmax()
largest_gap = correlation_gap.loc[largest_pair]

print(f"\nLargest-gap pair: {largest_pair}")
print(f"Largest gap: {largest_gap:.6f}")


## problem 4

problem4 = pd.read_csv("problem4.csv")

x1_p4 = problem4["x1"]
x2_p4 = problem4["x2"]

cov_matrix = problem4[["x1", "x2"]].cov()
print("\nProblem 4 covariance matrix")
print(cov_matrix)

sigma_11 = cov_matrix.loc["x1", "x1"]
sigma_12 = cov_matrix.loc["x1", "x2"]
sigma_21 = cov_matrix.loc["x2", "x1"]
sigma_22 = cov_matrix.loc["x2", "x2"]

conditional_variance = sigma_22 - sigma_21 * sigma_12 / sigma_11
variance_factor = conditional_variance / sigma_22

print(f"Conditional variance: {conditional_variance:.6f}")
print(f"Variance factor: {variance_factor:.6f}")

mean_x1 = x1_p4.mean()
mean_x2 = x2_p4.mean()

conditional_slope = sigma_21 / sigma_11

conditional_mean = (mean_x2 + conditional_slope * (x1_p4 - mean_x1))

print(f"Conditional mean slope: {conditional_slope:.6f}")


conditional_sd = np.sqrt(conditional_variance)
critical_value = stats.norm.ppf(0.975)

lower_band = conditional_mean - critical_value * conditional_sd
upper_band = conditional_mean + critical_value * conditional_sd

plot_order = np.argsort(x1_p4)
sorted_x1 = x1_p4.iloc[plot_order]
sorted_mean = conditional_mean.iloc[plot_order]
sorted_lower = lower_band.iloc[plot_order]
sorted_upper = upper_band.iloc[plot_order]

plt.figure(figsize=(8, 5))
plt.scatter(x1_p4, x2_p4, alpha=0.5, label="Observed data")
plt.plot(sorted_x1, sorted_mean, color="red", label="Conditional mean")
plt.fill_between(sorted_x1, sorted_lower, sorted_upper, color="red", alpha=0.2, label="95% conditional band")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Problem 4: Conditional Distribution of x2 Given x1")
plt.legend()
plt.tight_layout()
plt.savefig("problem4_conditional.png", dpi=300)
plt.close()


inside_band = ((x2_p4 >= lower_band) & (x2_p4 <= upper_band))

print(f"Overall coverage: {inside_band.mean():.4f}")

distance = np.abs((x1_p4 - mean_x1) / x1_p4.std())

bucket = pd.cut(distance, bins=[-np.inf, 1, 2, np.inf], labels=["Within 1 SD", "Between 1 and 2 SD", "Beyond 2 SD"])

coverage_by_bucket = (pd.DataFrame({"bucket": bucket, "inside_band": inside_band})
      .groupby("bucket", observed=True)["inside_band"]
      .agg(["count", "mean"]))

print("\nCoverage by x1-distance bucket")
print(coverage_by_bucket)



## problem 5

problem5 = pd.read_csv("problem5.csv")
x5 = problem5["x"]

n5 = len(x5)
significance_band = 1.96 / np.sqrt(n5)

print("\nProblem 5")
print(f"Number of observations: {n5}")
print(f"Approximate 95% significance band: ±{significance_band:.4f}")

fig, axes = plt.subplots(3, 1, figsize=(9, 10))

axes[0].plot(x5)
axes[0].set_title("Problem 5: Time Series")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("x")

plot_acf(x5, lags=20, alpha=None, ax=axes[1])
plot_pacf(x5, lags=20, alpha=None, method="ywm", ax=axes[2])

for ax in axes[1:]:
    ax.axhline(significance_band, color="red", linestyle="--")
    ax.axhline(-significance_band, color="red", linestyle="--")

plt.tight_layout()
plt.savefig("problem5_acf_pacf.png", dpi=300)
plt.close()


model_results = {}

for model_name, order in {
      "AR(1)": (1, 0, 0),
      "AR(2)": (2, 0, 0),
      "AR(3)": (3, 0, 0),
      "MA(1)": (0, 0, 1),
      "MA(2)": (0, 0, 2),
      "MA(3)": (0, 0, 3),
}.items():

    fitted_model = ARIMA(x5, order=order, trend="c").fit()

    k = len(fitted_model.params)

    aicc = (fitted_model.aic+ 2 * k * (k + 1) / (n5 - k - 1))

    model_results[model_name] = {"model": fitted_model, "aicc": aicc}

    print(f"\n{model_name}")
    print(fitted_model.params)
    print(f"AICc: {aicc:.4f}")


best_model = min(model_results, key=lambda name: model_results[name]["aicc"])
print(f"\nSelected model: {best_model}")
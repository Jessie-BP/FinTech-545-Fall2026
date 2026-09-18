from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats


# Tests 1.1-1.4:

def missing_cov(x, skip_missing=True, correlation=False):

    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional array")
    calculate = np.corrcoef if correlation else np.cov
    if not np.isnan(x).any():
        return calculate(x, rowvar=False)
    if skip_missing:
        complete = x[~np.isnan(x).any(axis=1)]
        if complete.shape[0] < 2:
            raise ValueError("fewer than two complete rows")
        return calculate(complete, rowvar=False)
    number_of_columns = x.shape[1]
    output = np.empty((number_of_columns, number_of_columns))
    for i in range(number_of_columns):
        for j in range(i + 1):
            keep = ~np.isnan(x[:, i]) & ~np.isnan(x[:, j])
            pair = x[keep][:, [i, j]]
            if pair.shape[0] < 2:
                raise ValueError(f"fewer than two observations for columns {i}, {j}")
            value = calculate(pair, rowvar=False)[0, 1]
            output[i, j] = value
            output[j, i] = value
    return output


# Tests 2.1-2.3:

def exponential_weights(number_of_observations, decay):
    if number_of_observations < 1:
        raise ValueError("number_of_observations must be positive")
    if not 0 < decay < 1:
        raise ValueError("decay must be between zero and one")
    powers = np.arange(number_of_observations - 1, -1, -1)
    weights = (1 - decay) * decay**powers
    return weights / weights.sum()


def ew_covariance(x, decay):
    x = np.asarray(x, dtype=float)
    weights = exponential_weights(x.shape[0], decay)
    centered = x - weights @ x
    weighted = np.sqrt(weights)[:, None] * centered
    return weighted.T @ weighted


def covariance_to_correlation(covariance):
    covariance = np.asarray(covariance, dtype=float)
    inverse_sd = 1 / np.sqrt(np.diag(covariance))
    return inverse_sd[:, None] * covariance * inverse_sd[None, :]


# Tests 3.1-3.4 and 4.1:

def near_psd(matrix, epsilon=0.0):
    matrix = np.asarray(matrix, dtype=float)
    output = matrix.copy()
    is_correlation = np.allclose(np.diag(output), 1.0)
    standard_deviations = None
    if not is_correlation:
        standard_deviations = np.sqrt(np.diag(output))
        output = output / standard_deviations[:, None] / standard_deviations[None, :]

    eigenvalues, eigenvectors = np.linalg.eigh(output)
    eigenvalues = np.maximum(eigenvalues, epsilon)
    scale = 1 / np.sum(eigenvectors**2 * eigenvalues[None, :], axis=1)
    B = np.diag(np.sqrt(scale)) @ eigenvectors @ np.diag(np.sqrt(eigenvalues))
    output = B @ B.T

    if standard_deviations is not None:
        output = standard_deviations[:, None] * output * standard_deviations[None, :]
    return (output + output.T) / 2


def _positive_semidefinite_projection(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def higham_nearest_psd(matrix, epsilon=1e-9, max_iterations=100, tolerance=1e-9):
    matrix = np.asarray(matrix, dtype=float)
    output = matrix.copy()
    is_correlation = np.allclose(np.diag(output), 1.0)
    standard_deviations = None
    if not is_correlation:
        standard_deviations = np.sqrt(np.diag(output))
        output = output / standard_deviations[:, None] / standard_deviations[None, :]

    original = output.copy()
    correction = np.zeros_like(output)
    previous_norm = np.inf
    for _ in range(max_iterations):
        residual = output - correction
        psd_projection = _positive_semidefinite_projection(residual)
        correction = psd_projection - residual
        output = psd_projection.copy()
        np.fill_diagonal(output, 1.0)
        current_norm = np.sum((output - original) ** 2)
        minimum_eigenvalue = np.linalg.eigvalsh(output).min()
        if abs(current_norm - previous_norm) < tolerance and minimum_eigenvalue > -epsilon:
            break
        previous_norm = current_norm

    if standard_deviations is not None:
        output = standard_deviations[:, None] * output * standard_deviations[None, :]
    return (output + output.T) / 2


def chol_psd(matrix, epsilon=-1e-8):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    size = matrix.shape[0]
    root = np.zeros_like(matrix)
    for j in range(size):
        diagonal_value = matrix[j, j] - root[j, :j] @ root[j, :j]
        # A mathematically zero pivot can arrive as a few machine epsilons
        # after the preceding eigendecomposition.
        if epsilon <= diagonal_value <= 1e-14:
            diagonal_value = 0.0
        if diagonal_value < 0:
            raise np.linalg.LinAlgError("matrix is not positive semidefinite")
        root[j, j] = np.sqrt(diagonal_value)
        if root[j, j] == 0:
            continue
        for i in range(j + 1, size):
            root[i, j] = (matrix[i, j] - root[i, :j] @ root[j, :j]) / root[j, j]
    return root


# Tests 5.1-5.5:

def simulate_normal(number_of_simulations, covariance, mean=None, seed=1234, fix_method=near_psd):
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    dimension = covariance.shape[0]
    location = np.zeros(dimension) if mean is None else np.asarray(mean, dtype=float)
    if location.shape != (dimension,):
        raise ValueError("mean must have one value per covariance column")
    try:
        root = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        try:
            root = chol_psd(covariance)
        except np.linalg.LinAlgError:
            root = chol_psd(fix_method(covariance))
    rng = np.random.default_rng(seed)
    standard_normals = rng.standard_normal((number_of_simulations, dimension))
    return standard_normals @ root.T + location


def simulate_pca(covariance, number_of_simulations, percent_explained=1.0, mean=None, seed=1234):
    covariance = np.asarray(covariance, dtype=float)
    dimension = covariance.shape[0]
    location = np.zeros(dimension) if mean is None else np.asarray(mean, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues >= 1e-8
    positive_values = eigenvalues[positive]
    positive_vectors = eigenvectors[:, positive]
    if percent_explained < 1:
        cumulative = np.cumsum(positive_values) / eigenvalues.sum()
        number_to_keep = np.searchsorted(cumulative, percent_explained) + 1
        positive_values = positive_values[:number_to_keep]
        positive_vectors = positive_vectors[:, :number_to_keep]
    loading = positive_vectors @ np.diag(np.sqrt(positive_values))
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((number_of_simulations, len(positive_values)))
    return factors @ loading.T + location


# Tests 6.1-6.2:

def calculate_returns(prices, method="DISCRETE", date_column="Date"):
    if date_column not in prices.columns:
        raise ValueError(f"date column {date_column!r} was not found")
    asset_columns = [column for column in prices.columns if column != date_column]
    price_values = prices[asset_columns].to_numpy(dtype=float)
    ratios = price_values[1:] / price_values[:-1]
    method = method.upper()
    if method == "DISCRETE":
        returns = ratios - 1
    elif method == "LOG":
        returns = np.log(ratios)
    else:
        raise ValueError("method must be 'DISCRETE' or 'LOG'")
    output = pd.DataFrame(returns, columns=asset_columns)
    output.insert(0, date_column, prices[date_column].iloc[1:].to_numpy())
    return output


# Tests 7.1-7.6:

def fit_normal(x):
    x = np.asarray(x, dtype=float)
    return {"mu": float(np.mean(x)), "sigma": float(np.std(x, ddof=1))}


def _general_t_negative_log_likelihood(theta, x):
    mu, log_sigma, log_nu_minus_two = theta
    sigma = np.exp(log_sigma)
    nu = 2 + np.exp(log_nu_minus_two)
    return -np.sum(stats.t.logpdf(x, df=nu, loc=mu, scale=sigma))


def fit_general_t(x):
    x = np.asarray(x, dtype=float)
    excess_kurtosis = stats.kurtosis(x, fisher=True, bias=True)
    start_nu = max(2.1, 6 / excess_kurtosis + 4) if excess_kurtosis > 0 else 30.0
    start_sigma = np.sqrt(np.var(x, ddof=1) * (start_nu - 2) / start_nu)
    start = [np.mean(x), np.log(start_sigma), np.log(start_nu - 2)]
    result = optimize.minimize(
        _general_t_negative_log_likelihood, start, args=(x,), method="BFGS",
        options={"gtol": 1e-10, "maxiter": 10_000},
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"t-distribution optimization failed: {result.message}")
    mu, log_sigma, log_nu_minus_two = result.x
    return {
        "mu": float(mu),
        "sigma": float(np.exp(log_sigma)),
        "nu": float(2 + np.exp(log_nu_minus_two)),
        "log_likelihood": float(-result.fun),
    }


def _t_regression_negative_log_likelihood(theta, design, y):
    beta = theta[: design.shape[1]]
    sigma = np.exp(theta[-2])
    nu = 2 + np.exp(theta[-1])
    residuals = y - design @ beta
    return -np.sum(stats.t.logpdf(residuals, df=nu, loc=0, scale=sigma))


def fit_t_regression(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    design = np.column_stack((np.ones(len(y)), x))
    beta_start = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ beta_start
    excess_kurtosis = stats.kurtosis(residuals, fisher=True, bias=True)
    start_nu = max(2.1, 6 / excess_kurtosis + 4) if excess_kurtosis > 0 else 30.0
    start_sigma = np.sqrt(np.var(residuals, ddof=1) * (start_nu - 2) / start_nu)
    start = np.r_[beta_start, np.log(start_sigma), np.log(start_nu - 2)]
    result = optimize.minimize(
        _t_regression_negative_log_likelihood, start, args=(design, y), method="BFGS",
        options={"gtol": 1e-10, "maxiter": 10_000},
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"t-regression optimization failed: {result.message}")
    beta = result.x[: design.shape[1]]
    return {
        "mu": 0.0,
        "sigma": float(np.exp(result.x[-2])),
        "nu": float(2 + np.exp(result.x[-1])),
        "beta": beta,
        "log_likelihood": float(-result.fun),
    }


def aicc(log_likelihood, number_of_parameters, sample_size):
    aic = -2 * log_likelihood + 2 * number_of_parameters
    correction = (2 * number_of_parameters * (number_of_parameters + 1)
                  / (sample_size - number_of_parameters - 1))
    return aic + correction


def fit_nig_moments(x):
    """Estimate NIG (mu, alpha, beta, delta) by matching four moments."""
    x = np.asarray(x, dtype=float)
    sample_mean = np.mean(x)
    sample_variance = np.var(x, ddof=1)
    sample_skewness = stats.skew(x, bias=True)
    sample_excess_kurtosis = stats.kurtosis(x, fisher=True, bias=True)
    if sample_variance <= 0 or sample_excess_kurtosis <= 0:
        raise ValueError("NIG moment fitting requires positive variance and kurtosis")
    ratio = sample_skewness**2 / sample_excess_kurtosis
    if ratio >= 3 / 5:
        raise ValueError("sample is outside the NIG moment region")
    rho_squared = ratio / (3 - 4 * ratio)
    rho = np.sign(sample_skewness) * np.sqrt(rho_squared)
    delta_times_gamma = 3 * (1 + 4 * rho_squared) / sample_excess_kurtosis
    alpha = np.sqrt(delta_times_gamma / (sample_variance * (1 - rho_squared) ** 2))
    beta = rho * alpha
    gamma = np.sqrt(alpha**2 - beta**2)
    delta = delta_times_gamma / gamma
    mu = sample_mean - delta * beta / gamma
    return {"mu": mu, "alpha": alpha, "beta": beta, "delta": delta}


def fit_nig_mle(x):
    x = np.asarray(x, dtype=float)
    # SciPy: a=alpha*delta, b=beta*delta, loc=mu, scale=delta.
    a, b, loc, scale = stats.norminvgauss.fit(x)
    return {"mu": float(loc), "alpha": float(a / scale),
            "beta": float(b / scale), "delta": float(scale)}


def _matrix_frame(values, columns):
    return pd.DataFrame(values, columns=columns)


def run_functional_tests(data_directory):
    """Run Tests 1.1-7.6 and return their outputs as DataFrames."""
    data_directory = Path(data_directory)
    outputs = {}

    test1 = pd.read_csv(data_directory / "test1.csv")
    outputs["1.1"] = _matrix_frame(missing_cov(test1, True), test1.columns)
    outputs["1.2"] = _matrix_frame(missing_cov(test1, True, True), test1.columns)
    outputs["1.3"] = _matrix_frame(missing_cov(test1, False), test1.columns)
    outputs["1.4"] = _matrix_frame(missing_cov(test1, False, True), test1.columns)

    test2 = pd.read_csv(data_directory / "test2.csv")
    ew97 = ew_covariance(test2, 0.97)
    ew94 = ew_covariance(test2, 0.94)
    correlation94 = covariance_to_correlation(ew94)
    sd97 = np.sqrt(np.diag(ew97))
    mixed_covariance = sd97[:, None] * correlation94 * sd97[None, :]
    outputs["2.1"] = _matrix_frame(ew97, test2.columns)
    outputs["2.2"] = _matrix_frame(correlation94, test2.columns)
    outputs["2.3"] = _matrix_frame(mixed_covariance, test2.columns)

    covariance13 = outputs["1.3"].to_numpy()
    correlation14 = outputs["1.4"].to_numpy()
    outputs["3.1"] = _matrix_frame(near_psd(covariance13), test1.columns)
    outputs["3.2"] = _matrix_frame(near_psd(correlation14), test1.columns)
    outputs["3.3"] = _matrix_frame(higham_nearest_psd(covariance13), test1.columns)
    outputs["3.4"] = _matrix_frame(higham_nearest_psd(correlation14), test1.columns)
    outputs["4.1"] = _matrix_frame(chol_psd(outputs["3.1"]), test1.columns)

    covariance51 = pd.read_csv(data_directory / "test5_1.csv")
    covariance52 = pd.read_csv(data_directory / "test5_2.csv")
    covariance53 = pd.read_csv(data_directory / "test5_3.csv")
    simulations = {
        "5.1": simulate_normal(100_000, covariance51),
        "5.2": simulate_normal(100_000, covariance52),
        "5.3": simulate_normal(100_000, covariance53, fix_method=near_psd),
        "5.4": simulate_normal(100_000, covariance53, fix_method=higham_nearest_psd),
        "5.5": simulate_pca(covariance52, 100_000, percent_explained=0.99),
    }
    for test_number, simulation in simulations.items():
        outputs[test_number] = _matrix_frame(
            np.cov(simulation, rowvar=False), covariance51.columns
        )

    prices = pd.read_csv(data_directory / "test6.csv")
    outputs["6.1"] = calculate_returns(prices, "DISCRETE")
    outputs["6.2"] = calculate_returns(prices, "LOG")

    test71 = pd.read_csv(data_directory / "test7_1.csv").iloc[:, 0].to_numpy()
    test72 = pd.read_csv(data_directory / "test7_2.csv").iloc[:, 0].to_numpy()
    test73 = pd.read_csv(data_directory / "test7_3.csv")
    test75 = pd.read_csv(data_directory / "test7_5.csv").iloc[:, 0].to_numpy()
    normal_fit = fit_normal(test71)
    t_fit = fit_general_t(test72)
    regression_fit = fit_t_regression(test73["y"], test73.drop(columns="y"))
    outputs["7.1"] = pd.DataFrame([normal_fit])
    outputs["7.2"] = pd.DataFrame([{key: t_fit[key] for key in ("mu", "sigma", "nu")}])
    outputs["7.3"] = pd.DataFrame([{
        "mu": regression_fit["mu"], "sigma": regression_fit["sigma"],
        "nu": regression_fit["nu"], "Alpha": regression_fit["beta"][0],
        "B1": regression_fit["beta"][1], "B2": regression_fit["beta"][2],
        "B3": regression_fit["beta"][3],
    }])
    outputs["7.4"] = pd.DataFrame(
        {"AICC": [aicc(t_fit["log_likelihood"], 3, len(test72))]}
    )
    outputs["7.5"] = pd.DataFrame([fit_nig_moments(test75)])
    outputs["7.6"] = pd.DataFrame([fit_nig_mle(test75)])
    return outputs


def _expected_output_path(data_directory, test_number):
    """Handle the historical underscore inconsistency in expected filenames."""
    candidates = (
        data_directory / f"testout{test_number.replace('.', '_')}.csv",
        data_directory / f"testout_{test_number}.csv",
        data_directory / f"testout{test_number}.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"expected output for Test {test_number} was not found")


def validate_outputs(outputs, data_directory):
    failures = []
    for test_number, actual in outputs.items():
        expected = pd.read_csv(_expected_output_path(Path(data_directory), test_number))
        if test_number.startswith("5."):
            # Julia and NumPy use different random streams; compare covariances.
            relative_error = (np.linalg.norm(actual.to_numpy() - expected.to_numpy(), ord="fro")
                              / np.linalg.norm(expected.to_numpy(), ord="fro"))
            passed = relative_error < 0.02
            detail = f"relative covariance error={relative_error:.3%}"
        else:
            try:
                np.testing.assert_allclose(
                    actual.select_dtypes(include=[np.number]).to_numpy(),
                    expected.select_dtypes(include=[np.number]).to_numpy(),
                    rtol=2e-4, atol=1e-8,
                )
                passed = True
                detail = "matches expected output"
            except AssertionError:
                passed = False
                difference = np.max(np.abs(
                    actual.select_dtypes(include=[np.number]).to_numpy()
                    - expected.select_dtypes(include=[np.number]).to_numpy()
                ))
                detail = f"maximum absolute difference={difference:.6g}"
        print(f"Test {test_number:>3}: {'PASS' if passed else 'FAIL'} - {detail}")
        if not passed:
            failures.append(test_number)
    return failures


def main():
    repository_root = Path(__file__).resolve().parent.parent
    data_directory = (Path(sys.argv[1]) if len(sys.argv) > 1
                      else repository_root / "testfiles" / "data")
    outputs = run_functional_tests(data_directory)
    failures = validate_outputs(outputs, data_directory)
    if failures:
        print(f"\nFailed tests: {', '.join(failures)}")
        raise SystemExit(1)
    print(f"\nAll {len(outputs)} Functional Test 1 cases passed.")


if __name__ == "__main__":
    main()

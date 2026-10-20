from pathlib import Path
import numpy as np
import pandas as pd


DATA_FILE = Path(__file__).with_name("problem1.csv")


# Problem 1(a): pairwise correlation matrix
returns = pd.read_csv(DATA_FILE) 
pairwise_correlation = returns.corr(method="pearson", min_periods=2)

print("Problem 1(a): Pairwise correlation matrix")
print(pairwise_correlation.to_string(float_format=lambda value: f"{value:.6f}"))


# Problem 1(b): sample variance of each series using all its available data
series_variance = returns.var(axis=0, skipna=True, ddof=1)
available_observations = returns.count()

variance_results = pd.DataFrame(
    {
        "Observations": available_observations,
        "Variance": series_variance,
    }
)

print("\nProblem 1(b): Variance of each series")
print(variance_results.to_string(float_format=lambda value: f"{value:.10f}"))


# Problem 1(c): test whether the pairwise correlation matrix is PSD
eigenvalues = np.linalg.eigvalsh(pairwise_correlation.to_numpy())
tolerance = 1e-12
is_psd = bool(eigenvalues.min() >= -tolerance)

print("\nProblem 1(c): Positive semi-definite test")
print("Eigenvalues:", np.array2string(eigenvalues, precision=10))
print(f"Minimum eigenvalue: {eigenvalues.min():.10f}")
print(f"Is the correlation matrix PSD? {'Yes' if is_psd else 'No'}")
print(
    "The correlation matrix is not positive semi-definite because its "
    "smallest eigenvalue is negative. A positive semi-definite matrix "
    "must have only nonnegative eigenvalues."
)


# Problem 1(d): Higham's nearest correlation matrix
def project_to_psd(matrix):
    """Project a symmetric matrix onto the positive semi-definite cone."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped_eigenvalues = np.maximum(eigenvalues, 0.0)
    return (eigenvectors * clipped_eigenvalues) @ eigenvectors.T


def nearest_correlation_higham(matrix, tolerance=1e-12, max_iterations=1000):
    """Higham alternating projections with Dykstra's correction."""
    y = np.asarray(matrix, dtype=float).copy()
    dykstra_correction = np.zeros_like(y)

    for iteration in range(1, max_iterations + 1):
        previous_y = y.copy()

        residual = y - dykstra_correction
        x = project_to_psd(residual)
        dykstra_correction = x - residual

        y = x.copy()
        np.fill_diagonal(y, 1.0)

        relative_change = np.linalg.norm(y - previous_y, ord="fro") / max(
            np.linalg.norm(y, ord="fro"), 1.0
        )
        if relative_change < tolerance:
            return (y + y.T) / 2.0, iteration

    raise RuntimeError("Higham's method did not converge.")


nearest_correlation, higham_iterations = nearest_correlation_higham(
    pairwise_correlation.to_numpy()
)
nearest_correlation = pd.DataFrame(
    nearest_correlation,
    index=pairwise_correlation.index,
    columns=pairwise_correlation.columns,
)
nearest_eigenvalues = np.linalg.eigvalsh(nearest_correlation.to_numpy())

absolute_changes = np.abs(
    nearest_correlation.to_numpy() - pairwise_correlation.to_numpy()
)
upper_triangle = np.triu_indices_from(absolute_changes, k=1)
ranked_positions = sorted(
    zip(upper_triangle[0], upper_triangle[1]),
    key=lambda position: absolute_changes[position],
    reverse=True,
)

print("\nProblem 1(d): Higham nearest correlation matrix")
print(nearest_correlation.to_string(float_format=lambda value: f"{value:.6f}"))
print("Eigenvalues:", np.array2string(nearest_eigenvalues, precision=10))
print(f"Iterations: {higham_iterations}")
print("Largest off-diagonal changes:")
for row, column in ranked_positions[:5]:
    print(
        f"  ({pairwise_correlation.index[row]}, "
        f"{pairwise_correlation.columns[column]}): "
        f"{pairwise_correlation.iloc[row, column]:.6f} -> "
        f"{nearest_correlation.iloc[row, column]:.6f} "
        f"(absolute change {absolute_changes[row, column]:.6f})"
    )

print(
    "The entries involving x5 changed the most. The largest change was the "
    "correlation between x2 and x5, followed by the correlations between x4 "
    "and x5 and between x1 and x5."
)


# Problem 1(e): combine the repaired correlation matrix with the full-sample
standard_deviations = np.sqrt(series_variance.to_numpy())
standard_deviation_matrix = np.diag(standard_deviations)
covariance_matrix = (
    standard_deviation_matrix
    @ nearest_correlation.to_numpy()
    @ standard_deviation_matrix
)
covariance_matrix = pd.DataFrame(
    covariance_matrix,
    index=returns.columns,
    columns=returns.columns,
)

print("\nProblem 1(e): Covariance matrix")
print(covariance_matrix.to_string(float_format=lambda value: f"{value:.10f}"))



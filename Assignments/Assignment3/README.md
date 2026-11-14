# Assignment 3

This folder contains the completed responses to the options, option risk, and portfolio construction assignment.

## Files

- `responses.pdf`: written responses, tables, and figures.
- `analysis.py`: all pricing, statistics, simulations, regressions, and plots.
- `make_report.py`: generates the printable report from `results.json`.
- `pdf_report.py`: creates the final PDF from the HTML report.
- `responses.html`: source for the PDF.
- `problem1.csv`, `problem3.csv`, `problem3_portfolios.csv`, `problem4.csv`, `problem5.csv`, and the two Fama–French CSV files: supplied data.
- `problem1_results.csv`, `results.json`, and `figures/`: generated results.

## Requirements

Python 3.10 or later with NumPy, pandas, SciPy, Matplotlib, statsmodels, Beautiful Soup, and ReportLab.

Install packages with:

```bash
python3 -m pip install numpy pandas scipy matplotlib statsmodels beautifulsoup4 reportlab
```

## Reproduce all results

Run these commands from this folder:

```bash
MPLCONFIGDIR=/tmp/mpl-assignment3 python3 analysis.py
python3 make_report.py
python3 pdf_report.py
```

The first command prints progress and writes `results.json`, `problem1_results.csv`, and all figures. It takes several minutes (mostly Problem 2): the discrete-dividend tree values a separate subtree at every ex-dividend node, and the convergence table also runs N = 300 and 305. The fixed random seeds reproduce the simulations and bootstrap. The second command writes `responses.html` using the saved values.

The final command writes `responses.pdf`.

## Conventions

- Returns are arithmetic; reported VaR and ES are positive amounts for a loss.
- Problem 1 uses 35 calendar days divided by 365.
- Problem 2 uses CRR steps that land on both ex-dividend dates. Early exercise at those dates compares continuation after the dividend with immediate exercise before the dividend.
- Problem 3 uses 255 trading days per year, ten independent daily returns, zero simulation mean, and fixed option implied volatility. The fitted Student distribution's location is set to zero for the simulation while retaining its fitted degrees of freedom and scale.
- Problem 4 uses 3%/12 as the monthly risk-free return and monthly historical risk measures.
- Problem 5 divides Fama–French percent data by 100 and annualizes daily arithmetic means with 252 trading days.

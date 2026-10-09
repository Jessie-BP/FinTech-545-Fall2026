# Assignment 2

This folder contains the complete analysis for Assignment 2: Covariance, VaR,
and Copulas.

## Submitted files

- `responses.pdf` contains the written answers, tables, and figures.
- `analysis.py` performs every statistical calculation and simulation.
- `responses.html` is the written report; `responses.pdf` is printed from it.
- `problem1.csv` through `problem5.csv` are the supplied input data.

Generated supporting files are `results.json` and the PNG
files under `figures/`.

## Requirements

- Python 3.10 or later
- NumPy
- pandas
- SciPy
- Matplotlib
- Google Chrome (only needed to reproduce the PDF from the HTML)

Install the Python packages with:

```bash
python3 -m pip install -r requirements.txt
```

## Reproduce every result

Open a terminal in this `Assignment2` folder and run:

```bash
MPLCONFIGDIR=/tmp/mplconfig python3 analysis.py
```

The command prints every numerical result, creates the figures, and
writes `results.json`. The random seed is fixed inside `analysis.py`, so the
100,000-draw simulations reproduce the reported values.

On macOS, create the final PDF with:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$(pwd)/responses.pdf" "file://$(pwd)/responses.html"
```

Any modern browser can also open `responses.html`; use its Print command and
select “Save as PDF.”

## Numerical conventions

- Arithmetic returns are used throughout.
- VaR and ES are positive dollar loss numbers.
- Sample variance uses `n - 1`; kurtosis is excess kurtosis.
- Historical ES is the mean P&L at or below the estimated P&L quantile.
- Copula comparisons freeze the fitted margins and Kendall-tau correlation
  matrix. Gaussian therefore has zero remaining free parameters and the t
  copula has one (degrees of freedom), following the Week 5 notes.


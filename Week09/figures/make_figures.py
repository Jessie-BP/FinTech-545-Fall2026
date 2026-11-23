#!/usr/bin/env python3
"""
Regenerate the two data-driven figures for the Week 09 notes.

    python3 make_figures.py

Produces scree.png and vardecomp.png in this directory, using the same
FF3 fit that week09_02_ff3_time_series.jl performs.  The TikZ figures (taxonomy, sigma,
simflow) are built from their .tex sources by `make figures` at the
repository root.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
T_WINDOW = 60          # same window as week09_common.jl

# ---- load and fit FF3 --------------------------------------------------
# Same data and window as the Julia scripts: returns from adjusted closes,
# the last 60 days that the Fama-French file also covers.
px = pd.read_csv(os.path.join(DATA, "prices.csv"), parse_dates=["Date"]).set_index("Date")
ret = (px / px.shift(1) - 1).iloc[1:].reset_index()
stocks = [c for c in ret.columns if c not in ("Date", "SPY")]

ff = pd.read_csv(os.path.join(DATA, "ff3_daily.csv"))
ff["Date"] = pd.to_datetime(ff["Date"].astype(str), format="%Y%m%d")
for c in ["Mkt-RF", "SMB", "HML", "RF"]:
    ff[c] = ff[c] / 100.0

d = ret.merge(ff, on="Date", how="inner").sort_values("Date").tail(T_WINDOW).reset_index(drop=True)
fn = ["Mkt-RF", "SMB", "HML"]
T, n, m = len(d), len(stocks), len(fn)

Y = d[stocks].values - d[["RF"]].values
F = d[fn].values
X = np.column_stack([np.ones(T), F])
coef = np.linalg.lstsq(X, Y, rcond=None)[0]
B = coef[1:].T
E = Y - X @ coef
Fcov = np.cov(F, rowvar=False)
Dv = (E ** 2).sum(axis=0) / (T - m - 1)
S = np.cov(d[stocks].values, rowvar=False)

# ---- scree -------------------------------------------------------------
ev = np.linalg.eigvalsh(S)[::-1]
tot = ev.sum()
pct, cum = ev[:30] / tot * 100, np.cumsum(ev[:30]) / tot * 100

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].bar(range(1, 31), pct, color="#4a6fa5", edgecolor="none")
ax[0].axvline(3.5, color="#c44", ls="--", lw=1)
ax[0].set(xlabel="Component", ylabel="% of variance")
ax[0].set_title("Scree", fontsize=10)
ax[1].plot(range(1, 31), cum, marker="o", ms=3, color="#4a6fa5")
ax[1].axhline(90, color="#999", ls=":", lw=1)
ax[1].axvline(3, color="#c44", ls="--", lw=1)
ax[1].set(xlabel="Number of components", ylabel="Cumulative % explained", ylim=(0, 100))
ax[1].set_title("Cumulative variance", fontsize=10)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(alpha=.25, lw=.5)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "scree.png"), dpi=200)

# ---- variance decomposition -------------------------------------------
sysv = np.einsum("ij,jk,ik->i", B, Fcov, B)
tot_v = sysv + Dv
sel = ["NVDA", "TSLA", "AAPL", "MSFT", "XOM", "BRK-B",
       "CSCO", "JNJ", "PG", "KO", "MRK", "LLY"]
idx = [stocks.index(s) for s in sel]
share = sysv[idx] / tot_v[idx] * 100
o = np.argsort(-share)

w = np.ones(n) / n
x = B.T @ w
pshare = (x @ Fcov @ x) / (x @ Fcov @ x + (w ** 2 * Dv).sum()) * 100

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [3.6, 1]})
xs = np.arange(len(sel))
ax[0].bar(xs, share[o], color="#e08214", label="systematic $\\beta' F \\beta$")
ax[0].bar(xs, 100 - share[o], bottom=share[o], color="#4a9e6f", label="idiosyncratic $d_i$")
ax[0].set_xticks(xs)
ax[0].set_xticklabels([sel[i] for i in o], rotation=45, ha="right", fontsize=8)
ax[0].set(ylabel="% of variance", ylim=(0, 100))
ax[0].set_title("Single assets: idiosyncratic risk dominates", fontsize=10)
ax[0].legend(fontsize=8, frameon=False, loc="lower left")
ax[0].axhline(sysv.sum() / tot_v.sum() * 100, color="k", ls=":", lw=1)

ax[1].bar([0], [pshare], color="#e08214")
ax[1].bar([0], [100 - pshare], bottom=[pshare], color="#4a9e6f")
ax[1].set_xticks([0])
ax[1].set_xticklabels(["equal-wt\nportfolio"], fontsize=8)
ax[1].set(ylim=(0, 100), yticks=[])
ax[1].set_title("...and shrinks\nin the portfolio", fontsize=10)
ax[1].text(0, pshare / 2, "%.0f%%" % pshare, ha="center", va="center",
           fontsize=9, color="white", weight="bold")
ax[1].text(0, pshare + (100 - pshare) / 2, "%.1f%%" % (100 - pshare),
           ha="center", va="center", fontsize=8)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "vardecomp.png"), dpi=200)

print("wrote scree.png and vardecomp.png")

"""Regenerate the Week 08 plot figures.

    python3 make_figures.py

Writes money-vs-risk.png and risk-parity-weights.png.

No data is read. The numbers are the published results in the tables of
week08.qmd (tbl-p1weights, tbl-mvbudget, tbl-rpweights, tbl-unequal, tbl-esrp),
which come from week08_problem1.jl through week08_problem3.jl. If those results
change, update the constants below to match the notes.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

INK = "#1f2933"
MUTED = "#7b8794"
C1 = "#1f6feb"     # primary series
C2 = "#b54708"     # secondary series
C3 = "#087443"     # third series
C4 = "#9aa5b1"     # reference / baseline series

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.titlesize": 9, "axes.titleweight": "normal", "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e4e7eb", "grid.linewidth": 0.6,
})

STOCKS = ["AAPL", "MSFT", "BRK-B", "CSCO", "JNJ"]

# tbl-p1weights and tbl-mvbudget: the maximum Sharpe Ratio portfolio
MS_WEIGHT = np.array([10.076, 20.951, 43.839, 8.118, 17.015])
MS_RISK = np.array([11.8, 29.5, 39.7, 9.15, 9.8])

# tbl-rpweights, tbl-unequal, tbl-esrp: risk parity variants
RP_EQUAL = np.array([15.4, 14.3, 26.5, 15.1, 28.7])
RP_UNEQUAL = np.array([13.8, 23.2, 29.1, 14.7, 19.3])
RP_ES = np.array([14.9, 13.5, 26.4, 14.2, 31.0])


def _bars(ax, series, width):
    x = np.arange(len(STOCKS))
    k = len(series)
    for j, (label, vals, color) in enumerate(series):
        off = (j - (k - 1) / 2) * width
        ax.bar(x + off, vals, width, label=label, color=color, zorder=3)
    ax.set_xticks(x, STOCKS)
    ax.set_ylabel("percent of portfolio")
    ax.grid(axis="y", zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, loc="upper left", fontsize=8)


def money_vs_risk():
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    _bars(ax, [("share of capital (weight)", MS_WEIGHT, C1),
               ("share of risk (ex-ante)", MS_RISK, C2)], 0.36)
    ax.set_ylim(0, 50)
    ax.set_title("Maximum Sharpe Ratio portfolio: money against risk",
                 loc="left", pad=6)
    fig.tight_layout()
    fig.savefig("money-vs-risk.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote money-vs-risk.png")
    for s, w, r in zip(STOCKS, MS_WEIGHT, MS_RISK):
        print(f"  {s:6s} weight {w:7.3f}%  risk {r:6.2f}%")


def risk_parity_weights():
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    _bars(ax, [("equal budgets, volatility", RP_EQUAL, C1),
               ("MSFT 2, JNJ 0.5, volatility", RP_UNEQUAL, C2),
               ("equal budgets, ES", RP_ES, C3)], 0.26)
    ax.set_ylim(0, 36)
    ax.set_title("Risk parity weights", loc="left", pad=6)
    fig.tight_layout()
    fig.savefig("risk-parity-weights.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote risk-parity-weights.png")
    for s, a, b, c in zip(STOCKS, RP_EQUAL, RP_UNEQUAL, RP_ES):
        print(f"  {s:6s} equal {a:5.1f}%  unequal {b:5.1f}%  ES {c:5.1f}%")


if __name__ == "__main__":
    money_vs_risk()
    risk_parity_weights()

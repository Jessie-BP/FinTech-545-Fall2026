"""Assignment 3 calculations.

Run from any directory:

    python3 analysis.py

Every number in the written report comes from this script. Results are written to
results.json (read by make_report.py), problem1_results.csv, and figures/*.png.

Conventions used throughout (see README.md for the full list):
* Returns are arithmetic.
* VaR and ES are reported as positive numbers for a loss.
* Option pricing is written from scratch: generalized Black-Scholes-Merton (GBSM),
  its closed-form Greeks, and a Cox-Ross-Rubinstein (CRR) binomial tree.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
import statsmodels.api as sm
import matplotlib

matplotlib.use('Agg')  # write figures to files; no display needed
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
OUT = {}


def save(name, data):
    """Store one problem's results and rewrite results.json after every problem."""
    OUT[name] = data
    (HERE / 'results.json').write_text(json.dumps(OUT, indent=2, allow_nan=True))


# ---------------------------------------------------------------------------
# Option pricing building blocks
# ---------------------------------------------------------------------------

def intrinsic(s, k, typ):
    """Exercise value of a call or put at stock price s."""
    return np.maximum(s - k, 0) if typ == 'Call' else np.maximum(k - s, 0)


def gbsm(s, k, t, r, b, v, typ):
    """Generalized Black-Scholes-Merton price of a European option.

    b is the cost of carry: b = r for a non-dividend stock, b = r - q for a
    continuous dividend yield q. s may be an array (used for full revaluation).
    """
    s = np.asarray(s, dtype=float)
    if t <= 0:
        return intrinsic(s, k, typ)
    d1 = (np.log(s / k) + (b + v * v / 2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    carry = np.exp((b - r) * t)  # = exp(-q t): discount on the stock leg
    if typ == 'Call':
        return s * carry * stats.norm.cdf(d1) - k * np.exp(-r * t) * stats.norm.cdf(d2)
    return k * np.exp(-r * t) * stats.norm.cdf(-d2) - s * carry * stats.norm.cdf(-d1)


def greeks(s, k, t, r, b, v, typ):
    """Closed-form GBSM delta, gamma, theta, vega.

    Theta is dV/dt per year (calendar time passing), so a long option has negative
    theta. Vega is dollars per 1.00 (100 vol points) change in volatility.
    """
    d1 = (np.log(s / k) + (b + v * v / 2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    carry = np.exp((b - r) * t)
    phi = stats.norm.pdf(d1)
    if typ == 'Call':
        delta = carry * stats.norm.cdf(d1)
    else:
        delta = carry * (stats.norm.cdf(d1) - 1)
    gamma = carry * phi / (s * v * np.sqrt(t))
    vega = s * carry * phi * np.sqrt(t)
    decay = -s * carry * phi * v / (2 * np.sqrt(t))  # common time-decay term
    if typ == 'Call':
        theta = decay - (b - r) * s * carry * stats.norm.cdf(d1) - r * k * np.exp(-r * t) * stats.norm.cdf(d2)
    else:
        theta = decay + (b - r) * s * carry * stats.norm.cdf(-d1) + r * k * np.exp(-r * t) * stats.norm.cdf(-d2)
    return delta, gamma, theta, vega


def implied_vol(price, s, k, t, r, b, typ):
    """Invert GBSM for volatility with Brent's method.

    Returns NaN when no volatility can reproduce the price. A European option must
    lie between its discounted intrinsic value (vol -> 0) and an upper bound
    (vol -> infinity: the discounted stock for a call, the discounted strike for a
    put). A bid below the lower bound therefore has no implied volatility.
    """
    fwd_stock = s * np.exp((b - r) * t)
    pv_strike = k * np.exp(-r * t)
    if typ == 'Call':
        lower, upper = max(0.0, fwd_stock - pv_strike), fwd_stock
    else:
        lower, upper = max(0.0, pv_strike - fwd_stock), pv_strike
    if price < lower - 1e-10 or price >= upper or price <= 0:
        return np.nan
    if abs(price - lower) < 1e-10:
        return 0.0
    return optimize.brentq(lambda v: float(gbsm(s, k, t, r, b, v, typ)) - price, 1e-8, 10)


def tree(s, k, t, r, v, typ, n, divs=(), american=True):
    """CRR binomial tree with discrete cash dividends (non-recombining at ex-dates).

    divs is a list of (time, amount). Each dividend date must fall exactly on a
    tree step, which is why N is a multiple of 5 in Problem 2 (t = 0.1 and 0.4 of
    T = 0.5). Between dividends the tree recombines. At an ex-dividend step every
    node starts its own new recombining subtree from (price - dividend); that is
    the "non-recombining" part, because S*u - D then times d is not S*d - D then
    times u.

    Early exercise at an ex-dividend node compares exercising just BEFORE the
    dividend (pre-dividend price minus strike) with continuing AFTER it.
    """
    dt = t / n
    u = np.exp(v * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)  # risk-neutral up probability
    disc = np.exp(-r * dt)

    # Convert dividend times to step numbers and check they land exactly on steps.
    dates = [(int(round(time / dt)), float(amount)) for time, amount in divs]
    for (time, _), (step, _) in zip(divs, dates):
        assert abs(step * dt - time) < 1e-8, 'dividend date must fall on a tree step'

    def segment(root, start, remaining):
        """Value of the option at a node with price `root` at step `start`.

        Builds a recombining tree up to the next dividend step (or expiry) and
        rolls back to `start`.
        """
        stop = remaining[0][0] if remaining else n
        steps = stop - start
        up_moves = np.arange(steps + 1)
        nodes = root * u ** up_moves * d ** (steps - up_moves)  # prices at `stop`

        if remaining:
            # Ex-dividend step: each node seeds a subtree starting at price - D.
            amount = remaining[0][1]
            post = np.array([segment(max(x - amount, 1e-9), stop, remaining[1:]) for x in nodes])
            values = np.maximum(post, intrinsic(nodes, k, typ)) if american else post
        else:
            values = intrinsic(nodes, k, typ)  # expiry payoff

        # Standard backward induction inside the recombining segment.
        for j in range(steps - 1, -1, -1):
            values = disc * (p * values[1:] + (1 - p) * values[:-1])
            if american:
                up_j = np.arange(j + 1)
                now = root * u ** up_j * d ** (j - up_j)
                values = np.maximum(values, intrinsic(now, k, typ))
        return float(values[0])

    return segment(s, 0, dates)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def risk(x, a):
    """Empirical VaR and ES at level a, as positive losses.

    VaR = -(a quantile). ES = -(mean of outcomes at or below that quantile).
    """
    q = float(np.quantile(x, a))
    tail = x[x <= q]
    return {'VaR': -q, 'ES': -float(np.mean(tail))}


def moments(x):
    """First four moments; unbiased sample versions (ddof=1, bias-corrected)."""
    return {
        'mean': float(np.mean(x)),
        'std': float(np.std(x, ddof=1)),
        'skew': float(stats.skew(x, bias=False)),
        'excess_kurtosis': float(stats.kurtosis(x, bias=False)),
    }


def aicc(loglik, k, n):
    """Small-sample corrected AIC; k counts all estimated parameters."""
    return 2 * k - 2 * loglik + 2 * k * (k + 1) / (n - k - 1)


# ---------------------------------------------------------------------------
# Problem 1: implied volatility from a dirty chain
# ---------------------------------------------------------------------------

def problem1():
    df = pd.read_csv(HERE / 'problem1.csv')
    s, r, q = 182.4, .0425, .006
    b = r - q          # cost of carry with a continuous dividend yield
    t = 35 / 365       # 11/13/2026 -> 12/18/2026 is 35 calendar days

    rows = []
    for k, group in df.groupby('Strike'):
        c = group[group.Type == 'Call'].iloc[0]
        p = group[group.Type == 'Put'].iloc[0]
        call_mid = (c.Bid + c.Ask) / 2
        put_mid = (p.Bid + p.Ask) / 2

        # Put-call parity: C - P = S exp(-qT) - K exp(-rT).
        # gap > 0 means the call is rich relative to the put (or the put is cheap).
        gap = call_mid - put_mid - (s * np.exp((b - r) * t) - k * np.exp(-r * t))
        # A gap is only tradeable if it exceeds both spreads combined.
        band = (c.Ask - c.Bid) + (p.Ask - p.Bid)
        row = {'strike': k, 'parity_gap': gap, 'band': band, 'flag': abs(gap) > band}

        for typ, quote in [('Call', c), ('Put', p)]:
            # Invert bid, mid, and ask separately; the bid/ask vols show how wide
            # the volatility uncertainty implied by the quote really is.
            for side, price in [('bid', quote.Bid), ('mid', (quote.Bid + quote.Ask) / 2), ('ask', quote.Ask)]:
                row[typ + '_' + side] = implied_vol(price, s, k, t, r, b, typ)
            vol = row[typ + '_mid']
            row[typ + '_vega'] = greeks(s, k, t, r, b, vol, typ)[3] if np.isfinite(vol) else np.nan
            # Vol change from a $0.01 price change: dsigma = 0.01 / vega, in vol points (x100).
            row[typ + '_cent_vol_points'] = 100 * .01 / row[typ + '_vega'] if row[typ + '_vega'] > 0 else np.nan
        rows.append(row)

    tab = pd.DataFrame(rows)
    tab.to_csv(HERE / 'problem1_results.csv', index=False)

    plt.figure(figsize=(10, 5))
    for typ, color in [('Call', 'C0'), ('Put', 'C1')]:
        plt.plot(tab.strike, 100 * tab[typ + '_mid'], 'o-', label=typ + ' mid', color=color)
        plt.fill_between(tab.strike, 100 * tab[typ + '_bid'], 100 * tab[typ + '_ask'],
                         alpha=.2, color=color, label=typ + ' bid–ask')
    plt.xlabel('Strike')
    plt.ylabel('Implied volatility (%)')
    plt.legend()
    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(FIG / 'problem1_smile.png', dpi=170)
    plt.close()

    save('problem1', {'t': t, 'rows': tab.replace({np.nan: None}).to_dict('records')})


# ---------------------------------------------------------------------------
# Problem 2: American options and discrete dividends
# ---------------------------------------------------------------------------

def problem2():
    s, k, t, r, v = 48.5, 50, .5, .045, .3
    divs = [(.1, .25), (.4, 1.5)]

    # Week 7 necessary condition for early call exercise just before dividend n:
    # D_n > K (1 - exp(-r (T - t_n))).
    out = {'thresholds': [k * (1 - np.exp(-r * (t - ti))) for ti, _ in divs]}

    for typ in ['Call', 'Put']:
        out[typ] = {}
        # Neighbouring N values show the odd/even oscillation of the tree.
        for n in [100, 105, 150, 155, 200, 205]:
            out[typ][str(n)] = {
                'american_div': tree(s, k, t, r, v, typ, n, divs, True),
                'european_div': tree(s, k, t, r, v, typ, n, divs, False),
                'american_no_div': tree(s, k, t, r, v, typ, n, (), True),
                'european_no_div': tree(s, k, t, r, v, typ, n, (), False),
            }
        # Larger N for the convergence table only (slow: the non-recombining tree
        # grows roughly with N^3, so only the American dividend case is run).
        out[typ]['convergence_large_n'] = {str(n): tree(s, k, t, r, v, typ, n, divs, True) for n in [300, 305]}

        out[typ]['gbsm_no_div'] = float(gbsm(s, k, t, r, r, v, typ))
        # Escrowed-dividend approximation: subtract PV of dividends from spot.
        pv = sum(d * np.exp(-r * ti) for ti, d in divs)
        out[typ]['gbsm_spot_pv'] = float(gbsm(s - pv, k, t, r, r, v, typ))

        # Central finite differences on the N = 200 tree. N is held fixed, so the
        # tree value is a piecewise-linear function of S0 (see report, part h).
        out[typ]['greeks'] = {}
        n = 200
        base = out[typ][str(n)]['american_div']
        for h in [1, .1, .01, .001]:
            up = tree(s + h, k, t, r, v, typ, n, divs, True)
            dn = tree(s - h, k, t, r, v, typ, n, divs, True)
            out[typ]['greeks'][str(h)] = {'delta': (up - dn) / (2 * h), 'gamma': (up - 2 * base + dn) / (h * h)}

        # dV/dD_i by central difference with a 1 cent bump on one dividend at a time.
        out[typ]['dividend_sensitivity'] = {}
        bump = .01
        for i in range(2):
            plus = divs.copy()
            minus = divs.copy()
            plus[i] = (plus[i][0], plus[i][1] + bump)
            minus[i] = (minus[i][0], minus[i][1] - bump)
            for amer, label in [(False, 'european'), (True, 'american')]:
                out[typ]['dividend_sensitivity'][f'{label}_D{i + 1}'] = (
                    tree(s, k, t, r, v, typ, n, plus, amer) - tree(s, k, t, r, v, typ, n, minus, amer)) / (2 * bump)

    # Exercise boundary just before the second dividend: the stock price x at which
    # exercising now (x - K) equals holding through the dividend. After D2 there are
    # no more dividends, so the continuation value is a 0.1-year call on x - 1.5.
    # It is valued with a separate 40-step CRR tree and cross-checked with GBSM
    # (an American call with no further dividends equals the European call).
    remaining, nsub = .1, 40
    out['call_exercise_boundary_before_D2'] = optimize.brentq(
        lambda x: x - k - tree(x - 1.5, k, remaining, r, v, 'Call', nsub, (), True), k + 1.5, 200)
    out['call_exercise_boundary_before_D2_gbsm'] = optimize.brentq(
        lambda x: x - k - float(gbsm(x - 1.5, k, remaining, r, r, v, 'Call')), k + 1.5, 200)
    save('problem2', out)


# ---------------------------------------------------------------------------
# Problem 3: VaR of an option book
# ---------------------------------------------------------------------------

def problem3():
    px = pd.read_csv(HERE / 'problem3.csv')
    positions = pd.read_csv(HERE / 'problem3_portfolios.csv')
    s = float(px.ABC.iloc[-1])   # value everything at the last price
    r = .0425                    # no dividends, so cost of carry b = r
    year = 255                   # trading days per year

    # Daily arithmetic returns, moments, and the two candidate distributions.
    rets = px.ABC.pct_change().dropna().to_numpy()
    m = moments(rets)
    mu, sd = stats.norm.fit(rets)   # MLE (ddof = 0) normal
    tf = stats.t.fit(rets)          # (df, loc, scale)
    fits = {
        'normal': {'params': [mu, sd], 'AICc': aicc(np.sum(stats.norm.logpdf(rets, mu, sd)), 2, len(rets))},
        'student_t': {'params': list(tf), 'AICc': aicc(np.sum(stats.t.logpdf(rets, *tf)), 3, len(rets))},
    }

    # Simulate 10-day terminal prices. Mean/location is forced to zero as the
    # problem says; scale (and df for t) keep their fitted values. Ten daily
    # returns are compounded, so the 10-day return is not exactly normal.
    rng = np.random.default_rng(54503)
    n_draw = 250000
    horizon = {}
    for name in fits:  # order matters for reproducibility: normal first, then t
        if name == 'normal':
            daily = rng.normal(0, sd, (n_draw, 10))
        else:
            daily = stats.t.rvs(tf[0], loc=0, scale=tf[2], size=(n_draw, 10), random_state=rng)
        horizon[name] = s * np.prod(1 + daily, axis=1)

    books = {}
    for book, group in positions.groupby('Portfolio', sort=False):

        def full(new_s, days):
            """Full revaluation of the book at price new_s with `days` trading days left.

            Implied vols are held constant, as instructed.
            """
            result = 0
            for _, row in group.iterrows():
                if row.Type == 'Stock':
                    result += row.Holding * new_s
                else:
                    result += row.Holding * gbsm(new_s, row.Strike, days / year, r, r, row.ImpliedVol, row.Type)
            return result

        # Current value and book Greeks (sum of holding x per-option Greek).
        value = delta = gamma = theta = 0.0
        for _, row in group.iterrows():
            if row.Type == 'Stock':
                value += row.Holding * s
                delta += row.Holding
            else:
                t = row.DaysToExpiry / year
                g = greeks(s, row.Strike, t, r, r, row.ImpliedVol, row.Type)
                value += row.Holding * gbsm(s, row.Strike, t, r, r, row.ImpliedVol, row.Type)
                delta += row.Holding * g[0]
                gamma += row.Holding * g[1]
                theta += row.Holding * g[2]

        # 30 days to expiry today; after a 10-day horizon, 20 days remain.
        item = {'value': float(value), 'delta': float(delta), 'gamma': float(gamma),
                'theta_annual': float(theta), 'flat_price_pnl': float(full(s, 20) - value)}

        # Delta-normal: P&L = delta * dS, dS ~ N(0, (S sd sqrt(10))^2).
        dn_sd = abs(delta) * s * sd * np.sqrt(10)
        item['delta_normal'] = {}
        for a in [.05, .01]:
            z = stats.norm.ppf(a)
            item['delta_normal'][str(a)] = {'VaR': -dn_sd * z, 'ES': dn_sd * stats.norm.pdf(z) / a}

        item['simulated'] = {}
        for name, st in horizon.items():
            pnl = full(st, 20) - value
            res = {'mean': float(np.mean(pnl)), '5%': risk(pnl, .05), '1%': risk(pnl, .01)}
            # Second-order (delta-gamma-theta) approximation on the same scenarios.
            approx = delta * (st - s) + .5 * gamma * (st - s) ** 2 + theta * 10 / year
            res['gamma_approx'] = {str(a): risk(approx, a) for a in [.05, .01]}
            item['simulated'][name] = res

        # P&L profile: full revaluation vs delta-only line.
        grid = np.linspace(s * .6, s * 1.4, 400)
        plt.figure(figsize=(8, 4.5))
        plt.plot(grid, full(grid, 20) - value, label='Full revaluation')
        plt.plot(grid, delta * (grid - s), label='Delta only')
        plt.axhline(0, color='black', lw=.6)
        plt.xlabel('ABC after 10 days ($)')
        plt.ylabel('P&L ($)')
        plt.legend()
        plt.grid(alpha=.25)
        plt.tight_layout()
        plt.savefig(FIG / f'problem3_{book}.png', dpi=170)
        plt.close()
        books[book] = item

    save('problem3', {'spot': s, 'return_moments': m, 'fits': fits, 'books': books, 'simulation_draws': n_draw})


# ---------------------------------------------------------------------------
# Problem 4: a Sharpe ratio that sells insurance
# ---------------------------------------------------------------------------

def problem4():
    df = pd.read_csv(HERE / 'problem4.csv')
    names = ['EQ', 'BOND', 'STK', 'PUTW']
    x = df[names].to_numpy()
    rf = .03 / 12                     # monthly risk-free return
    mu = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    excess = mu - rf

    # Per-asset statistics. "sharpe_without_worst" drops the single worst month
    # to show how much one observation drives each Sharpe ratio.
    asset = {}
    for i, name in enumerate(names):
        col = x[:, i]
        worst = int(np.argmin(col))
        rest = np.delete(col, worst)
        asset[name] = {
            **moments(col),
            'sharpe_annual': (mu[i] - rf) / np.std(col, ddof=1) * np.sqrt(12),
            'mean_se': np.std(col, ddof=1) / np.sqrt(len(x)),
            'worst_month': float(col.min()),
            'worst_month_index': worst,
            'sharpe_without_worst': float((rest.mean() - rf) / rest.std(ddof=1) * np.sqrt(12)),
        }

    def sharpe(w):
        """Annualized Sharpe ratio of weights w (monthly ratio x sqrt(12))."""
        return (w @ excess) / np.sqrt(w @ cov @ w) * np.sqrt(12)

    def max_sharpe(bounds):
        """Fully invested max-Sharpe portfolio under box bounds (SLSQP)."""
        return optimize.minimize(lambda w: -sharpe(w), np.ones(4) / 4, bounds=bounds,
                                 constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1},
                                 method='SLSQP', options={'ftol': 1e-13, 'maxiter': 500}).x

    # Unconstrained tangency has a closed form: w proportional to inv(cov) @ excess.
    w0 = np.linalg.solve(cov, excess)
    w0 /= w0.sum()
    w1 = max_sharpe([(0, 1)] * 4)                 # long only
    w2 = max_sharpe([(0, 1)] * 3 + [(0, .2)])     # long only, PUTW <= 20%

    portfolios = {}
    for label, w in [('unconstrained', w0), ('long_only', w1), ('putw_cap_20', w2)]:
        p = x @ w   # realized monthly portfolio returns
        psd = np.std(p, ddof=1)
        entry = {'weights': dict(zip(names, map(float, w))), 'sharpe_annual': float(sharpe(w)),
                 'mean_monthly': float(np.mean(p)), 'std_monthly': float(psd), 'historical': {}, 'normal': {}}
        for a in [.05, .01]:
            entry['historical'][str(a)] = risk(p, a)
            # Normal with the same mean and sd as the historical portfolio returns.
            z = stats.norm.ppf(a)
            entry['normal'][str(a)] = {'VaR': -float(np.mean(p) + psd * z),
                                       'ES': -float(np.mean(p) - psd * stats.norm.pdf(z) / a)}
        portfolios[label] = entry

    # CAPM of PUTW on EQ, all months, then only months with EQ excess < -5%.
    capm = sm.OLS(x[:, 3] - rf, sm.add_constant(x[:, 0] - rf)).fit()
    down = (x[:, 0] - rf) < -.05
    downfit = sm.OLS(x[down, 3] - rf, sm.add_constant(x[down, 0] - rf)).fit()

    # Long-only efficient frontier: minimum variance at each target mean.
    fr = []
    for target in np.linspace(mu.min(), mu.max(), 120):
        res = optimize.minimize(lambda w: w @ cov @ w, np.ones(4) / 4, bounds=[(0, 1)] * 4,
                                constraints=[{'type': 'eq', 'fun': lambda w: sum(w) - 1},
                                             {'type': 'eq', 'fun': lambda w: w @ mu - target}],
                                method='SLSQP', options={'ftol': 1e-12, 'maxiter': 1000})
        if res.success:
            fr.append((np.sqrt(res.fun) * np.sqrt(12), target * 12))
    fr = np.array(fr)

    plt.figure(figsize=(8, 5))
    plt.plot(fr[:, 0], fr[:, 1], label='Long-only frontier')
    vol = np.linspace(0, max(fr[:, 0]), 100)
    plt.plot(vol, .03 + portfolios['long_only']['sharpe_annual'] * vol, label='Capital market line')
    for i, name in enumerate(names):
        point = (np.sqrt(cov[i, i] * 12), mu[i] * 12)
        plt.scatter(*point)
        plt.annotate(name, point, xytext=(4, 4), textcoords='offset points')
    plt.xlabel('Annual volatility')
    plt.ylabel('Annual arithmetic mean')
    plt.legend()
    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(FIG / 'problem4_frontier.png', dpi=170)
    plt.close()

    # Bootstrap: resample the 240 months with replacement and re-optimize the
    # long-only tangency. The spread of weights shows estimation error.
    rng = np.random.default_rng(54504)
    boots = []
    for _ in range(500):
        xb = x[rng.integers(len(x), size=len(x))]
        mb = xb.mean(axis=0)
        cb = np.cov(xb, rowvar=False)
        rr = optimize.minimize(lambda w: -(w @ (mb - rf)) / np.sqrt(w @ cb @ w), np.ones(4) / 4,
                               bounds=[(0, 1)] * 4, constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1},
                               method='SLSQP')
        if rr.success:
            boots.append(rr.x)
    boots = np.array(boots)

    save('problem4', {
        'assets': asset,
        'portfolios': portfolios,
        'capm': {'alpha_monthly': capm.params[0], 'alpha_t': capm.tvalues[0], 'beta': capm.params[1],
                 'r2': capm.rsquared, 'downside_beta': downfit.params[1], 'downside_n': int(down.sum()),
                 'downside_beta_se': downfit.bse[1]},
        'bootstrap_long_only_weight_quantiles': {name: np.quantile(boots[:, i], [.05, .5, .95]).tolist()
                                                 for i, name in enumerate(names)},
    })


# ---------------------------------------------------------------------------
# Problem 5: alpha or factor exposure?
# ---------------------------------------------------------------------------

def problem5():
    funds = pd.read_csv(HERE / 'problem5.csv', parse_dates=['Date'])
    ff = pd.read_csv(HERE / 'F-F_Research_Data_Factors_daily.CSV')
    mom = pd.read_csv(HERE / 'F-F_Momentum_Factor_daily.CSV')
    ff['Date'] = pd.to_datetime(ff.Date.astype(str), format='%Y%m%d')
    mom['Date'] = pd.to_datetime(mom.Date.astype(str), format='%Y%m%d')
    mom.columns = ['Date', 'Mom']  # the raw header has trailing spaces

    fac = ff.merge(mom, on='Date')
    cols = ['Mkt-RF', 'SMB', 'HML', 'RF', 'Mom']
    fac[cols] = fac[cols] / 100    # Fama-French files are in percent

    merged = funds.merge(fac, on='Date', how='inner')
    # "Most recent 10 years in the factor files" = last 10 years before the final date.
    recent = fac[fac.Date >= fac.Date.max() - pd.DateOffset(years=10)]
    premia = merged[['Mkt-RF', 'SMB', 'HML', 'Mom']].mean()   # in-sample factor premia

    out = {'aligned_days': len(merged), 'factor_last_date': str(fac.Date.max().date()),
           'recent_factor_mean_daily': recent[cols].mean().to_dict(),
           'sample_factor_mean_daily': premia.to_dict(), 'funds': {}}
    worst = fac.loc[fac.Mom.idxmin()]   # worst momentum day in the whole file
    out['worst_momentum'] = {'date': str(worst.Date.date()), 'return': worst.Mom}

    for name in ['FUND1', 'FUND2']:
        y = merged[name] - merged.RF   # daily excess return
        yraw = merged[name]
        d = {'mean_excess_daily': float(y.mean()), 'vol_daily': float(yraw.std(ddof=1)),
             'sharpe_annual': float(y.mean() / yraw.std(ddof=1) * np.sqrt(252)),
             'realized_annual_arithmetic': float(yraw.mean() * 252)}

        models = {}
        for label, predictors in [('CAPM', ['Mkt-RF']), ('FF3', ['Mkt-RF', 'SMB', 'HML']),
                                  ('Carhart4', ['Mkt-RF', 'SMB', 'HML', 'Mom'])]:
            fit = sm.OLS(y, sm.add_constant(merged[predictors])).fit()
            k, n = len(fit.params), len(y)
            # statsmodels AIC counts the intercept + slopes; add the small-sample term.
            aic_c = fit.aic + 2 * k * (k + 1) / (n - k - 1)
            models[label] = {'coefficients': fit.params.to_dict(), 'standard_errors': fit.bse.to_dict(),
                             'alpha_t': float(fit.tvalues['const']), 'r2': float(fit.rsquared), 'AICc': float(aic_c)}
        d['models'] = models

        # Which factors does the CAPM residual still contain?
        resid = sm.OLS(y, sm.add_constant(merged[['Mkt-RF']])).fit().resid
        d['capm_resid_factor_corr'] = {c: float(np.corrcoef(resid, merged[c])[0, 1]) for c in ['SMB', 'HML', 'Mom']}

        # Expected return = rf + sum(beta x recent factor mean), annualized x252.
        beta = models['Carhart4']['coefficients']
        predicted = recent.RF.mean() + sum(beta[c] * recent[c].mean() for c in ['Mkt-RF', 'SMB', 'HML', 'Mom'])
        d['expected_annual_no_alpha'] = float(252 * predicted)
        d['expected_annual_with_alpha'] = float(252 * (predicted + beta['const']))

        # Momentum crash replay: only the Mom factor moves; loss positive.
        shock = beta['Mom'] * worst.Mom
        d['momentum_shock_loss_10m'] = -float(1e7 * shock)
        d['momentum_shock_std_multiple'] = -float(shock / yraw.std(ddof=1))
        out['funds'][name] = d

    save('problem5', out)


if __name__ == '__main__':
    for fn in [problem1, problem2, problem3, problem4, problem5]:
        print('Running', fn.__name__, flush=True)
        fn()
    print('Wrote', HERE / 'results.json')

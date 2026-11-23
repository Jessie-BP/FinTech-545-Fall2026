#!/usr/bin/env python3
"""
Download the Week 09 data set.

    python get_data.py

Writes, into this folder:

  spy_holdings.xlsx   raw State Street SPY daily holdings file (the source record)
  constituents.csv    the 100 largest stocks in SPY: ticker, name, sector, the
                      8-group industry used in the notes, SPY weight, and the
                      float-adjusted market cap
  prices.csv          daily adjusted closes for SPY and the 100 stocks, one year
  ff3_daily.csv       Fama-French 3 factor daily returns, in PERCENT as published

Sources
  - SPY holdings and weights: State Street (ssga.com), daily holdings file.
    SPY replicates the S&P 500, so its weights ARE the index's float-adjusted
    market cap weights.
  - Prices and sectors: Yahoo Finance, through the yfinance package.
  - Fama-French factors: Ken French's data library.  Some networks block that
    site.  If the download fails, get F-F_Research_Data_Factors_daily_CSV.zip
    in a browser, put it in this folder, and run the script again -- it will
    use the local copy.

Float-adjusted market cap
  SPY holds the same fraction of every company's float-adjusted shares, so
  float cap_i is proportional to (shares SPY holds)_i x price_i.  The constant
  of proportionality is SPY's ownership fraction, k.  We estimate k as the
  median over the 100 names of (SPY shares held / Yahoo float shares) and
  report  FloatAdjMktCap_i = shares held_i x price_i / k.  The RATIOS -- which
  is all a cap weight uses -- are exactly SPY's weights; k only sets the
  dollar level.
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

HERE = Path(__file__).resolve().parent
N_STOCKS = 100
HOLDINGS_URL = ("https://www.ssga.com/us/en/intermediary/library-content/"
                "products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx")
FF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_daily_CSV.zip")
FF_ZIP = HERE / "F-F_Research_Data_Factors_daily_CSV.zip"
PRICE_START = "2025-10-01"

# Yahoo sector -> the 8 groups used in the notes.  The four smallest sectors
# are pooled into "Other" so no group has only one or two stocks.
GROUP = {
    "Technology": "Tech",
    "Communication Services": "Comm",
    "Consumer Cyclical": "Discr",
    "Financial Services": "Fin",
    "Healthcare": "Health",
    "Industrials": "Indust",
    "Consumer Defensive": "Staples",
    "Energy": "Other",
    "Utilities": "Other",
    "Basic Materials": "Other",
    "Real Estate": "Other",
}


def get_holdings() -> tuple[pd.DataFrame, str]:
    """Top N stocks in SPY by weight, and the holdings date."""
    raw = HERE / "spy_holdings.xlsx"
    r = requests.get(HOLDINGS_URL, timeout=60)
    r.raise_for_status()
    raw.write_bytes(r.content)

    head = pd.read_excel(raw, header=None, nrows=4)
    as_of = str(head.iloc[2, 1]).replace("As of ", "")
    h = pd.read_excel(raw, skiprows=4)
    h = h[h["Ticker"].notna() & (h["Ticker"] != "-") & h["Weight"].notna()]
    h = h[h["Local Currency"] == "USD"]
    h = h.sort_values("Weight", ascending=False).head(N_STOCKS).copy()
    h["Ticker"] = h["Ticker"].str.replace(".", "-", regex=False)   # BRK.B -> BRK-B
    return h[["Ticker", "Name", "Weight", "Shares Held"]].reset_index(drop=True), as_of


def get_info(tickers: list[str]) -> pd.DataFrame:
    """Sector, float shares, total market cap and last price from Yahoo."""
    rows = []
    for t in tickers:
        info = yf.Ticker(t).info
        rows.append({
            "Ticker": t,
            "Sector": info.get("sector"),
            "FloatShares": info.get("floatShares"),
            "MktCap": info.get("marketCap"),
            "Price": info.get("currentPrice") or info.get("regularMarketPrice"),
        })
    return pd.DataFrame(rows)


def get_prices(tickers: list[str]) -> pd.DataFrame:
    """Daily adjusted closes, SPY first."""
    px = yf.download(["SPY"] + tickers, start=PRICE_START, auto_adjust=True,
                     progress=False)["Close"]
    px = px[["SPY"] + tickers].dropna(how="all")
    px.index.name = "Date"
    return px


def get_ff3() -> pd.DataFrame:
    """Fama-French 3 daily factors, in percent, Date as yyyymmdd."""
    if not FF_ZIP.exists():
        r = requests.get(FF_URL, timeout=60)
        r.raise_for_status()
        FF_ZIP.write_bytes(r.content)
    with zipfile.ZipFile(FF_ZIP) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(",Mkt-RF"))
    body = []
    for l in lines[start + 1:]:
        if not l.strip() or not l.strip()[0].isdigit():
            break
        body.append(l)
    ff = pd.read_csv(io.StringIO("\n".join(body)), header=None,
                     names=["Date", "Mkt-RF", "SMB", "HML", "RF"])
    ff["Date"] = ff["Date"].astype(str).str.strip()
    return ff


def write_ff3() -> None:
    try:
        ff = get_ff3()
        ff.to_csv(HERE / "ff3_daily.csv", index=False)
        print(f"ff3_daily.csv: through {ff['Date'].iloc[-1]}")
    except Exception as e:
        print(f"Fama-French download failed ({type(e).__name__}). Download\n  {FF_URL}\n"
              f"in a browser, save it in {HERE}, and run\n  python get_data.py --ff-only")


def main() -> None:
    if "--ff-only" in sys.argv:       # refresh only the factors, keep prices and holdings
        write_ff3()
        return
    hold, as_of = get_holdings()
    print(f"SPY holdings as of {as_of}: top {len(hold)} names, "
          f"{hold['Weight'].sum():.1f}% of the fund")

    info = get_info(hold["Ticker"].tolist())
    c = hold.merge(info, on="Ticker")
    missing = c[c["Sector"].isna()]["Ticker"].tolist()
    if missing:
        raise SystemExit(f"no Yahoo sector for: {missing}")
    c["Group"] = c["Sector"].map(GROUP)

    k = (c["Shares Held"] / c["FloatShares"]).median()
    c["FloatAdjMktCap"] = c["Shares Held"] * c["Price"] / k / 1e9
    c["MktCap"] = c["MktCap"] / 1e9
    c["HoldingsDate"] = as_of
    c = c.rename(columns={"Weight": "SPYWeight", "Shares Held": "SPYSharesHeld"})
    c.insert(0, "Rank", range(1, len(c) + 1))
    c = c[["Rank", "Ticker", "Name", "Sector", "Group", "SPYWeight", "SPYSharesHeld",
           "Price", "FloatAdjMktCap", "MktCap", "HoldingsDate"]]
    c.to_csv(HERE / "constituents.csv", index=False, float_format="%.6f")
    print(f"SPY ownership fraction k = {k:.4%}; constituents.csv written")
    print(c.groupby("Group").size().to_string())

    px = get_prices(c["Ticker"].tolist())
    bad = [t for t in px.columns if px[t].isna().any()]
    if bad:
        print(f"warning: missing prices for {bad}")
    px.to_csv(HERE / "prices.csv", float_format="%.6f")
    print(f"prices.csv: {px.index[0].date()} to {px.index[-1].date()}, {len(px)} days")

    write_ff3()


if __name__ == "__main__":
    main()

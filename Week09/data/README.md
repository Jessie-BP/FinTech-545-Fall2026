# Week 09 data

All Week 09 scripts and figures read from this folder. Rebuild it with

    python get_data.py

(requires `pip install yfinance openpyxl pandas requests`).

| File | Contents | Source |
|:--|:--|:--|
| `constituents.csv` | The 100 largest stocks in SPY: ticker, name, sector, the 8-group industry used in the notes, SPY weight, float-adjusted market cap ($bn), total market cap ($bn) | State Street SPY holdings, Yahoo Finance |
| `prices.csv` | Daily adjusted closes for SPY and the 100 stocks | Yahoo Finance |
| `ff3_daily.csv` | Fama-French 3 factor daily returns, in percent as published | Ken French data library |
| `spy_holdings.xlsx` | The raw SPY holdings file the constituents came from | State Street |

## Notes

- **Universe.** The top 100 names by weight in SPY on the holdings date in `constituents.csv`. Both Alphabet share classes (GOOGL, GOOG) are included, as they are in SPY.
- **Window.** The scripts use the last 60 trading days that the Fama-French file also covers (`T_WINDOW` in `week09_common.jl`). Ken French's file lags by a month or two, so the window ends earlier than the price data.
- **Industry groups.** Yahoo sectors, with Energy, Utilities, Basic Materials, and Real Estate pooled into "Other." Yahoo's classification is close to GICS but not identical -- it puts UBER and ADP in Technology, for example.
- **Market cap.** SPY holds the same fraction of every company's float-adjusted shares, so SPY's weights are float-adjusted cap weights. `FloatAdjMktCap` keeps those ratios exactly and sets the dollar level from SPY's estimated ownership fraction. It is measured on the holdings date, not at the end of the return window.
- **Fama-French download.** Some networks block Ken French's site. If `get_data.py` cannot download it, save `F-F_Research_Data_Factors_daily_CSV.zip` from <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html> into this folder and run the script again.

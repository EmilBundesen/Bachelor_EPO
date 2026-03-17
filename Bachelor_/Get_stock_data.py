import yfinance as yf
import pandas as pd


# ── Konfiguration ─────────────────────────────────────────────
sectors = {
    "pharma":   ["PFE","MRK","BMY","JNJ","AMGN","ABT","BAX","LLY",
                 "GILD","MDT","CAH","MCK","CI","UNH","BDX",
                 "SYK","ZBH","HUM","DVA","WST"],
    "defense":  ["LMT","NOC","RTX","GD","BA","TXT","LHX","HII",
                 "CW","TDG","HEI","BWXT","LDOS","SAIC","CACI",
                 "HON","GE","MMM","EMR","ETN"],
    "energy":   ["XOM","CVX","COP","EOG","OXY","DVN","APA","SLB",
                 "HAL","BKR","KMI","WMB","OKE","PSX","VLO",
                 "MPC","SUN","FANG"],
    "tech":     ["MSFT","INTC","IBM","ORCL","CSCO","QCOM","TXN","ADI",
                 "AMAT","LRCX","KLAC","HPQ","ADBE","SAP","ACN",
                 "MU","NXPI","AVGO","CRM","DELL"],
    "consumer": ["KO","PEP","PG","CL","KMB","GIS","MDLZ","UL",
                 "PM","MO","BTI","DEO","BUD","STZ","BF-B",
                 "HRL","CPB","SJM","HSY","MKC"],
}

START_DATE = "1990-01-01"
END_DATE   = "2025-12-31"

# Filstier
DAILY_RETS_PATH   = "data_daily_returns.parquet"
DAILY_PRICES_PATH = "data_daily_prices.parquet"
SECTOR_PATH       = "data_ticker_sector.csv"


def fetch_and_save():
    """
    Henter aktiedata fra Yahoo Finance og gemmer som parquet/csv.
    Køres kun én gang — eller når data skal opdateres.
    """
    all_returns, all_prices, ticker_to_sector = [], [], {}

    for sector, tickers in sectors.items():
        print(f"Henter {sector}...")
        data = yf.download(tickers, start=START_DATE, end=END_DATE,
                           auto_adjust=True, threads=False,
                           progress=True)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])

        rets  = data.pct_change(fill_method=None)
        valid = rets.columns[rets.isna().mean() <= 0.20].tolist()
        rets  = rets[valid]
        data  = data[valid]

        for t in valid:
            ticker_to_sector[t] = sector

        all_returns.append(rets)
        all_prices.append(data)

    daily_rets   = pd.concat(all_returns, axis=1)
    daily_prices = pd.concat(all_prices,  axis=1)
    daily_rets.index   = pd.to_datetime(daily_rets.index)
    daily_prices.index = pd.to_datetime(daily_prices.index)

    # Gem
    daily_rets.to_parquet(DAILY_RETS_PATH)
    daily_prices.to_parquet(DAILY_PRICES_PATH)
    pd.DataFrame(list(ticker_to_sector.items()),
                 columns=["ticker", "sector"]).to_csv(SECTOR_PATH, index=False)

    print(f"\nData gemt:")
    print(f"  {DAILY_RETS_PATH}   ({daily_rets.shape})")
    print(f"  {DAILY_PRICES_PATH} ({daily_prices.shape})")
    print(f"  {SECTOR_PATH}       ({len(ticker_to_sector)} aktier)")
    print(f"\nPeriode: {daily_rets.index[0].date()} → {daily_rets.index[-1].date()}")


if __name__ == "__main__":
    fetch_and_save()
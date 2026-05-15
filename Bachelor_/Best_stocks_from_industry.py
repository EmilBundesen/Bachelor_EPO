import yfinance as yf
import pandas as pd


sectors = {
    "agric": ["ADM", "AGCO", "AGRO", "ANDE", "BG", "CAG", "CALM", "CF", "CPB", "CVGW", "FDP"],
    "autos": ["ALV", "APTV", "BWA", "F", "GM", "GT", "HMC", "LEA", "MGA", "PCAR", "STLA", "TEN","TSLA", "WKHS"],
    "bussv": ["ACN", "ADP", "BABA", "BNBX", "BR", "CTAS", "FISV", "IQV", "MA", "MELI", "PAYX", "PYPL"],
    "chips": ["ADI", "AMAT", "AMD", "AVGO", "INTC", "MU", "NVDA", "TSM", "TXN"],
    "drugs": ["ABBV", "ALNY", "AMGN", "AZN", "BIIB", "BMY", "GILD", "INCY", "JNJ", "LLY", "MRK", "NVO"],
    "food":  ["ABEV", "CCEP", "GIS", "HSY", "KHC", "MDLZ"],
    "guns":  ["AXON", "BA", "DRS", "GD", "HII", "KTOS", "LMT", "NOC", "NPK", "OLN", "RGR", "RTX"],
    "insur": ["AJG", "AON", "CB", "CI", "ELV", "HUM", "MRSH", "PGR", "UNH"],
    "oil":   ["BP", "COP", "CVX", "ENB", "EOG", "EQNR", "OXY", "SHEL", "TTE", "XOM"],
    "rlest": ["AMT", "BN", "CCI", "EQIX", "O", "PLD", "PSA", "SPG", "WELL", "WHLR"],
    "rtail": ["AMZN", "COST", "CVS", "HD", "LOW", "TGT", "TJX", "WMT"],
    "rubbr": ["ATR", "CROX", "CSL", "DECK", "ENTG", "NKE", "NWL", "WMS"],
    "ships": ["BIP", "CCL", "CUK", "DSX", "FRO", "GLNG", "MATX", "NCLH", "NMM", "RCL", "SBLK", "STNG"],
    "softw": ["ADBE", "CDNS", "CRM", "GOOGL", "MANH", "META", "MSFT", "NOW", "ORCL", "PTC", "SAP", "SNPS", "WDAY", "XTIA"],
    "telcm": ["AMX", "BCE", "CHT", "CHTR", "CMCSA", "T", "TMUS", "VZ"],
    "toys":  ["CHDN", "DIS", "HAS", "LYV", "MAT", "MTN", "PLNT"],
    "txtls": ["COLM", "GIL", "LULU", "MHK", "RL", "UAA", "VFC"],
    "util":  ["AEP", "D", "DUK", "EPD", "EXC", "NEE", "NGG", "SO", "SRE", "WM"],
    "soda":  ["CELH", "COKE", "FIZZ", "KDP", "KO", "KOF", "MNST", "PEP"],
    "beer":  ["BF-B", "BUD", "CCU", "DEO", "FMX", "HEINY", "MGPI", "SAM", "STZ", "TAP"],
}

START_DATE = "2010-01-01"
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
        valid = rets.columns[rets.isna().mean() <= 0.35].tolist()
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
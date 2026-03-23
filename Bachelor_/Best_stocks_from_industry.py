import yfinance as yf
import pandas as pd

"""
# ── Konfiguration - 60 måneders risikovindue med 12 måneders signal
sectors = {
    "aero":   ["GE", "RTX", "BA", "LMT", "NOC", "GD", "HWM", "TDG", "LHX", "AXON"],
    "gold":   ["NEM", "AU", "RGLD", "CDE", "AUGO", "HYMC", "SA", "IDR", "CMCL", "CTGO"],
    "drugs":  ["LLY", "JNJ", "ABBV", "MRK", "AMGN", "GILD", "PFE", "BMY", "BIIB", "OGN"],
    "smoke":  ["PM", "MO", "TPB", "UVV", "RYM"],
    "soda":   ["KO", "PEP", "MNST", "KDP", "COKE", "CELH", "PRMB", "FIZZ", "COCO", "BUDA"],
    "beer":   ["STZ", "TAP", "SAM", "BF-B", "MGPI", "AGCC"],
    "food":   ["KHC", "GIS", "JBS", "MKC", "HRL", "SJM", "SFD", "DAR", "PPC", "CAG"],
    "paper":  ["TT", "JCI", "CARR", "LII", "CSL", "MAS", "WMS", "SPXC", "BLDR", "OC"],
    "hshld":  ["PG", "EL", "CL", "WHR", "HOG", "SWK", "NWL", "HELE", "SIG", "CPRI"],
    "insur":  ["CB", "PGR", "TRV", "ALL", "WRB", "CINF", "MKL", "L", "CNA", "AIZ"]
}

# ── Konfiguration - 4 måneders risikovindue med 12 måneders signal
sectors = {
    "fabpr": ["HON", "ETN", "PH", "EMR", "DOV", "ROP", "AME", "IR", "XYL", "GNRC"],
    "banks": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "COF"],
    "steel": ["NUE", "STLD", "RS", "CMC", "ATI", "TX", "CLF", "WOR", "ZEUS", "MTX"],
    "mines": ["FCX", "MP", "ALB", "SCCO", "NEM", "AA", "CENX", "HL", "LTHM", "UUUU"],
    "hlth": ["UNH", "ELV", "CVS", "HCA", "CI", "MCK", "CAH", "MOH", "CNC", "HUM"],
    "cnstr": ["CAT", "DE", "VMC", "MLM", "NVR", "DHI", "LEN", "PHM", "TOL", "FND"],
    "whlsl": ["COST", "WMT", "HD", "LOW", "SYY", "GPC", "FAST", "POOL", "MSC", "BECN"],
    "bussv": ["ADP", "FI", "PAYX", "CTAS", "BR", "VRSK", "DNB", "RHI", "MAN", "KELYA"],
    "rubbr": ["GT", "COE", "TREX", "AZEK", "BCPC", "ARCH", "VC", "SXC", "APOG", "ASTE"],
    "paper": ["IP", "PKG", "SEE", "SON", "SLVM", "CLW", "SVES", "BERY", "GPK", "ATR"],
    "trans": ["UPS", "FDX", "CSX", "UNP", "NSC", "JBHT", "ODFL", "SAIA", "XPO", "CHRW"],
    "util": ["NEE", "SO", "DUK", "AEP", "SRE", "D", "EXC", "PCG", "ED", "XEL"]
}
"""

# ── Konfiguration - 24 måneders risikovindue med 12 måneders signal
sectors = {
  "smoke":  ["PM", "MO", "BTI", "UVV", "TPB", "RLX", "XXII"],
  "other":  ["BRK-B", "HON", "GE", "RTX", "CAT", "DE", "ITW", "DOV", "EMR", "MMM"],
  "util":   ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "PEG", "ED"],
  "chips":  ["NVDA", "TSM", "AVGO", "ASML", "AMD", "INTC", "QCOM", "TXN", "MU", "ADI"],
  "guns":   ["SWBI", "RGR", "OLN", "POWW", "SPWH", "VSTO", "DKS", "ASO"],
  "rtail":  ["AMZN", "WMT", "COST", "HD", "LOW", "TGT", "TJX", "ROST", "DG", "DLTR"],
  "hardw":  ["AAPL", "DELL", "HPQ", "NTAP", "STX", "WDC", "ANET", "CSCO", "HPE", "SMCI"],
  "fun":    ["DIS", "NFLX", "SONY", "EA", "TTWO", "RBLX", "LYV", "WBD", "FOX"],
  "fin":    ["V", "MA", "PYPL", "AXP", "COF", "SYF", "FIS", "FISV", "GPN"],
  "banks":  ["JPM", "BAC", "WFC", "C", "HSBC", "HDB", "RY", "TD", "UBS", "SAN"],
  "beer":   ["BUD", "DEO", "STZ", "TAP", "SAM", "CCU"],
  "food":   ["NSRGY", "MDLZ", "KHC", "GIS", "HSY", "SJM", "HRL", "CAG", "CPB"],
  "soda":   ["KO", "PEP", "KDP", "MNST", "CELH", "COKE", "FIZZ"],
  "drugs":  ["LLY", "JNJ", "NVO", "MRK", "ABBV", "PFE", "AZN", "BMY", "GILD", "AMGN"],
  "hlth":   ["UNH", "ELV", "CI", "CVS", "HUM", "HCA", "UHS", "THC", "MOH"],
  "chems":  ["LIN", "SHW", "APD", "ECL", "DD", "DOW", "PPG", "LYB", "IFF", "FMC"],
  "oil":    ["XOM", "CVX", "SHEL", "TTE", "BP", "COP", "EOG", "SLB", "PBR", "ENB"],
  "clths":  ["NKE", "ADDYY", "LULU", "PVH", "VFC", "COLM"],
  "rubbr":  ["GT", "TWI", "MYE"],
  "steel":  ["NUE", "STLD", "PKX", "MT", "VALE", "SID", "GGB", "TX", "CMC"]
}


START_DATE = "2015-01-01"
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
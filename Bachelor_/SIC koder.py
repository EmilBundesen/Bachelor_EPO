import os
import pandas as pd
import yfinance as yf
import requests
from tqdm import tqdm

# ── Konfiguration ─────────────────────────────────────────────
SIC_CACHE_PATH  = "sic_cache.csv"
MARKET_CAP_DATE = "2022-12-31"
N_PER_SECTOR    = 15
MIN_MARKET_CAP  = 500e6

# ── 1. Ønskede sektorer (rediger kun her) ─────────────────────
WANTED_SECTORS = [
    # Long
    "Food", "Guns", "Agric", "Drugs", "Beer", "Soda",
    "Insur", "Oil", "Ships", "Util", "Smoke", "Aero",
    "Coal", "Whlsl",
    # Short
    "Softw", "Toys", "Rubbr", "Rtail", "BusSv", "Telcm",
    "Autos", "Txtls", "RlEst", "Chips", "ElcEq", "Paper",
    "Fun", "Hlth", "MedEq",
]

# ── FF49 SIC-koder ────────────────────────────────────────────
FF49_SIC = {
    "Food":  list(range(2000, 2100)),
    "Guns":  [3760,3761,3762,3763,3764,3765,3766,3767,3768,3769,
              3795,3489,3482,3483,3484,3485,3486,3487,3488],
    "Agric": list(range(100, 1000)),
    "Drugs": list(range(2830, 2837)),
    "Beer":  [2082, 2083, 2084, 2085],
    "Soda":  [2080, 2081, 2086],
    "Insur": list(range(6311, 6412)),
    "Oil":   list(range(1311, 1382)) + list(range(2911, 2913)) + [4610, 4611],
    "Ships": list(range(4400, 4500)) + list(range(3730, 3740)),
    "Util":  list(range(4900, 4942)) + list(range(4950, 4992)),
    "Smoke": list(range(2100, 2112)),
    "Aero":  list(range(3720, 3730)) + [3812],
    "Coal":  list(range(1200, 1300)),
    "Whlsl": list(range(5000, 5200)),
    "Softw": list(range(7370, 7380)),
    "Toys":  list(range(3940, 3950)) + list(range(7900, 7999)),
    "Rubbr": list(range(3000, 3100)),
    "Rtail": list(range(5200, 5600)) + list(range(5600, 5700))
             + list(range(5900, 5963)),
    "BusSv": list(range(7380, 7395)) + list(range(8700, 8750)),
    "Telcm": list(range(4800, 4900)),
    "Autos": list(range(3710, 3717)) + list(range(5510, 5560)),
    "Txtls": list(range(2200, 2300)) + list(range(2300, 2400)),
    "RlEst": list(range(6500, 6553)) + [6798],
    "Chips": [3674],
    "ElcEq": list(range(3600, 3700)),
    "Paper": list(range(2620, 2660)) + list(range(2670, 2680)),
    "Fun":   list(range(7800, 7834)) + list(range(7929, 7931)) + [4833],
    "Hlth":  list(range(8000, 8100)),
    "MedEq": list(range(3840, 3860)) + [5047],
}

# Flad SIC → FF49 lookup (første match vinder)
SIC_LOOKUP: dict[int, str] = {}
for ff, sics in FF49_SIC.items():
    for s in sics:
        if s not in SIC_LOOKUP:
            SIC_LOOKUP[s] = ff

def sic_to_ff49(sic) -> str | None:
    if pd.isna(sic):
        return None
    return SIC_LOOKUP.get(int(sic))


# ── 2. Hent tickers + SIC fra EDGAR (med cache) ───────────────

def fetch_sec_universe() -> pd.DataFrame:
    if os.path.exists(SIC_CACHE_PATH):
        df = pd.read_csv(SIC_CACHE_PATH)
        print(f"Cache indlæst: {len(df)} virksomheder")
    else:
        headers = {"User-Agent": "research@example.com"}

        print("Henter tickers fra SEC EDGAR...")
        r = requests.get(
            "https://www.sec.gov/files/company_tickers_exchange.json",
            headers=headers
        )
        raw = r.json()
        df  = pd.DataFrame(raw["data"], columns=raw["fields"])
        df.columns = df.columns.str.lower()
        df  = df[df["exchange"].isin(["NYSE", "Nasdaq"])]
        df["ticker"] = df["ticker"].str.upper().str.strip()
        df  = df[df["ticker"].str.match(r'^[A-Z]{1,5}$')]
        print(f"  {len(df)} tickers på NYSE/Nasdaq")

        print("Henter SIC-koder (gemmes i cache)...")
        sic_map = {}
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                s = requests.get(
                    f"https://data.sec.gov/submissions/"
                    f"CIK{str(row['cik']).zfill(10)}.json",
                    headers=headers, timeout=8
                ).json().get("sic")
                sic_map[row["ticker"]] = s
            except Exception:
                sic_map[row["ticker"]] = None

        df["sic"] = df["ticker"].map(sic_map)
        df.to_csv(SIC_CACHE_PATH, index=False)
        print(f"Cache gemt: {SIC_CACHE_PATH}")

    # Map SIC → FF49 og filtrer til WANTED_SECTORS
    df["sic"]  = pd.to_numeric(df["sic"], errors="coerce")
    df["ff49"] = df["sic"].apply(sic_to_ff49)
    df = df[df["ff49"].isin(WANTED_SECTORS)].copy()
    df = df.drop_duplicates(subset=["ticker"])

    # Verificer dækning
    found   = set(df["ff49"].unique())
    missing = set(WANTED_SECTORS) - found
    if missing:
        print(f"  ADVARSEL: Ingen tickers for: {missing}")

    print(f"Efter filter: {len(df)} tickers "
          f"i {df['ff49'].nunique()}/{len(WANTED_SECTORS)} sektorer")
    return df.reset_index(drop=True)


# ── 3. Hent historisk market cap pr. 2022-12-31 ───────────────

def fetch_historical_market_caps(tickers: list[str]) -> pd.Series:
    mc  = {}
    end = pd.Timestamp(MARKET_CAP_DATE) + pd.Timedelta(days=7)
    print(f"Henter market cap pr. {MARKET_CAP_DATE} "
          f"for {len(tickers)} tickers...")

    for t in tqdm(tickers):
        try:
            ticker = yf.Ticker(t)

            shares = ticker.fast_info.get("shares")
            if not shares or shares <= 0:
                shares = ticker.info.get("sharesOutstanding")
            if not shares or shares <= 0:
                continue

            hist = ticker.history(
                start=MARKET_CAP_DATE,
                end=end,
                auto_adjust=True
            )
            if hist.empty:
                continue

            price = hist["Close"].iloc[0]
            if price > 0:
                mc[t] = price * shares

        except Exception:
            pass

    return pd.Series(mc, name="market_cap")


# ── 4. Saml og vælg top-N ─────────────────────────────────────

def build_universe(n: int = N_PER_SECTOR) -> dict:
    # Hent tickers filtreret til WANTED_SECTORS
    df = fetch_sec_universe()

    # Historisk market cap
    mc = fetch_historical_market_caps(df["ticker"].tolist())
    df["market_cap"] = df["ticker"].map(mc)
    df = df.dropna(subset=["market_cap"])
    df = df[df["market_cap"] >= MIN_MARKET_CAP]
    df = df.sort_values(["ff49", "market_cap"], ascending=[True, False])

    # Top-N per sektor
    top = df.groupby("ff49").head(n)

    # Print oversigt
    print("\n" + "=" * 62)
    print(f"TOP {n} PER SEKTOR  —  market cap pr. {MARKET_CAP_DATE}")
    print("=" * 62)
    print(f"  {'Sektor':<8} {'N':>4}  {'Top-1':<8} {'Mkt Cap':>10}")
    print("  " + "-" * 50)
    for ff, grp in top.groupby("ff49"):
        t1 = grp.iloc[0]
        print(f"  {ff:<8} {len(grp):>4}  {t1['ticker']:<8} "
              f"{t1['market_cap']/1e9:>9.1f}B")
    print("=" * 62)

    # Byg sector dict uden dubletter
    seen    = set()
    sectors = {}
    for ff, grp in top.groupby("ff49"):
        sectors[ff.lower()] = []
        for t in grp["ticker"]:
            if t not in seen:
                sectors[ff.lower()].append(t)
                seen.add(t)

    return sectors


if __name__ == "__main__":
    sectors = build_universe(n=N_PER_SECTOR)

    print("\nsectors = {")
    for k, v in sectors.items():
        print(f'    "{k}": {v},')
    print("}")
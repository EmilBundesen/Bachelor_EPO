# Bachelor — EPO

Implementering af **Enhanced Portfolio Optimization (EPO)** fra Pedersen, med flere (2021), anvendt på industri- og aktieporteføljer med cross-sectional momentum (XSMOM) og time-series momentum (TSMOM) som signaler.

---

## Projektstruktur og rollefordeling

Koden er organiseret sådan, at **`Equity_1.py` udgør kernen** — alle centrale EPO-funktioner er implementeret her og genbruges i de øvrige filer. De andre scripts udvider, visualiserer eller anvender disse funktioner på nye datasæt og konfigurationer.

```
Bachelor_/
│
├── Data/                                  # Oprettes manuelt — se nedenfor
│   ├── 49_Industry_monthly.csv            # Kenneth French — månedlige industrier
│   ├── 49_Industry_Portfolios_Daily.csv   # Kenneth French — daglige industrier
│   └── Månedlig_rf.csv                    # Fama-French risikofri rente
│
├── Equity_1.py                            ← KERNEFIL: EPO-funktioner, XSMOM, risikomodel, backtest
├── equity_genskabning.py                  Robusthedstest: 8 Equity-konfigurationer med varierende risikovindue,
                                           signalvindue, signaltype og optimeringsmetode
├── Stock_Data.py                          Udvider Equity_1 til enkeltaktier (Yahoo Finance): TSMOM-signal, buy-and-hold,
                                           årlig rebalancering, gearing
├── Signal_Visual.py                       Visualiseringer: XSMOM-signal, rullende volatilitet, turnover
├── Best_stocks_from_industry.py           Datahentning: enkeltaktier pr. sektor (Yahoo Finance)
└── SIC koder.py                           Bygger aktieunivers via SEC EDGAR + market cap-filter
```

### Hvad er implementeret i Equity_1.py

`Equity_1.py` indeholder alle byggeklodser til EPO-analysen:

- **`get_monthly_return()` / `get_monthly_risk()`** — dataindlæsning og rensning
- **`compute_xsmom()`** — XSMOM-signal (lign. 24–25): kumuleret afkast over `LOOKBACK_MONTHS`, demeanet og normaliseret til unit-leverage
- **`compute_risk_model()`** — rullende kovariansmatrix med pre-shrinkage mod identitetsmatrix: `Σ_θ = (1−θ)Σ + θI`
- **`epo_weights()`** — EPO-vægte: `x = (1/γ) Σ_w⁻¹ s`, hvor `Σ_w = (1−w)Σ + w·diag(Σ)` (lign. 19–20)
- **`build_epo_panel()`** — beregner EPO-portefølje for alle kandidat-w-værdier
- **`build_dynamic_oos_epo()`** — vælger w dynamisk out-of-sample via rullende Sharpe-maksimering
- **`backtest_indmom()`** — INDMOM-benchmark (signal direkte som vægte, ingen kovariansoptimering)
- **`backtest_equal_weight()`** — 1/N-benchmark
- **`backtest_mvo_no_shrink()`** — MVO uden shrinkage (θ=0, w=0)

De øvrige scripts bygger oven på disse: `equity_genskabning.py` tilføjer TSMOM-signal og forankret EPO (`epo_anchored_weights`); `Stock_Data.py` tilføjer er implementering i aktieuniverset. Dvs. vi introducerer simple momentum long/short, buy-and-hold og årlig rebalancering.

## Konstanter til replikering af artiklens SR i perioden 1942-2018
```python
# ── Tidsperiode ──────────────────────────────────────────────
DATA_START_DATE     = "1926-07-01"   # Hvorfra data indlæses
BACKTEST_START_DATE = "1942-01-01"   # OOS-periodens start
BACKTEST_END_DATE   = "2018-12-31"   # OOS-periodens slut

# ── Risikomodel ──────────────────────────────────────────────
RISK_WINDOW      = 60    # Rullende vindue for kovariansestimering (måneder)
CORR_PRESHRINK   = 0.05  # θ: pre-shrinkage af korrelationer mod identitetsmatrix

# ── Signal ───────────────────────────────────────────────────
LOOKBACK_MONTHS  = 12    # XSMOM

# ── EPO ──────────────────────────────────────────────────────
GAMMA            = 3
CANDIDATE_WS     = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00]
MIN_HISTORY_OOS  = 12
```
---

## Konstanter der skal tilpasses

> **Alle nøgleparametre styres via konstanter øverst i hvert script.** Inden kørsel skal man aktivt tage stilling til hvilken periode, hvilket risikovindue og hvilket signal man ønsker — og justere konstanterne tilsvarende. Dette gælder både `Equity_1.py` og de scripts, der kalder visualiseringer og backtests baseret på samme parametre.

### Konstanter i `Equity_1.py`

```python
# ── Tidsperiode ──────────────────────────────────────────────
DATA_START_DATE     = "2010-01-01"   # Hvorfra data indlæses
BACKTEST_START_DATE = "2020-01-01"   # OOS-periodens start
BACKTEST_END_DATE   = "2025-12-31"   # OOS-periodens slut

# ── Risikomodel ──────────────────────────────────────────────
RISK_WINDOW      = 24    # Rullende vindue for kovariansestimering (måneder)
                         # Equity 1: 60m | Equity 2: 36m | Equity 3+: 24m
CORR_PRESHRINK   = 0.05  # θ: pre-shrinkage af korrelationer mod identitetsmatrix

# ── Signal ───────────────────────────────────────────────────
LOOKBACK_MONTHS  = 12    # XSMOM/TSMOM lookback (måneder)
                         # Equity 3: 12m | Equity 4: 24m | Equity 5: 6m | Equity 6: 3m

# ── EPO ──────────────────────────────────────────────────────
GAMMA            = 3
CANDIDATE_WS     = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00]
MIN_HISTORY_OOS  = 6     # Minimum IS-måneder til dynamisk w-valg
```

Disse konstanter bruges direkte i alle centrale beregninger. Visualiseringsscripts som `Signal_Visual.py` har deres egne tilsvarende konstanter øverst — disse skal matche det ønskede setup.

### Equity-konfigurationer fra opgaven

Opgaven analyserer 8 konfigurationer med varierende risikovindue, signalvindue, signaltype og optimeringsmetode. For at genskabe en specifik konfiguration i `Equity_1.py` justeres konstanterne således:

| Navn | `RISK_WINDOW` | `LOOKBACK_MONTHS` | Signal | Metode |
|------|:---:|:---:|--------|--------|
| Equity 1 | 60 | 12 | XSMOM | Simpel EPO |
| Equity 2 | 36 | 12 | XSMOM | Simpel EPO |
| Equity 3 | 24 | 12 | XSMOM | Simpel EPO |
| Equity 4 | 24 | 24 | XSMOM | Simpel EPO |
| Equity 5 | 24 |  6 | XSMOM | Simpel EPO |
| Equity 6 | 24 |  3 | XSMOM | Simpel EPO |
| Equity 7 | 24 | 12 | TSMOM | Simpel EPO — skift til `compute_tsmom_signal()` |
| Equity 8 | 24 | 12 | XSMOM | Forankret EPO — brug `epo_anchored_weights()` fra `equity_genskabning.py` |

`equity_genskabning.py` kører alle 8 automatisk via en `EQUITY_CONFIGS`-liste og kræver ingen manuelle konstantændringer pr. konfiguration.

### Filstier skal opdateres

Alle scripts har hardkodede absolutte stier. Søg og erstat `/Users/emilbundesen/Desktop/Bachelor/` med din egen rodmappe:

```python
# Eksempel i Equity_1.py
df = pd.read_csv(
    "/din/sti/Data/49_Industry_monthly.csv",   # <-- tilpas
    sep=",", header=6
)
```

Scripts med hardkodede stier: `Equity_1.py`, `equity_genskabning.py`, `Sammensat.py`, `rf_monthly.py`, `Signal_Visual.py`.

---

## Opsætning

### Krav

```bash
pip install numpy pandas matplotlib seaborn scipy yfinance tqdm requests
```

Python 3.10+ anbefales.

### Data

Hent følgende filer fra [Kenneth French's hjemmeside](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html):

| Fil | Indhold | Header-linje |
|-----|---------|:---:|
| `49_Industry_monthly.csv` | Månedlige value-weighted industri-afkast | 6 |
| `49_Industry_Portfolios_Daily.csv` | Daglige afkast, 49 industrier | 5 |
| `Månedlig_rf.csv` | Fama-French faktorer — RF-kolonnen bruges | skip 3 |

---

## Kørsel

### Primær analyse — industrier

```bash
python Equity_1.py
```

Udfører komplet backtest og printer performance-tabel, EPO-panel for alle `CANDIDATE_WS`, dynamisk OOS EPO og porteføljevægte. Tilpas `BACKTEST_START_DATE`, `RISK_WINDOW` og `LOOKBACK_MONTHS` til den ønskede konfiguration inden kørsel.

### Robusthedsanalyse — alle 8 Equity-konfigurationer

```bash
python equity_genskabning.py
```

Kører samtlige konfigurationer automatisk og producerer plots af rullende IC og gennemsnitlig korrelation pr. konfiguration. Ingen manuelle konstantændringer nødvendige.

### Visualiseringer

```bash
python Signal_Visual.py
```

Producerer tre plots:

1. Rullende annualiseret volatilitet for de N mest volatile industrier
2. XSMOM-signalværdier over tid
3. Rullende månedlig turnover: MVO vs. EPO med udvalgte w-værdier

Konstanterne øverst i `Signal_Visual.py` (`RISK_WINDOW`, `LOOKBACK_MONTHS`, `BACKTEST_START_DATE`, outputstier) skal matche den konfiguration, man ønsker at visualisere. Plots gemmes som PNG-filer på den hardkodede sti — tilpas den inden kørsel.

### Enkeltaktier — Yahoo Finance-univers

**Trin 1:** Hent aktiedata (køres kun én gang eller ved dataopdatering):

```bash
python Best_stocks_from_industry.py
```

Gemmer tre filer i arbejdsmappen:
```
data_daily_returns.parquet
data_daily_prices.parquet
data_ticker_sector.csv
```

Sektor-universet defineres i `sectors`-dict'en øverst i filen. Filen indeholder flere kommenterede konfigurationer svarende til forskellige risikovinduer og perioder fra opgaven — vælg den relevante og fjern kommentarerne.

**Trin 2:** Kør backtest:

```bash
python Stock_Data.py
```

Ud over de sædvanlige EPO-strategier implementerer `Stock_Data.py`:

- **TSMOM** (sektor-neutraliseret time-series momentum) som alternativt signal
- **Simple momentum** (long top-N / short bottom-N) uden porteføljeoptimering
- **Buy-and-hold** — vægte fryses ved `signal_date` og holdes uændret hele perioden
- **Månedlig / årlig rebalancering** — direkte sammenligning af rebalanceringsfrekvenser
- **Leverage-skalering** — normaliserer strategier til samme brutto-eksponering (gross exposure)

Konstanter for periode, signal og risikovindue øverst i `Stock_Data.py` skal matche de valg, der er truffet i `Best_stocks_from_industry.py`.

### Byg aktieunivers via SEC EDGAR (valgfrit)

```bash
python "SIC koder.py"
```

Henter SIC-koder fra SEC EDGAR, mapper til FF49-sektorer og vælger de N aktier pr. sektor med størst market cap pr. `MARKET_CAP_DATE`. Output er en Python-dict klar til indsætning i `Best_stocks_from_industry.py`. Resultater caches i `sec_universe_cache.csv` for at undgå gentagne API-kald.

---

## Metodeoversigt

| Komponent | Formel | Styres af |
|-----------|--------|-----------|
| XSMOM | Kumuleret afkast, demeanet, unit-leverage normaliseret (lign. 24–25) | `LOOKBACK_MONTHS` |
| TSMOM | Sektor-neutraliseret time-series momentum | `LOOKBACK_MONTHS` |
| Risikomodel | `Σ_θ = (1−θ)Σ + θI` — rullende kovarians med pre-shrinkage | `RISK_WINDOW`, `CORR_PRESHRINK` |
| Simpel EPO | `x = (1/γ) Σ_w⁻¹ s`, `Σ_w = (1−w)Σ + w·diag(Σ)` (lign. 19–20) | `GAMMA`, `CANDIDATE_WS` |
| Forankret EPO | `EPO_a(w) = Σ_w⁻¹ [(1−w)κs + wVa]` — 1/N som anker (lign. 27–30) | `CANDIDATE_WS` |
| Dynamisk w | OOS Sharpe-maksimering over `CANDIDATE_WS` | `MIN_HISTORY_OOS` |

---

## Kendte begrænsninger

- **Hardkodede stier** — skal opdateres lokalt i alle scripts.
- **Look-ahead bias** er aktivt undgået: signal og risikomodel bruger udelukkende data fra *inden* porteføljemåneden. Verificér tidsflowet med `visualize_date_flow_simple()` i `Stock_Data.py`.
- **Transaktionsomkostninger** er ikke modelleret — rapporterede Sharpe-ratioer er brutto.
- `SIC koder.py` kan fejle ved SEC EDGAR rate-limiting — brug den cachede CSV ved gentagende kørsel.
- Yahoo Finance-data filtreres automatisk: tickers med mere end 20% (industrier) eller 35% (enkeltaktier) manglende data frasorteres.

---

## Referencer

Pedersen, L. H., Babu, A., & Levine, A. (2021). *Enhanced Portfolio Optimization*. Financial Analysts Journal, 77(2), 124–151.

Kenneth French Data Library: [https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

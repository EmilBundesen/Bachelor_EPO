# Bachelorprojekt — Enhanced Portfolio Optimisation (EPO)

Implementering af **Enhanced Portfolio Optimisation (EPO)** metoden fra Pedersen, Babu & Levine (2021), anvendt på Fama-French 49 industrier og individuelle aktier fra Yahoo Finance.

---

## Filstruktur

| Fil | Beskrivelse |
|-----|-------------|
| `Equity_1.py` | Kerne-EPO implementering på Fama-French 49 industri-data |
| `equity_genskabning.py` | Kører Equity 1–8 robusthedskonfigurationer (varierende risikovindue, signal og anker) |
| `moment_mod_EPO_ind.py` | EPO vs. TSMOM Vol vs. TSMOM EW over tre delperioder (FF49-data) |
| `Signal_Visual.py` | Rullende volatilitet og XS-momentum signal-plots (FF49-data) |
| `SIC_koder.py` | Universkonstruktion — finder top-N aktier per sektor via markedskapitalisering og SEC EDGAR |
| `Best_stocks_from_industry.py` | Henter aktiedata fra Yahoo Finance for det valgte univers og gemmer lokalt |
| `Stock_Data.py` | EPO-backtest på enkeltaktier inkl. transaktionsomkostningsanalyse |
| `Moment_mod_EPO_stock.py` | EPO vs. TSMOM på Yahoo Finance-aktier (2020–2025) |
| `Investeringsomkostninger.py` | Turnover- og netto-afkastberegninger (importeres af Stock_Data.py) |

---

## Datakrav

### Fama-French data (bruges af Equity_1, equity_genskabning, moment_mod_EPO_ind, Signal_Visual)

Download fra [Kenneth Frenchs datalibrary](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) og placer filerne på følgende stier:

```
/Users/<dit navn>/Desktop/Bachelor/Data/49_Industry_monthly.csv   # 49 Industry Portfolios (månedlige, value-weighted)
/Users/<dit navn>/Desktop/Bachelor/Data/Månedlig_rf.csv           # Fama-French faktorer (månedlig risikofri rente)
```

> Stierne er hardcoded i `Equity_1.py` og `Signal_Visual.py`. Opdatér dem hvis din mappestruktur er anderledes.

### Yahoo Finance aktiedata (bruges af Stock_Data, Moment_mod_EPO_stock)

Genereres lokalt ved at køre `Best_stocks_from_industry.py` (se trin 2 nedenfor). Ingen manuel download nødvendig.

---

## Sådan køres koden

### Trin 1 — Installér afhængigheder

```bash
pip install yfinance pandas numpy matplotlib seaborn requests tqdm pyarrow
```

### Trin 2 — Hent aktiedata fra Yahoo Finance

Skal kun køres én gang (eller når data skal opdateres).

```bash
python Best_stocks_from_industry.py
```

Henter daglige priser og afkast for det 20-sektor aktieuniverset (2010–2025) og gemmer:
- `data_daily_returns.parquet`
- `data_daily_prices.parquet`
- `data_ticker_sector.csv`

> Aktieuniverset er fastlagt ved hjælp af `SIC_koder.py`, som identificerer de 15 største selskaber (markedskapitalisering pr. 31-12-2022) i hver af de 29 Fama-French sektorer via SEC EDGAR. De resulterende tickers er hardcoded i `Best_stocks_from_industry.py`.

### Trin 3 — Kør EPO-analysen på industriniveau (FF49-data)

```bash
python Equity_1.py
```

Udskriver en performancetabel (Sharpe-ratio) for EPO over alle shrinkage-parametre `w ∈ {0%, 10%, 25%, 50%, 75%, 90%, 99%, 100%}` samt INDMOM, MVO og 1/N benchmarks for OOS-perioden.

### Trin 4 — Kør robusthedstjek (Equity 1–8)

```bash
python equity_genskabning.py
```

Kører otte konfigurationer med varierende risikovindue (24m/36m/60m), signal-lookback (3m/6m/12m/24m), signaltype (XSMOM/TSMOM) og porteføljeanker. Udskriver en samlet Sharpe-ratio tabel.

### Trin 5 — EPO vs. TSMOM dekomponering (FF49-data)

```bash
python moment_mod_EPO_ind.py
```

Sammenligner EPO w=0.75, TSMOM Vol-skaleret og TSMOM Equal-Weighted med samme signal. Isolerer bidraget fra korrelationsshrinkage. Gemmer plottene `epo_vs_tsmom_full.png` og `epo_vs_tsmom_subperiods.png`.

### Trin 6 — EPO-backtest på enkeltaktier

```bash
python Stock_Data.py
```

Kører det fulde EPO-backtest på enkeltaktier (OOS 2020–2025), inklusiv:
- Sammenligning af månedlig og årlig rebalancering
- Transaktionsomkostningsfølsomhedsanalyse
- Leverage-justeret performancetabel
- Gemmer `performance_summary_2020_2025.csv`

### Trin 7 — EPO vs. TSMOM på aktier (2020–2025)

```bash
python Moment_mod_EPO_stock.py
```

Sammenligner EPO w=0.75 mod TSMOM Vol-skaleret, TSMOM EW og TSMOM Long/Short på Yahoo Finance-aktieuniverset. Gemmer plottet `epo_vs_tsmom_yf_2023_2025.png`.

### Trin 8 — Signalvisualisering (valgfrit)

```bash
python Signal_Visual.py
```

Plotter rullende volatilitet og XS-momentum signal for de 10 mest volatile FF49 industrier.

---

## Universkonstruktion (reference)

Aktieuniverset er konstrueret med `SIC_koder.py`:

```bash
python SIC_koder.py
```

Scriptet forespørger SEC EDGAR for alle NYSE/Nasdaq-noterede selskaber, mapper dem til Fama-French 49 sektorer via SIC-koder og vælger de 15 største målt på markedskapitalisering pr. 31-12-2022 per sektor. Outputtet printes som en Python-dict og er hardcoded ind i `Best_stocks_from_industry.py`. Dette trin behøver ikke køres igen, medmindre universet skal revideres.

---

## Metodiske noter

- Alle backtests bruger **merafkast** (råafkast minus månedlig risikofri rente).
- EPO-signalet er **XSMOM** (cross-sectional momentum, lign. 24–25 i Pedersen et al.) for industrianalysen og **TSMOM** (time-series momentum) for aktieanalysen.
- Risikomodellen bruger en rullende kovariansestimator med 5% pre-shrinkage af korrelationsmatricen mod identitetsmatricen (θ = 0,05).
- Shrinkage-parameteren `w` blander den fulde kovariansmatrix mod dens diagonal: `w = 0` svarer til standard MVO, `w = 1` svarer til vol-skaleret (INDMOM-anker).
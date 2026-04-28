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


# ── Konfiguration - 24 måneders risikovindue med 12 måneders signal baseret på EPO vægte ultimo 2024
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


# ── Konfiguration - 24 måneders risikovindue med 12 måneders XSMOM signal baseret på EPO vægte gennemsnit 2022
sectors = {
  "smoke":  ["PM", "MO", "BTI", "UVV", "TPB", "RLX", "XXII"],
  "other":  ["BRK-B", "HON", "GE", "RTX", "CAT", "DE", "ITW", "DOV", "EMR", "MMM"],
  "util":   ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "PEG", "ED", "EIX", "AWK", "CNP", "WEC"],
  "chips":  ["NVDA", "TSM", "AVGO", "ASML", "AMD", "INTC", "QCOM", "TXN", "MU", "ADI"],
  "guns":   ["SWBI", "RGR", "OLN", "POWW", "SPWH", "DKS", "ASO", "AXON", "NPK", "AOUT"],
  "rtail":  ["AMZN", "WMT", "COST", "HD", "LOW", "TGT", "TJX", "ROST", "DG", "DLTR"],
  "hardw":  ["AAPL", "DELL", "HPQ", "NTAP", "STX", "WDC", "ANET", "CSCO", "HPE", "SMCI"],
  "banks":  ["JPM", "BAC", "WFC", "C", "HSBC", "HDB", "RY", "TD", "UBS", "SAN"],
  "beer":   ["BUD", "DEO", "STZ", "TAP", "SAM", "CCU", "HEINY", "FMX", "BF-B"],
  "food":   ["NSRGY", "MDLZ", "KHC", "GIS", "HSY", "SJM", "HRL", "CAG", "CPB", "TSN", "ADM", "BG"],
  "soda":   ["KO", "PEP", "KDP", "MNST", "CELH", "COKE", "FIZZ", "KOF", "PRMB", "COCO", "ZVIA"],
  "drugs":  ["LLY", "JNJ", "NVO", "MRK", "ABBV", "PFE", "AZN", "BMY", "GILD", "AMGN"],
  "hlth":   ["UNH", "ELV", "CI", "CVS", "HUM", "HCA", "UHS", "THC", "MOH"],
  "chems":  ["LIN", "SHW", "APD", "ECL", "DD", "DOW", "PPG", "LYB", "IFF", "FMC"],
  "oil":    ["XOM", "CVX", "SHEL", "TTE", "BP", "COP", "EOG", "SLB", "PBR", "ENB", "OXY", "DVN", "MPC", "PSX", "VLO", "FANG"],
  "clths":  ["NKE", "ADDYY", "LULU", "PVH", "VFC", "COLM"],
  "rubbr":  ["GT", "TWI", "MYE"],
  "steel":  ["NUE", "STLD", "PKX", "MT", "VALE", "SID", "GGB", "TX", "CMC"],
  "agric":  ["CTVA", "NTR", "MOS", "CF", "AGCO", "CALM", "ANDE"],
  "insur":  ["MET", "PRU", "AFL", "ALL", "PGR", "TRV", "AIG", "KNSL"],
  "coal":   ["BTU", "AMR", "ARLP", "HCC", "METC", "NRP", "SXC", "NC"],
  "whlsl":  ["SYY", "GWW", "WCC", "MSM", "AIT", "WSO", "USFD", "UNFI", "FERG", "CNM", "PFGC"],
}

# ── Konfiguration - 24 måneders risikovindue med 12 måneders XSMOM signal baseret på EPO vægte ultimo 2022
sectors = {
    # ── LONG sektorer (behold/udvid) ──────────────────────────
    "food":   ["NSRGY", "MDLZ", "KHC", "GIS", "HSY", "SJM", "HRL", "CAG", "CPB",
               "TSN", "ADM", "BG", "MKC", "K", "POST", "THS", "LANC", "INGR"],
    # Guns (Long 2, +9.31)
    "guns":   ["SWBI", "RGR", "OLN", "POWW", "SPWH", "DKS", "ASO", "AXON", "NPK",
               "AOUT", "VSTO", "KTOS", "HEI"],
    # Agric (Long 3, +8.61)
    "agric":  ["CTVA", "NTR", "MOS", "CF", "AGCO", "CALM", "ANDE", "FMC",
               "MBII", "LSCC", "FLWS", "VITL"],
    # Drugs (Long 4, +7.66)
    "drugs":  ["LLY", "JNJ", "NVO", "MRK", "ABBV", "PFE", "AZN", "BMY",
               "GILD", "AMGN", "REGN", "VRTX", "BIIB", "ALNY", "JAZZ"],
    # Beer (Long 5, +7.10)
    "beer":   ["BUD", "DEO", "STZ", "TAP", "SAM", "CCU", "HEINY", "FMX",
               "BF-B", "MGPI", "EAST", "CBL"],
    # Soda (Long 6, +6.97)
    "soda":   ["KO", "PEP", "KDP", "MNST", "CELH", "COKE", "FIZZ", "KOF",
               "COCO", "ZVIA", "REED"],
    # Insur (Long 7, +6.89)
    "insur":  ["MET", "PRU", "AFL", "ALL", "PGR", "TRV", "AIG", "KNSL",
               "CB", "HIG", "MKL", "RNR", "ERIE", "WRB", "CINF", "UFG"],
    # Oil (Long 8, +6.87)
    "oil":    ["XOM", "CVX", "SHEL", "TTE", "BP", "COP", "EOG", "SLB",
               "PBR", "ENB", "OXY", "DVN", "MPC", "PSX", "VLO", "FANG",
               "HES", "HAL", "BKR", "NOV"],
    # Ships (Long 9, +6.51) — ny sektor
    "ships":  ["ZIM", "MATX", "SBLK", "GOGL", "NMM", "ESEA", "SALT",
               "GNK", "EGLE", "TOPS", "CTRM", "SHIP", "DSX"],

    # Util (Long 10, +5.11)
    "util":   ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
               "PEG", "ED", "EIX", "AWK", "CNP", "WEC", "ETR", "FE",
               "PPL", "NI", "CMS", "LNT"],

    # ── SHORT sektorer (behold men trim) ──────────────────────

    # Softw (Short 1, -7.86)
    "softw":  ["MSFT", "ORCL", "SAP", "ADBE", "CRM", "NOW", "INTU",
               "WDAY", "SNOW", "DDOG"],

    # Toys (Short 2, -7.35) — ny sektor
    "toys":   ["HAS", "MAT", "FNKO", "JAKK", "LEGH", "PLBY", "GHC"],

    # Rubbr (Short 3, -5.62)
    "rubbr":  ["GT", "TWI", "MYE", "TREX", "NGVT"],

    # Rtail (Short 4, -5.36)
    "rtail":  ["AMZN", "WMT", "COST", "HD", "LOW", "TGT", "TJX",
               "ROST", "DG", "DLTR"],

    # BusSv (Short 5, -4.30) — ny sektor
    "bussv":  ["ADP", "PAYX", "CDAY", "RHI", "MAN", "KELYA", "KFRC",
               "TBI", "HCSG", "PFMT"],

    # Telcm (Short 6, -3.87)
    "telcm":  ["T", "VZ", "TMUS", "CMCSA", "CHTR", "LUMN", "FYBR",
               "CABO", "WOW", "SHEN"],

    # Autos (Short 7, -3.37)
    "autos":  ["TSLA", "F", "GM", "STLA", "TM", "HMC", "RIVN",
               "LCID", "NKLA", "FSR"],

    # Txtls (Short 8, -3.11) — ny sektor
    "txtls":  ["PVH", "VFC", "HBI", "COLM", "URBN", "ANF", "AEO",
               "EXPR", "CATO", "BURL"],

    # RlEst (Short 9, -3.06)
    "rlest":  ["AMT", "PLD", "EQIX", "SPG", "O", "VICI", "PSA",
               "EQR", "AVB", "MAA"],

    # Chips (Short 10, -2.92)
    "chips":  ["NVDA", "TSM", "AVGO", "ASML", "AMD", "INTC", "QCOM",
               "TXN", "MU", "ADI"]
}

sectors = {
    # ── LONG sektorer ─────────────────────────────────────────

    # Drugs (Long 1, +13.29)
    "drugs":  ["LLY", "JNJ", "NVO", "MRK", "ABBV", "PFE", "AZN", "BMY",
               "GILD", "AMGN", "REGN", "VRTX", "BIIB", "ALNY", "JAZZ",
               "INCY", "EXEL", "UTHR", "IONS"],

    # Oil (Long 2, +7.98)
    "oil":    ["XOM", "CVX", "SHEL", "TTE", "BP", "COP", "EOG", "SLB",
               "PBR", "ENB", "OXY", "DVN", "MPC", "PSX", "VLO", "FANG",
               "HAL", "BKR", "NOV", "SM", "MTDR"],

    # Soda (Long 3, +7.27)
    "soda":   ["KO", "PEP", "KDP", "MNST", "CELH", "COKE", "FIZZ",
               "KOF", "PRMB", "REED"],

    # Coal (Long 4, +6.45)
    "coal":   ["BTU", "AMR", "ARLP", "HCC", "METC", "NRP", "SXC", "NC"],

    # Util (Long 5, +5.98)
    "util":   ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
               "PEG", "ED", "EIX", "AWK", "CNP", "WEC", "ETR", "FE",
               "PPL", "NI", "CMS", "LNT", "EVRG", "OGE"],

    # Beer (Long 6, +5.59)
    "beer":   ["BUD", "DEO", "STZ", "TAP", "SAM", "CCU", "HEINY",
               "FMX", "BF-B", "MGPI"],

    # Insur (Long 7, +5.50)
    "insur":  ["MET", "PRU", "AFL", "ALL", "PGR", "TRV", "AIG",
               "CB", "HIG", "MKL", "RNR", "ERIE", "WRB", "CINF", "GL"],

    # Ships (Long 8, +5.43)
    "ships":  ["ZIM", "MATX", "SBLK", "NMM", "ESEA",
               "GNK", "EGLE", "DSX", "TOPS", "SHIP"],

    # Agric (Long 9, +5.36)
    "agric":  ["CTVA", "NTR", "MOS", "CF", "AGCO", "CALM", "ANDE",
               "FMC", "INGR"],

    # LabEq (Long 10, +4.28)
    "labeq":  ["TMO", "DHR", "A", "WAT", "BIO", "BRKR", "IDXX",
               "FLS", "FELE"],

    # ── SHORT sektorer ────────────────────────────────────────

    # Telcm (Short 1, -14.47)
    "telcm":  ["T", "VZ", "TMUS", "CMCSA", "CHTR", "LUMN", "SHEN"],

    # Toys (Short 2, -10.51)
    "toys":   ["HAS", "MAT", "JAKK", "EA", "TTWO"],

    # Paper (Short 3, -8.23)
    "paper":  ["IP", "PKG", "SEE", "SON", "GPK", "CLW", "CEVA", "SLGN"],

    # Hshld (Short 4, -6.93)
    "hshld":  ["PG", "CL", "KMB", "CHD", "SPB", "HRB", "ENR",
               "CENT", "RCKY"],

    # Softw (Short 5, -6.77)
    "softw":  ["MSFT", "ORCL", "SAP", "ADBE", "CRM", "NOW", "INTU",
               "WDAY", "SNOW", "DDOG"],

    # Gold (Short 6, -6.33)
    "gold":   ["NEM", "WPM", "KGC", "AGI", "EGO",
               "IAG", "PAAS", "CDE", "HL", "AEM"],

    # Trans (Short 7, -5.48)
    "trans":  ["UPS", "FDX", "DAL", "UAL", "AAL", "LUV", "JBLU",
               "ALGT", "SKYW"],

    # BusSv (Short 8, -4.65)
    "bussv":  ["ADP", "PAYX", "RHI", "MAN", "KELYA",
               "KFRC", "TBI", "HCSG", "CNXN"],

    # Boxes (Short 9, -4.54)
    "boxes":  ["PLD", "STAG", "EGP", "REXR", "FR", "TRNO"],

    # Rtail (Short 10, -4.30)
    "rtail":  ["AMZN", "WMT", "COST", "HD", "LOW", "TGT", "TJX",
               "ROST", "DG", "DLTR", "BBY", "KSS"],
}


sectors = {
    "aero": ['BA', 'RTX', 'HON', 'NOC', 'LHX', 'TDG', 'HEI', 'UAVS', 'TDY', 'GRMN', 'TXT', 'ESLT', 'EVTL', 'AVAV', 'JOBY'],
    "agric": ['EDBL', 'CTVA', 'CALM', 'FDP', 'AGRO', 'DOLE', 'AVO', 'BV', 'CVGW'],
    "autos": ['EMPD', 'DCX', 'FFAI', 'TSLA', 'TM', 'WKHS', 'RACE', 'F', 'STLA', 'PCAR', 'GM', 'HMC', 'NIO', 'VFS', 'APTV'],
    "bussv": ['DVLT', 'V', 'MA', 'BABA', 'TGL', 'ACN', 'BNBX', 'PDD', 'PYPL', 'FISV', 'UBER', 'RELX', 'MELI', 'PAYX', 'IQV'],
    "chips": ['TSM', 'NVDA', 'ASTI', 'AVGO', 'TXN', 'INTC', 'AMD', 'ADI', 'AMAT', 'MU', 'NXPI', 'MCHP', 'ENPH', 'MRVL', 'STM'],
    "coal": ['BTU', 'CNR', 'ARLP', 'AMR', 'HCC', 'NRP', 'METC'],
    "drugs": ['ADTX', 'CDT', 'ALLR', 'APVO', 'HIND', 'PBM', 'GRI', 'RNAZ', 'JNJ', 'LLY', 'NVO', 'ABBV', 'MRK', 'BDRX', 'PFE'],
    "elceq": ['CETX', 'BURU', 'WTO', 'QCOM', 'SONY', 'KUST', 'NVVE', 'GE', 'EMR', 'APH', 'MSI', 'OTIS', 'NOK', 'ERIC', 'PLUG'],
    "food": ['KO', 'PEP', 'BUD', 'DEO', 'MDLZ', 'MNST', 'KDP', 'HSY', 'KHC', 'ADM', 'GIS', 'STZ', 'ABEV', 'FMX', 'CCEP'],
    "fun": ['AMC', 'LION', 'CNK', 'FUBO'],
    "guns": ['LMT', 'MNTS', 'RKLB', 'KTOS'],
    "hlth": ['ACON', 'HCA', 'NIVF', 'LH', 'DGX', 'UHS', 'FMS', 'ACHC', 'CHE', 'AGL', 'EHC', 'NTRA', 'ENSG', 'DVA', 'OPCH'],
    "insur": ['XHG', 'UNH', 'ELV', 'CB', 'CI', 'MRSH', 'PGR', 'AON', 'HUM', 'AJG', 'MET', 'CNC', 'TRV', 'AFL', 'PUK'],
    "medeq": ['STSS', 'ISRG', 'SYK', 'MDT', 'BSX', 'BDX', 'MMM', 'DXCM', 'EW', 'NUWE', 'ALC', 'RMD', 'GEHC', 'BAX', 'ZBH'],
    "oil": ['XOM', 'CVX', 'SHEL', 'COP', 'TTE', 'BP', 'ENB', 'EQNR', 'EOG', 'OXY', 'CNQ', 'PBR', 'PSX', 'E', 'WDS'],
    "paper": ['KMB', 'SW', 'IP', 'AVY', 'SUZ', 'PKG', 'GPK', 'REYN', 'SON', 'SLVM', 'MAGN', 'MATV', 'CLW', 'PACK'],
    "rlest": ['WHLR', 'PLD', 'AMT', 'EQIX', 'CCI', 'O', 'BN', 'WELL', 'PSA', 'SPG', 'DLR', 'VICI', 'CMCT', 'SBAC', 'EXR'],
    "rtail": ['AMZN', 'WMT', 'HD', 'KXIN', 'COST', 'CVS', 'LOW', 'TJX', 'DBGI', 'TGT', 'SHW', 'DG', 'ORLY', 'AZO', 'ROST'],
    "rubbr": ['NKE', 'ENTG', 'CSL', 'DECK', 'ATR', 'WMS', 'ONON', 'CROX', 'NWL', 'GT', 'AWI', 'NPO', 'MYE', 'LWLG'],
    "ships": ['VMAR', 'GD', 'SVRN', 'RCL', 'BIP', 'CCL', 'CUK', 'HII', 'NCLH', 'KEX', 'CMBT', 'STNG', 'GLNG', 'FRO', 'MATX'],
    "smoke": ['XXII', 'PM', 'BTI', 'MO', 'RLX'],
    "softw": ['MSFT', 'XTIA', 'NXTT', 'GOOG', 'GOOGL', 'HOLO', 'YYAI', 'SVRE', 'META', 'ORCL', 'ADBE', 'CYN', 'SAP', 'CRM', 'INTU'],
    "telcm": ['CHAI', 'TMUS', 'VZ', 'TBB', 'T', 'CMCSA', 'AMX', 'CHTR', 'BCE', 'CHT', 'TU', 'WBD', 'RCI', 'TLK', 'VOD'],
    "toys": ['DIS', 'WMG', 'LYV', 'HAS', 'MTN', 'CHDN', 'PLNT', 'DKNG', 'MAT', 'TRUG', 'TKO', 'MSGS', 'FUN', 'MANU', 'CALY'],
    "txtls": ['CTAS', 'LULU', 'VFC', 'MHK', 'RL', 'LEVI', 'GIL', 'COLM', 'UAA', 'UA', 'PVH', 'ZGN', 'AIN', 'CRI', 'KTB'],
    "util": ['NEE', 'SO', 'DUK', 'WM', 'NGG', 'D', 'AEP', 'SRE', 'EPD', 'EXC', 'XEL', 'RSG', 'TRP', 'WMB', 'PCG'],
    "whlsl": ['GNLN', 'MCK', 'SYY', 'TEL', 'COR', 'GWW', 'FERG', 'GPC', 'CAH', 'QXO', 'LKQ', 'GWAV', 'DPZ', 'POOL', 'RS'],
}
"""

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

"""
sectors = {
    "agric": ["ADM", "AGCO", "AGRO", "ANDE", "BG", "CAG", "CALM", "CF", "CPB", "CVGW", "FDP"],
    "bussv": ["ACN", "ADP", "BABA", "BNBX", "BR", "CTAS", "FISV", "IQV", "MA", "MELI", "PAYX", "PYPL"],
    "chips": ["ADI", "AMAT", "AMD", "AVGO", "INTC", "MU", "NVDA", "TSM", "TXN"],
    "drugs": ["ABBV", "ALNY", "AMGN", "AZN", "BIIB", "BMY", "GILD", "INCY", "JNJ", "LLY", "MRK", "NVO"],
    "food":  ["ABEV", "CCEP", "GIS", "HSY", "KHC", "MDLZ"],
    "guns":  ["AXON", "BA", "DRS", "GD", "HII", "KTOS", "LMT", "NOC", "NPK", "OLN", "RGR", "RTX"],
    "insur": ["AJG", "AON", "CB", "CI", "ELV", "HUM", "MRSH", "PGR", "UNH"],
    "rtail": ["AMZN", "COST", "CVS", "HD", "LOW", "TGT", "TJX", "WMT"],
    "rubbr": ["ATR", "CROX", "CSL", "DECK", "ENTG", "NKE", "NWL", "WMS"],
    "softw": ["ADBE", "CDNS", "CRM", "GOOGL", "MANH", "META", "MSFT", "NOW", "ORCL", "PTC", "SAP", "SNPS", "WDAY", "XTIA"],
    "toys":  ["CHDN", "DIS", "HAS", "LYV", "MAT", "MTN", "PLNT"],
    "txtls": ["COLM", "GIL", "LULU", "MHK", "RL", "UAA", "VFC"],
    "beer":  ["BF-B", "BUD", "CCU", "DEO", "FMX", "HEINY", "MGPI", "SAM", "STZ", "TAP"],
}
"""

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
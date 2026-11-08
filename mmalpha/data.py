"""Download the raw inputs.

- CVM daily fund reports (informe diário), reduced to one row per fund per month
- CVM fund registries (legacy cad_fi + the post-RCVM 175 registro_classe)
- CDI and USD/BRL from the BCB SGS API
- Ibovespa from Yahoo Finance
- Tesouro Direto bond prices (for the long inflation-linked bond factor)

The daily reports are ~1 GB of zips for 2016-2025. Each month is reduced to the
last report of every fund as soon as it is read, so only the small monthly
snapshots are kept on disk (data/cvm/monthly/YYYYMM.parquet). Re-running skips
months that are already there.

    uv run python -m mmalpha.data --start 2016-01 --end 2025-12
"""

from __future__ import annotations

import argparse
import io
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATA = Path("data")
CVM_URL = "https://dados.cvm.gov.br/dados/FI"
SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"

# Up to 2020 the CVM only ships one zip per year; monthly zips start in 2021.
FIRST_MONTHLY_ZIP_YEAR = 2021

# The informe diário header changed twice (2021 added TP_FUNDO; RCVM 175 renamed
# the CNPJ column and added subclasses). These are the only columns we keep.
COLUMNS = {
    "CNPJ_FUNDO": "cnpj",
    "CNPJ_FUNDO_CLASSE": "cnpj",
    "DT_COMPTC": "date",
    "VL_QUOTA": "quota",
    "VL_PATRIM_LIQ": "nav",
    "NR_COTST": "holders",
}

SGS_SERIES = {"cdi": 12, "usdbrl": 1}  # CDI in % per day; PTAX sell rate

# Full price history of every Tesouro Direto bond, from the Treasury open-data portal.
TESOURO_URL = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/"
    "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
    "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv"
)


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


HTTP = _session()


def _get(url: str, **kwargs) -> bytes:
    r = HTTP.get(url, timeout=120, **kwargs)
    r.raise_for_status()
    return r.content


def month_end_snapshot(csv) -> pd.DataFrame:
    """Last report of each fund in one monthly informe diário file.

    Rows for subclasses (RCVM 175) are dropped: the class-level quota is what
    existed before 2024 and keeps the series comparable over time.
    """
    df = pd.read_csv(
        csv,
        sep=";",
        usecols=lambda c: c in COLUMNS or c == "ID_SUBCLASSE",
        dtype={"CNPJ_FUNDO": str, "CNPJ_FUNDO_CLASSE": str, "ID_SUBCLASSE": str},
    )
    if "ID_SUBCLASSE" in df:
        df = df[df["ID_SUBCLASSE"].isna()].drop(columns="ID_SUBCLASSE")
    df = df.rename(columns=COLUMNS)

    df["cnpj"] = df["cnpj"].str.replace(r"\D", "", regex=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["quota"] > 0]

    return (
        df.sort_values(["cnpj", "date"])
        .drop_duplicates("cnpj", keep="last")
        .reset_index(drop=True)[["cnpj", "date", "quota", "nav", "holders"]]
    )


def _months(start: str, end: str) -> list[str]:
    return [p.strftime("%Y%m") for p in pd.period_range(start, end, freq="M")]


def download_cvm_daily(start: str, end: str) -> None:
    out = DATA / "cvm" / "monthly"
    out.mkdir(parents=True, exist_ok=True)
    missing = [m for m in _months(start, end) if not (out / f"{m}.parquet").exists()]

    for year in sorted({int(m[:4]) for m in missing}):
        wanted = [m for m in missing if int(m[:4]) == year]

        if year < FIRST_MONTHLY_ZIP_YEAR:
            print(f"CVM {year} (yearly zip)")
            raw = _get(f"{CVM_URL}/DOC/INF_DIARIO/DADOS/HIST/inf_diario_fi_{year}.zip")
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for m in wanted:
                    with z.open(f"inf_diario_fi_{m}.csv") as f:
                        month_end_snapshot(f).to_parquet(out / f"{m}.parquet")
            continue

        for m in wanted:
            print(f"CVM {m}")
            raw = _get(f"{CVM_URL}/DOC/INF_DIARIO/DADOS/inf_diario_fi_{m}.zip")
            with zipfile.ZipFile(io.BytesIO(raw)) as z, z.open(z.namelist()[0]) as f:
                month_end_snapshot(f).to_parquet(out / f"{m}.parquet")


def download_cvm_registry() -> None:
    """Both registries are needed: funds closed before RCVM 175 only exist in
    cad_fi, and funds that adapted to it only carry their class in
    registro_classe. Saved as-is; parsing happens in the cleaning step."""
    out = DATA / "cvm" / "registry"
    out.mkdir(parents=True, exist_ok=True)
    print("CVM registry")
    (out / "cad_fi.csv").write_bytes(_get(f"{CVM_URL}/CAD/DADOS/cad_fi.csv"))
    raw = _get(f"{CVM_URL}/CAD/DADOS/registro_fundo_classe.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        (out / "registro_classe.csv").write_bytes(z.read("registro_classe.csv"))


def sgs_series(code: int, start: date, end: date) -> pd.Series:
    """The SGS API refuses daily-series requests longer than 10 years, so ask
    one calendar year at a time."""
    parts = []
    for year in range(start.year, end.year + 1):
        lo, hi = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        params = {
            "formato": "json",
            "dataInicial": lo.strftime("%d/%m/%Y"),
            "dataFinal": hi.strftime("%d/%m/%Y"),
        }
        r = HTTP.get(SGS_URL.format(code=code), params=params, timeout=60)
        r.raise_for_status()
        parts.append(pd.DataFrame(r.json()))

    df = pd.concat(parts)
    idx = pd.to_datetime(df["data"], format="%d/%m/%Y")
    return pd.Series(df["valor"].astype(float).to_numpy(), index=idx).sort_index()


def download_market(start: str, end: str) -> None:
    import yfinance as yf

    out = DATA / "market"
    out.mkdir(parents=True, exist_ok=True)
    lo = pd.Period(start, "M").start_time.date()
    hi = pd.Period(end, "M").end_time.date()

    for name, code in SGS_SERIES.items():
        print(f"BCB {name}")
        sgs_series(code, lo, hi).rename(name).to_frame().to_parquet(out / f"{name}.parquet")

    # Ibovespa is already a total-return index (dividends reinvested), so the
    # plain close is the right series to use.
    print("Ibovespa")
    ibov = yf.download("^BVSP", start=lo, end=hi + timedelta(days=1),
                       auto_adjust=False, progress=False)
    if ibov.empty:
        raise RuntimeError("Yahoo Finance returned no Ibovespa data")
    close = ibov["Close"].squeeze().rename("ibov")
    close.index = close.index.tz_localize(None)
    close.to_frame().to_parquet(out / "ibov.parquet")

    print("Tesouro Direto")
    (out / "tesouro_direto.csv").write_bytes(_get(TESOURO_URL))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--start", default="2016-01")
    p.add_argument("--end", default="2025-12")
    args = p.parse_args()

    download_market(args.start, args.end)
    download_cvm_registry()
    download_cvm_daily(args.start, args.end)


if __name__ == "__main__":
    main()

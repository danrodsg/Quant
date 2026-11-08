"""Baixa os dados brutos: CVM (informe diário + cadastro), BCB, Ibovespa e Tesouro Direto."""

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
TESOURO_URL = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/"
    "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
    "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv"
)

# até 2020 a CVM só tem zip anual
FIRST_MONTHLY_ZIP_YEAR = 2021

# o header mudou em 2021 e de novo com a RCVM 175
COLUMNS = {
    "CNPJ_FUNDO": "cnpj",
    "CNPJ_FUNDO_CLASSE": "cnpj",
    "DT_COMPTC": "date",
    "VL_QUOTA": "quota",
    "VL_PATRIM_LIQ": "nav",
    "NR_COTST": "holders",
}

SGS_SERIES = {"cdi": 12, "usdbrl": 1}

HTTP = requests.Session()
HTTP.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))


def _get(url: str, **kwargs) -> bytes:
    r = HTTP.get(url, timeout=120, **kwargs)
    r.raise_for_status()
    return r.content


def month_end_snapshot(csv) -> pd.DataFrame:
    """Última cota de cada fundo no mês."""
    df = pd.read_csv(
        csv,
        sep=";",
        usecols=lambda c: c in COLUMNS or c == "ID_SUBCLASSE",
        dtype={"CNPJ_FUNDO": str, "CNPJ_FUNDO_CLASSE": str, "ID_SUBCLASSE": str},
    )
    # fico só com a cota da classe, que é o que existia antes de 2024
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


def download_cvm_daily(start: str, end: str) -> None:
    out = DATA / "cvm" / "monthly"
    out.mkdir(parents=True, exist_ok=True)
    months = [p.strftime("%Y%m") for p in pd.period_range(start, end, freq="M")]
    missing = [m for m in months if not (out / f"{m}.parquet").exists()]

    for year in sorted({int(m[:4]) for m in missing}):
        wanted = [m for m in missing if int(m[:4]) == year]

        if year < FIRST_MONTHLY_ZIP_YEAR:
            print(f"CVM {year}")
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
    # precisa dos dois: fundos fechados antes da RCVM 175 só estão no cad_fi
    out = DATA / "cvm" / "registry"
    out.mkdir(parents=True, exist_ok=True)
    print("CVM cadastro")
    (out / "cad_fi.csv").write_bytes(_get(f"{CVM_URL}/CAD/DADOS/cad_fi.csv"))
    raw = _get(f"{CVM_URL}/CAD/DADOS/registro_fundo_classe.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        (out / "registro_classe.csv").write_bytes(z.read("registro_classe.csv"))


def sgs_series(code: int, start: date, end: date) -> pd.Series:
    # a API do SGS não aceita mais de 10 anos de série diária por request
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

    # Ibovespa já é índice de retorno total
    print("Ibovespa")
    ibov = yf.download("^BVSP", start=lo, end=hi + timedelta(days=1),
                       auto_adjust=False, progress=False)
    if ibov.empty:
        raise RuntimeError("Yahoo não retornou o Ibovespa")
    close = ibov["Close"].squeeze().rename("ibov")
    close.index = close.index.tz_localize(None)
    close.to_frame().to_parquet(out / "ibov.parquet")

    print("Tesouro Direto")
    (out / "tesouro_direto.csv").write_bytes(_get(TESOURO_URL))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2016-01")
    p.add_argument("--end", default="2025-12")
    args = p.parse_args()

    download_market(args.start, args.end)
    download_cvm_registry()
    download_cvm_daily(args.start, args.end)

"""Universo de multimercados, retornos mensais e fatores."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUT = DATA / "processed"

# 85% dos multimercados têm mediana de 1 cotista (exclusivos ou masters).
# Esse corte também evita contar master e FIC como dois fundos.
MIN_HOLDERS = 20
MIN_NAV = 10e6

CRYPTO = r"CRIPTO|CRYPTO|BITCOIN|BLOCKCHAIN|ETHEREUM|\bCOIN\b|\bDEFI\b"

# erro de cota: mudança de unidade (>10x) ou salto >30% que volta no mês seguinte
MAX_RATIO = 10.0
SPIKE = 0.30
SPIKE_UNDO = 0.05


def load_registry() -> pd.DataFrame:
    reg = DATA / "cvm" / "registry"

    cad = pd.read_csv(reg / "cad_fi.csv", sep=";", encoding="latin1", dtype=str)
    cad = (
        cad.assign(cnpj=cad["CNPJ_FUNDO"].str.replace(r"\D", "", regex=True))
        .sort_values("DT_REG")
        .drop_duplicates("cnpj", keep="last")  # tem CNPJ cancelado e registrado de novo
        .rename(columns={"DENOM_SOCIAL": "name", "CLASSE": "category",
                         "FUNDO_EXCLUSIVO": "exclusive", "CONDOM": "structure"})
        .assign(source="cad_fi")
    )

    new = pd.read_csv(reg / "registro_classe.csv", sep=";", encoding="latin1", dtype=str)
    new = (
        new.assign(cnpj=new["CNPJ_Classe"].str.replace(r"\D", "", regex=True))
        .drop_duplicates("cnpj", keep="last")
        .rename(columns={"Denominacao_Social": "name", "Classificacao": "category",
                         "Exclusivo": "exclusive", "Forma_Condominio": "structure"})
        .assign(source="registro_classe")
    )

    # cadastro novo tem prioridade, campos vazios vêm do antigo
    cols = ["cnpj", "name", "category", "exclusive", "structure", "source"]
    return new[cols].set_index("cnpj").combine_first(cad[cols].set_index("cnpj"))


def load_snapshots() -> pd.DataFrame:
    files = sorted((DATA / "cvm" / "monthly").glob("*.parquet"))
    return pd.concat(
        pd.read_parquet(f).assign(month=pd.Period(f.stem, "M")) for f in files
    )


def clean_quota_returns(quota: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # mês faltando vira NaN, não retorno de dois meses
    ret = quota.pct_change(fill_method=None)

    unit_change = (ret > MAX_RATIO - 1) | (ret < 1 / MAX_RATIO - 1)

    nxt = ret.shift(-1)
    spike = (ret.abs() > SPIKE) & (((1 + ret) * (1 + nxt) - 1).abs() < SPIKE_UNDO)
    spike = spike | spike.shift(1, fill_value=False)  # tira a ida e a volta

    bad = unit_change | spike
    return ret.mask(bad), int(bad.sum().sum())


def long_bond_returns(prices: pd.DataFrame, target_years: float = 10.0) -> pd.Series:
    """Retorno mensal do Tesouro IPCA+ (sem cupom) mais perto de 10 anos.

    Uso no lugar do IMA-B, que saiu do SGS em 2023.
    """
    month_end = (
        prices.assign(month=prices["date"].dt.to_period("M"))
        .sort_values("date")
        .groupby(["month", "maturity"])["price"].last()
        .unstack()
    )
    out = {}
    for prev, cur in zip(month_end.index[:-1], month_end.index[1:]):
        held = month_end.loc[prev].dropna()
        both = held.index.intersection(month_end.loc[cur].dropna().index)
        if both.empty:
            continue
        years = (both - prev.end_time).days / 365.25
        bond = both[abs(years - target_years).argmin()]
        out[cur] = month_end.loc[cur, bond] / month_end.loc[prev, bond] - 1
    return pd.Series(out)


def load_tesouro_ipca() -> pd.DataFrame:
    td = pd.read_csv(DATA / "market" / "tesouro_direto.csv", sep=";", decimal=",")
    td = td[td["Tipo Titulo"] == "Tesouro IPCA+"]
    return pd.DataFrame({
        "date": pd.to_datetime(td["Data Base"], format="%d/%m/%Y"),
        "maturity": pd.to_datetime(td["Data Vencimento"], format="%d/%m/%Y"),
        "price": td["PU Base Manha"],
    })


def monthly_factors() -> pd.DataFrame:
    mkt = DATA / "market"
    cdi = pd.read_parquet(mkt / "cdi.parquet")["cdi"]
    ibov = pd.read_parquet(mkt / "ibov.parquet")["ibov"]
    usd = pd.read_parquet(mkt / "usdbrl.parquet")["usdbrl"]

    month = lambda s: s.index.to_period("M")
    factors = pd.DataFrame({
        "cdi": (1 + cdi / 100).groupby(month(cdi)).prod() - 1,
        "ibov": ibov.groupby(month(ibov)).last().pct_change(),
        "usd": usd.groupby(month(usd)).last().pct_change(),
    })
    factors["ntnb"] = long_bond_returns(load_tesouro_ipca())
    return factors.dropna().rename_axis("month")


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    snaps = load_snapshots()
    last_month = snaps["month"].max()

    mm = registry[registry["category"] == "Multimercado"]
    snaps = snaps[snaps["cnpj"].isin(mm.index)]

    funds = mm.join(
        snaps.groupby("cnpj").agg(
            first_month=("month", "min"),
            last_month=("month", "max"),
            median_holders=("holders", "median"),
            median_nav=("nav", "median"),
        ),
        how="inner",
    )
    funds["alive"] = funds["last_month"] == last_month

    # cada fundo leva o primeiro filtro em que falha, pra o funil fechar
    rules = [
        ("exclusive", funds["exclusive"] == "S"),
        ("closed_end", funds["structure"] == "Fechado"),
        ("crypto", funds["name"].str.contains(CRYPTO, case=False, na=False)),
        ("few_holders", funds["median_holders"] < MIN_HOLDERS),
        ("small_nav", funds["median_nav"] < MIN_NAV),
    ]
    funds["excluded_by"] = pd.Series(np.nan, index=funds.index, dtype="object")
    for name, failed in rules:
        funds.loc[failed & funds["excluded_by"].isna(), "excluded_by"] = name

    keep = funds.index[funds["excluded_by"].isna()]
    quota = snaps[snaps["cnpj"].isin(keep)].pivot(index="month", columns="cnpj", values="quota")
    ret, n_bad = clean_quota_returns(quota)

    returns = ret.stack().dropna().rename("ret").reset_index()
    funds["n_returns"] = returns.groupby("cnpj").size().reindex(funds.index)

    funds.to_parquet(OUT / "funds.parquet")
    returns.to_parquet(OUT / "returns.parquet")
    monthly_factors().to_parquet(OUT / "factors.parquet")

    print(f"multimercados: {len(funds)}")
    for name, _ in rules:
        print(f"  - {name}: {(funds['excluded_by'] == name).sum()}")
    alive = funds.loc[keep, "alive"]
    print(f"amostra: {len(keep)} ({alive.sum()} vivos, {(~alive).sum()} encerrados)")
    print(f"retornos mensais: {len(returns)} ({n_bad} erros de cota removidos)")


if __name__ == "__main__":
    build()

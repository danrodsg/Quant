"""Build the fund universe and the monthly return panel.

Inputs come from mmalpha.data. Outputs, in data/processed/:

    funds.parquet    one row per multimercado fund, with the filter it failed (if any)
    returns.parquet  monthly net-of-fee returns of the funds that pass every filter
    factors.parquet  monthly CDI, Ibovespa and USD/BRL returns

    uv run python -m mmalpha.clean
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUT = DATA / "processed"

# A fund needs this many holders (median over its life) to count as a fund that
# someone could actually buy. 85% of multimercados have a median of 1: they are
# exclusive funds of a single family or company, or master funds held only by
# their own feeder funds. Dropping them also avoids counting a master and its
# feeder as two separate funds.
MIN_HOLDERS = 20
MIN_NAV = 10e6  # median NAV in BRL; below this it is usually an incubator fund

# Quota errors. A monthly move above +900% or below -90% in a multimercado is a
# unit change in the reported quota, not a return. A move above 30% that is
# undone the following month (within 5%) is a typo in one month's quota.
MAX_RATIO = 10.0
SPIKE = 0.30
SPIKE_UNDO = 0.05


def load_registry() -> pd.DataFrame:
    """Merge the two CVM registries into one row per CNPJ.

    When a CNPJ appears in both, the post-RCVM 175 registry wins because it is
    the one the CVM keeps updated; blanks there are filled from cad_fi. cad_fi has repeated CNPJs (funds cancelled
    and registered again); the most recent registration is kept.
    """
    reg = DATA / "cvm" / "registry"

    cad = pd.read_csv(reg / "cad_fi.csv", sep=";", encoding="latin1", dtype=str)
    cad = (
        cad.assign(cnpj=cad["CNPJ_FUNDO"].str.replace(r"\D", "", regex=True))
        .sort_values("DT_REG")
        .drop_duplicates("cnpj", keep="last")
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

    # Field by field, so a fund whose new registration left the category blank
    # still keeps the category it had in cad_fi.
    cols = ["cnpj", "name", "category", "exclusive", "structure", "source"]
    return new[cols].set_index("cnpj").combine_first(cad[cols].set_index("cnpj"))


def load_snapshots() -> pd.DataFrame:
    files = sorted((DATA / "cvm" / "monthly").glob("*.parquet"))
    return pd.concat(
        pd.read_parquet(f).assign(month=pd.Period(f.stem, "M")) for f in files
    )


def clean_quota_returns(quota: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Monthly returns from a month x fund quota table, with quota errors removed.

    Returns are only computed between consecutive months: a fund that skips a
    month gets NaN rather than a two-month return.
    """
    ret = quota.pct_change(fill_method=None)

    unit_change = (ret > MAX_RATIO - 1) | (ret < 1 / MAX_RATIO - 1)

    nxt = ret.shift(-1)
    spike = (ret.abs() > SPIKE) & (((1 + ret) * (1 + nxt) - 1).abs() < SPIKE_UNDO)
    # The bad quota is in month t, so both the move into it and out of it go.
    spike = spike | spike.shift(1, fill_value=False)

    bad = unit_change | spike
    return ret.mask(bad), int(bad.sum().sum())


def monthly_factors() -> pd.DataFrame:
    mkt = DATA / "market"
    cdi = pd.read_parquet(mkt / "cdi.parquet")["cdi"]
    ibov = pd.read_parquet(mkt / "ibov.parquet")["ibov"]
    usd = pd.read_parquet(mkt / "usdbrl.parquet")["usdbrl"]

    month = lambda s: s.index.to_period("M")
    return pd.DataFrame({
        "cdi": (1 + cdi / 100).groupby(month(cdi)).prod() - 1,
        "ibov": ibov.groupby(month(ibov)).last().pct_change(),
        "usd": usd.groupby(month(usd)).last().pct_change(),
    }).dropna().rename_axis("month")


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
    # Alive = still reporting in the last month of the sample. Everything else
    # was closed, merged or liquidated somewhere in between.
    funds["alive"] = funds["last_month"] == last_month

    # Each fund gets the first filter it fails, so the funnel adds up.
    rules = [
        ("exclusive", funds["exclusive"] == "S"),
        ("closed_end", funds["structure"] == "Fechado"),
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

    print(f"Multimercados with data: {len(funds):>6}")
    for name, _ in rules:
        print(f"  excluded {name:<12} {(funds['excluded_by'] == name).sum():>6}")
    print(f"In sample:               {len(keep):>6}  "
          f"({funds.loc[keep, 'alive'].sum()} alive, {(~funds.loc[keep, 'alive']).sum()} dead)")
    print(f"Monthly returns:         {len(returns):>6}  ({n_bad} removed as quota errors)")


if __name__ == "__main__":
    build()

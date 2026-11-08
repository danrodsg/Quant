"""Alpha por fundo, bootstrap sorte vs. habilidade e viés de sobrevivência.

    r - cdi = a + b1 (ibov - cdi) + b2 usd + b3 (ntnb - cdi) + e
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED = Path("data/processed")
RESULTS = Path("results")

FACTORS = ["ibov", "usd", "ntnb"]

# abaixo de 2% a.a. de vol é basicamente fundo de crédito marcado na curva:
# retorno liso, spread pequeno vira t-stat enorme. Analiso separado.
LOW_VOL = 2.0
MIN_OBS = 24
MIN_OBS_BOOT = 12  # FF2010 usam 8
N_BOOT = 1000
PERCENTILES = list(range(1, 100))
SHOW = [1, 5, 10, 25, 50, 75, 90, 95, 99]
SEED = 175


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    funds = pd.read_parquet(PROCESSED / "funds.parquet")
    funds = funds[funds["excluded_by"].isna()]
    returns = pd.read_parquet(PROCESSED / "returns.parquet")
    factors = pd.read_parquet(PROCESSED / "factors.parquet")

    ret = returns.pivot(index="month", columns="cnpj", values="ret")
    ret = ret.reindex(factors.index)
    excess = ret.sub(factors["cdi"], axis=0)

    X = pd.DataFrame({
        "const": 1.0,
        "ibov": factors["ibov"] - factors["cdi"],
        "usd": factors["usd"],
        "ntnb": factors["ntnb"] - factors["cdi"],
    })
    return excess, X, funds


def batch_ols(X: np.ndarray, Y: np.ndarray, weights: np.ndarray | None = None,
              min_obs: int = MIN_OBS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS de cada coluna de Y em X de uma vez, ignorando NaN por fundo.

    weights = quantas vezes cada mês foi sorteado no bootstrap.
    """
    T, k = X.shape
    w = np.ones(T) if weights is None else np.asarray(weights, dtype=float)

    valid = ~np.isnan(Y)
    Y0 = np.where(valid, Y, 0.0)
    W = valid * w[:, None]

    nobs = W.sum(axis=0)
    ok = nobs >= max(min_obs, k + 1)

    XtX = np.einsum("tn,ti,tj->nij", W, X, X)
    XtY = np.einsum("tn,ti,tn->ni", W, X, Y0)
    XtX[~ok] = np.eye(k)  # só pra não quebrar o solve

    coef = np.linalg.solve(XtX, XtY[..., None])[..., 0]
    resid = Y0 - X @ coef.T
    s2 = (W * resid**2).sum(axis=0) / np.maximum(nobs - k, 1)
    se = np.sqrt(s2[:, None] * np.diagonal(np.linalg.inv(XtX), axis1=1, axis2=2))

    coef[~ok] = np.nan
    se[~ok] = np.nan
    return coef, se, nobs


def fund_alphas(excess: pd.DataFrame, X: pd.DataFrame, funds: pd.DataFrame) -> pd.DataFrame:
    coef, se, nobs = batch_ols(X.to_numpy(), excess.to_numpy())
    out = pd.DataFrame(coef, index=excess.columns, columns=["alpha", *FACTORS])
    out["t_alpha"] = coef[:, 0] / se[:, 0]
    out["n_months"] = nobs.astype(int)
    out = out.dropna(subset=["alpha"])
    out["alpha_pct_year"] = out["alpha"] * 12 * 100
    out["vol_pct_year"] = excess[out.index].std() * np.sqrt(12) * 100
    out["group"] = np.where(out["vol_pct_year"] < LOW_VOL, "cdi_plus", "directional")
    info = funds[["name", "alive", "first_month", "last_month"]]
    return info.join(out, how="inner").sort_values("t_alpha", ascending=False)


def luck_vs_skill(excess: pd.DataFrame, X: pd.DataFrame, alphas: pd.DataFrame,
                  n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Bootstrap de Fama & French (2010).

    Tira o alpha estimado de cada fundo (alpha verdadeiro = 0), sorteia meses
    (os mesmos pra todos os fundos, mantém a correlação entre eles) e
    reestima. Dá a distribuição de t(alpha) que só a sorte produz.
    """
    cols = alphas.index
    Y = excess[cols].to_numpy()
    Xn = X.to_numpy()
    Y_null = Y - alphas["alpha"].to_numpy()

    actual = np.percentile(alphas["t_alpha"], PERCENTILES)

    rng = np.random.default_rng(seed)
    T = len(Xn)
    sims = np.empty((n_boot, len(PERCENTILES)))
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, T, T), minlength=T)
        coef, se, _ = batch_ols(Xn, Y_null, counts, min_obs=MIN_OBS_BOOT)
        sims[b] = np.nanpercentile(coef[:, 0] / se[:, 0], PERCENTILES)

    return pd.DataFrame({
        "percentile": PERCENTILES,
        "t_actual": actual,
        "t_luck_median": np.median(sims, axis=0),
        "t_luck_p5": np.percentile(sims, 5, axis=0),
        "t_luck_p95": np.percentile(sims, 95, axis=0),
        "pct_sims_below_actual": (sims < actual).mean(axis=0) * 100,
    })


def survivorship(excess: pd.DataFrame, X: pd.DataFrame, funds: pd.DataFrame) -> pd.DataFrame:
    # carteira equal-weighted com todos vs. só com quem sobreviveu.
    # usa todos os fundos, sem MIN_OBS: fundo que morre cedo é justamente o que interessa
    alive = funds.index[funds["alive"]].intersection(excess.columns)
    portfolios = {
        "all_funds": excess.mean(axis=1),
        "survivors_only": excess[alive].mean(axis=1),
    }
    rows = []
    for name, r in portfolios.items():
        fit = sm.OLS(r, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        rows.append({
            "portfolio": name,
            "excess_over_cdi_pct_year": r.mean() * 12 * 100,
            "alpha_pct_year": fit.params["const"] * 12 * 100,
            "t_alpha": fit.tvalues["const"],
            **{f"beta_{f}": fit.params[f] for f in FACTORS},
            "r2": fit.rsquared,
        })
    out = pd.DataFrame(rows).set_index("portfolio")
    out.loc["bias"] = out.loc["survivors_only"] - out.loc["all_funds"]
    out.loc["bias", ["t_alpha", "r2"]] = np.nan
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    excess, X, funds = load()

    alphas = fund_alphas(excess, X, funds)
    groups = {"all": alphas, **dict(tuple(alphas.groupby("group")))}
    lvs = pd.concat(
        luck_vs_skill(excess, X, g).assign(group=name) for name, g in groups.items()
    )
    surv = survivorship(excess, X, funds)

    alphas.to_csv(RESULTS / "alphas.csv", float_format="%.6g")
    lvs.to_csv(RESULTS / "luck_vs_skill.csv", index=False, float_format="%.4g")
    surv.to_csv(RESULTS / "survivorship.csv", float_format="%.4g")

    print(f"fundos com {MIN_OBS}+ meses: {len(alphas)}")
    for name, g in groups.items():
        print(f"  {name}: n={len(g)}, alpha mediano {g['alpha_pct_year'].median():+.2f}% a.a., "
              f"t>2 {(g['t_alpha'] > 2).mean():.1%}, t<-2 {(g['t_alpha'] < -2).mean():.1%}")
    print(lvs[lvs["percentile"].isin(SHOW)].round(2).to_string(index=False))
    print(surv.round(3).to_string())


if __name__ == "__main__":
    main()

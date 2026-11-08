"""Alpha of each fund, luck-vs-skill bootstrap, and survivorship bias.

Model, monthly, for each fund i (all returns net of fees):

    r_i - cdi = a_i + b1 (ibov - cdi) + b2 usd + b3 (ntnb - cdi) + e_i

usd is the plain change in USD/BRL. ntnb is the ~10y zero-coupon IPCA+ bond
(see clean.long_bond_returns), standing in for IMA-B.

Funds are split by volatility. Below LOW_VOL the fund is a "CDI+" product:
mostly private credit and structured receivables, marked to the curve, so
returns are smooth and a small spread over CDI shows up as a huge t-stat. That
spread is pay for credit and liquidity risk, not for picking trades, so the
luck-vs-skill test is run separately for the two groups.

Outputs go to results/ (small CSVs, versioned with the code):

    alphas.csv          one row per fund with >= MIN_OBS months
    luck_vs_skill.csv   cross-section of alpha t-stats vs. bootstrapped luck
    survivorship.csv    equal-weighted portfolios with and without dead funds

    uv run python -m mmalpha.alpha
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED = Path("data/processed")
RESULTS = Path("results")

FACTORS = ["ibov", "usd", "ntnb"]
LOW_VOL = 2.0  # annualized volatility of excess returns, in %
MIN_OBS = 24  # months of returns a fund needs to get its own regression
MIN_OBS_BOOT = 12  # same, inside a bootstrap draw (Fama & French 2010 use 8)
N_BOOT = 1000
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
SEED = 175


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fund excess returns (month x fund), factor matrix with a constant
    (month x k), and fund info for the funds in the sample."""
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
    """OLS of every column of Y on X at once, skipping NaNs fund by fund.

    X is T x k, Y is T x N with NaN where a fund has no return. `weights` are
    integer counts per month (a bootstrap draw is a set of months with
    repeats, which is the same as weighting each month by how often it was
    drawn). Returns coefficients and standard errors (N x k) and the number
    of observations (N); funds with fewer than `min_obs` get NaN.
    """
    T, k = X.shape
    w = np.ones(T) if weights is None else np.asarray(weights, dtype=float)

    valid = ~np.isnan(Y)
    Y0 = np.where(valid, Y, 0.0)
    W = valid * w[:, None]  # T x N

    nobs = W.sum(axis=0)
    ok = nobs >= max(min_obs, k + 1)

    XtX = np.einsum("tn,ti,tj->nij", W, X, X)
    XtY = np.einsum("tn,ti,tn->ni", W, X, Y0)
    XtX[~ok] = np.eye(k)  # keep solve() happy; these funds are masked below

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
    """Fama & French (2010) bootstrap.

    Subtract each fund's estimated alpha so that every fund has zero true
    alpha, resample months (the same months for all funds, which keeps the
    cross-correlation between funds), and re-estimate. Repeating this gives
    the distribution of t(alpha) percentiles that pure luck produces. If the
    best funds' actual t-stats beat most of these simulations, there is skill
    in the right tail; if not, the winners are what chance alone would give.
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
        t = coef[:, 0] / se[:, 0]
        sims[b] = np.nanpercentile(t, PERCENTILES)

    return pd.DataFrame({
        "percentile": PERCENTILES,
        "t_actual": actual,
        "t_luck_median": np.median(sims, axis=0),
        "pct_sims_below_actual": (sims < actual).mean(axis=0) * 100,
    })


def survivorship(excess: pd.DataFrame, X: pd.DataFrame, funds: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted portfolio of every fund alive each month vs. the same
    portfolio built only from funds that were still around at the end.

    Uses every fund in the sample, including those with less than MIN_OBS
    months: dropping short-lived funds would itself remove many of the dead.
    """
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

    print(f"Funds with >= {MIN_OBS} months: {len(alphas)} "
          f"({alphas['alive'].sum()} alive, {(~alphas['alive']).sum()} dead)")
    for name, g in groups.items():
        print(f"  {name:<12} n={len(g):>5}  median alpha {g['alpha_pct_year'].median():+.2f}% a year"
              f"  t>2 {(g['t_alpha'] > 2).mean():5.1%}  t<-2 {(g['t_alpha'] < -2).mean():5.1%}")
    print("  (by chance alone, about 2.3% of funds land on each side)")
    print("\nLuck vs skill\n", lvs.round(2).to_string(index=False))
    print("\nSurvivorship\n", surv.round(3).to_string())


if __name__ == "__main__":
    main()

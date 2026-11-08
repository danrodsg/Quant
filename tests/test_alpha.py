import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from mmalpha.alpha import batch_ols
from mmalpha.clean import long_bond_returns


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    T, N = 60, 5
    X = np.column_stack([np.ones(T), rng.normal(size=(T, 3))])
    Y = X @ rng.normal(size=(4, N)) + rng.normal(scale=0.5, size=(T, N))
    Y[:20, 1] = np.nan  # a fund that starts late
    Y[::7, 2] = np.nan  # a fund with gaps
    return X, Y


def test_matches_statsmodels_fund_by_fund(data):
    X, Y = data
    coef, se, nobs = batch_ols(X, Y, min_obs=10)
    for i in range(Y.shape[1]):
        fit = sm.OLS(Y[:, i], X, missing="drop").fit()
        np.testing.assert_allclose(coef[i], fit.params, rtol=1e-10)
        np.testing.assert_allclose(se[i], fit.bse, rtol=1e-10)
        assert nobs[i] == fit.nobs


def test_weights_equal_repeated_rows(data):
    X, Y = data
    counts = np.random.default_rng(1).integers(0, 3, len(X))
    coef, se, _ = batch_ols(X, Y, counts, min_obs=10)

    rows = np.repeat(np.arange(len(X)), counts)
    fit = sm.OLS(Y[rows, 0], X[rows]).fit()
    np.testing.assert_allclose(coef[0], fit.params, rtol=1e-10)
    np.testing.assert_allclose(se[0], fit.bse, rtol=1e-10)


def test_too_few_months_gives_nan(data):
    X, Y = data
    coef, _, _ = batch_ols(X, Y, min_obs=45)  # fund 1 only has 40 months
    assert np.isnan(coef[1]).all()
    assert not np.isnan(coef[0]).any()


def test_long_bond_picks_maturity_closest_to_target():
    near, far = pd.Timestamp("2030-05-15"), pd.Timestamp("2035-05-15")
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-31", "2025-01-31", "2025-02-28", "2025-02-28"]),
        "maturity": [near, far, near, far],
        "price": [100.0, 50.0, 101.0, 49.0],
    })
    ret = long_bond_returns(prices, target_years=10)
    assert ret[pd.Period("2025-02", "M")] == pytest.approx(49 / 50 - 1)
    ret = long_bond_returns(prices, target_years=5)
    assert ret[pd.Period("2025-02", "M")] == pytest.approx(0.01)

import numpy as np
import pandas as pd
import pytest

from mmalpha.clean import clean_quota_returns


def quotas(*values):
    idx = pd.period_range("2020-01", periods=len(values), freq="M")
    return pd.DataFrame({"f": values}, index=idx, dtype=float)


def test_normal_returns_untouched():
    ret, n_bad = clean_quota_returns(quotas(100, 101, 99.99))
    assert n_bad == 0
    assert ret["f"].iloc[1] == pytest.approx(0.01)
    assert ret["f"].iloc[2] == pytest.approx(-0.01)


def test_one_month_typo_removes_move_in_and_out():
    # 100 -> 150 -> 100.5: o 150 está errado
    ret, n_bad = clean_quota_returns(quotas(100, 100, 150, 100.5, 101))
    assert n_bad == 2
    assert ret["f"].iloc[2:4].isna().all()
    assert ret["f"].iloc[4] == pytest.approx(101 / 100.5 - 1)


def test_large_move_that_sticks_is_kept():
    # queda real de 40% não volta, fica
    ret, n_bad = clean_quota_returns(quotas(100, 60, 61))
    assert n_bad == 0
    assert ret["f"].iloc[1] == pytest.approx(-0.4)


def test_unit_change_is_removed():
    # cota passa a vir em outra unidade (x1000)
    ret, n_bad = clean_quota_returns(quotas(1.0, 1.01, 1020.0, 1030.0))
    assert n_bad == 1
    assert np.isnan(ret["f"].iloc[2])
    assert ret["f"].iloc[3] == pytest.approx(1030 / 1020 - 1)


def test_gap_gives_no_return():
    ret, _ = clean_quota_returns(quotas(100, np.nan, 102, 103))
    assert ret["f"].iloc[1:3].isna().all()

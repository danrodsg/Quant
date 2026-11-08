import io

import pandas as pd

from mmalpha.data import month_end_snapshot

# One sample per header layout the CVM has used since 2016.
PRE_2021 = """CNPJ_FUNDO;DT_COMPTC;VL_TOTAL;VL_QUOTA;VL_PATRIM_LIQ;CAPTC_DIA;RESG_DIA;NR_COTST
00.000.746/0001-03;2016-01-04;1.0;10.0;100.0;0;0;81
00.000.746/0001-03;2016-01-29;1.0;10.5;105.0;0;0;80
"""

FROM_2021 = """TP_FUNDO;CNPJ_FUNDO;DT_COMPTC;VL_TOTAL;VL_QUOTA;VL_PATRIM_LIQ;CAPTC_DIA;RESG_DIA;NR_COTST
FI;00.017.024/0001-53;2021-01-29;1.0;27.6;1000.0;0;0;1
FI;00.017.024/0001-53;2021-01-04;1.0;27.5;1000.0;0;0;1
"""

RCVM_175 = """TP_FUNDO_CLASSE;CNPJ_FUNDO_CLASSE;ID_SUBCLASSE;DT_COMPTC;VL_TOTAL;VL_QUOTA;VL_PATRIM_LIQ;CAPTC_DIA;RESG_DIA;NR_COTST
CLASSES - FIF;11.111.111/0001-11;;2025-10-30;1.0;2.0;50.0;0;0;10
CLASSES - FIF;11.111.111/0001-11;ABC123;2025-10-31;1.0;9.9;5.0;0;0;3
CLASSES - FIF;22.222.222/0001-22;;2025-10-31;1.0;0.0;0.0;0;0;0
"""


def snap(text):
    return month_end_snapshot(io.StringIO(text))


def test_same_schema_for_every_layout():
    for text in (PRE_2021, FROM_2021, RCVM_175):
        assert list(snap(text).columns) == ["cnpj", "date", "quota", "nav", "holders"]


def test_keeps_last_report_of_the_month_even_if_unsorted():
    row = snap(FROM_2021).iloc[0]
    assert row["date"] == pd.Timestamp("2021-01-29")
    assert row["quota"] == 27.6


def test_cnpj_is_digits_only():
    assert snap(PRE_2021)["cnpj"].tolist() == ["00000746000103"]


def test_subclass_rows_and_zero_quotas_are_dropped():
    out = snap(RCVM_175)
    assert out["cnpj"].tolist() == ["11111111000111"]
    assert out["quota"].iloc[0] == 2.0

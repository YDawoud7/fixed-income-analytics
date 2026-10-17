import pytest

from fixedincome.data.treasury import parse_par_curve_csv, tenor_in_years

SAMPLE = """Date,"1 Mo","1.5 Month","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"
12/31/2025,3.74,3.75,3.67,3.67,3.63,3.59,3.48,3.47,3.55,3.73,3.94,4.18,4.79,4.84
01/02/2025,4.45,,4.36,4.36,4.31,4.25,4.17,4.25,4.29,4.38,4.47,4.57,4.86,4.79
"""


@pytest.mark.parametrize(
    "label,years",
    [("1 Mo", 1 / 12), ("1.5 Month", 0.125), ("6 Mo", 0.5), ("10 Yr", 10.0), ("Date", None)],
)
def test_tenor_labels(label, years):
    assert tenor_in_years(label) == (pytest.approx(years) if years else None)


def test_parse_sorts_dates_and_converts_to_decimals():
    df = parse_par_curve_csv(SAMPLE)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2025-01-02", "2025-12-31"]
    assert df.columns[0] == pytest.approx(1 / 12)
    assert df.loc["2025-12-31", 10.0] == pytest.approx(0.0418)
    assert df.loc["2025-01-02"].isna().sum() == 1  # 1.5-month bill not yet issued

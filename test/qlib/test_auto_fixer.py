"""Tests for FactorAutoFixer — the pre-execution code patcher."""

import pytest

from rdagent.components.coder.factor_coder.auto_fixer import FactorAutoFixer


@pytest.fixture()
def fixer():
    return FactorAutoFixer()


class TestResetIndexGroupby:
    def test_replaces_level_groupby_on_reset_var(self, fixer):
        code = "df_r = df.reset_index()\ndf_r['x'] = df_r.groupby(level=1)['$close'].mean()"
        result = fixer.fix(code)
        assert "groupby('instrument')" in result

    def test_does_not_touch_normal_multiindex_groupby(self, fixer):
        code = "df['x'] = df.groupby(level=1)['$close'].mean()"
        result = fixer.fix(code)
        assert "groupby(level=1)" in result


class TestGroupbyMixedLevels:
    def test_strips_string_from_mixed_list(self, fixer):
        result = fixer.fix("df.groupby(level=[1, 'date']).apply(fn)")
        assert "groupby(level=1)" in result

    def test_multiple_ints_kept(self, fixer):
        result = fixer.fix("df.groupby(level=[0, 1, 'x']).apply(fn)")
        assert "groupby(level=[0, 1])" in result


class TestGroupbyColumnOnMultiindex:
    def test_instrument_date_becomes_two_level(self, fixer):
        code = "df['v'] = df.groupby(['instrument', 'date'])['$volume'].cumsum()"
        result = fixer.fix(code)
        assert "get_level_values(1)" in result
        assert "normalize()" in result
        assert "level=1)" not in result.split("get_level_values")[0]

    def test_date_instrument_becomes_two_level(self, fixer):
        code = "df['v'] = df.groupby(['date', 'instrument'])['$volume'].cumsum()"
        result = fixer.fix(code)
        assert "get_level_values(0).normalize()" in result
        assert "get_level_values(1)" in result

    def test_single_instrument_becomes_level1(self, fixer):
        result = fixer.fix("df.groupby(['instrument'])['x'].mean()")
        assert "groupby(level=1)" in result

    def test_reset_index_not_double_fixed(self, fixer):
        # After reset_index fix emits groupby('instrument'), this fixer must NOT
        # convert that to groupby(level=1).
        code = "df_r = df.reset_index()\ndf_r['x'] = df_r.groupby(level=1)['p'].mean()"
        result = fixer.fix(code)
        assert "groupby('instrument')" in result


class TestChainedGroupby:
    def test_chained_groupby_level_then_date(self, fixer):
        code = "df.groupby(level=1).groupby('date')['price_volume'].transform('cumsum')"
        result = fixer.fix(code)
        assert "get_level_values(1)" in result
        assert "get_level_values(0).normalize()" in result
        assert ".groupby('date')" not in result

    def test_chained_groupby_with_double_quotes(self, fixer):
        code = 'df.groupby(level=0).groupby("date")["col"].sum()'
        result = fixer.fix(code)
        assert "get_level_values" in result
        assert '.groupby("date")' not in result


class TestMinPeriodsNotTouched:
    def test_small_min_periods_preserved(self, fixer):
        # _fix_min_periods is disabled — LLM-set min_periods must not be changed.
        # window=60, min_periods=1 should stay as-is (was wrongly raised to 60 before).
        result = fixer.fix("df.groupby(level=1)['x'].transform(lambda x: x.rolling(window=60, min_periods=1).mean())")
        assert "min_periods=1" in result

    def test_large_window_min_periods_preserved(self, fixer):
        # window=240 > 96 bars/day: if min_periods were set to 240 the output would be
        # all-NaN for intraday data. Verify we leave it untouched.
        result = fixer.fix("df['x'] = df.groupby(level=1)['y'].transform(lambda x: x.rolling(240, min_periods=10).std())")
        assert "min_periods=10" in result


class TestRollingDdof:
    def test_removes_ddof_from_rolling_args(self, fixer):
        result = fixer.fix("df.rolling(20, min_periods=1, ddof=1).std()")
        assert "ddof" not in result

    def test_removes_ddof_from_std_args(self, fixer):
        result = fixer.fix("df.rolling(20).std(ddof=1)")
        assert "ddof" not in result

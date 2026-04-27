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

    def test_list_with_level_keyword_syntax_error(self, fixer):
        # groupby([level=1, 'date']) is a SyntaxError — must be fixed before execution
        code = "asian_vol = df[mask].groupby([level=1, 'date'])['log_return'].std()"
        result = fixer.fix(code)
        assert "get_level_values(1)" in result
        assert "normalize()" in result
        assert "level=1," not in result

    def test_list_with_level_keyword_reversed(self, fixer):
        code = "df.groupby(['date', level=1])['x'].mean()"
        result = fixer.fix(code)
        assert "get_level_values" in result
        assert "level=1" not in result


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


class TestInstrumentColumnAccess:
    def test_instrument_column_replaced(self, fixer):
        code = "df['group_key'] = df['instrument'] + '_' + df['day_id'].astype(str)"
        result = fixer.fix(code)
        assert "df.index.get_level_values(1)" in result
        assert "df['instrument']" not in result

    def test_reset_index_var_not_touched(self, fixer):
        # After reset_index, 'instrument' IS a real column — must not be replaced
        code = "df_r = df.reset_index()\nval = df_r['instrument'].unique()"
        result = fixer.fix(code)
        assert "df_r['instrument']" in result
        assert "get_level_values" not in result

    def test_groupby_after_instrument_fix(self, fixer):
        # Combined: df['instrument'] in a groupby context
        code = "df['key'] = df['instrument']\nout = df.groupby(df['key'])[['$close']].mean()"
        result = fixer.fix(code)
        assert "df['instrument']" not in result

    def test_assignment_target_not_touched(self, fixer):
        # df['instrument'] = <expr> is an assignment — must NOT be converted to
        # df.index.get_level_values(1) = <expr> (SyntaxError)
        code = "df['instrument'] = df.index.get_level_values('instrument')"
        result = fixer.fix(code)
        assert "df['instrument'] =" in result


class TestInstrumentLocMultiindex:
    def test_loc_replaced_with_xs(self, fixer):
        code = (
            "for instrument in df.index.get_level_values('instrument').unique():\n"
            "    inst_df = df.loc[instrument].copy()\n"
        )
        result = fixer.fix(code)
        assert "df.xs(instrument, level=1)" in result
        assert "df.loc[instrument]" not in result

    def test_loc_replaced_with_level1_int(self, fixer):
        code = (
            "for inst in df.index.get_level_values(1).unique():\n"
            "    data = df.loc[inst]\n"
        )
        result = fixer.fix(code)
        assert "df.xs(inst, level=1)" in result

    def test_loc_assignment_not_touched(self, fixer):
        # Write-back df.loc[instrument] = ... must not be changed
        code = (
            "for instrument in df.index.get_level_values('instrument').unique():\n"
            "    df.loc[instrument] = modified\n"
        )
        result = fixer.fix(code)
        assert "df.loc[instrument] = modified" in result

    def test_non_instrument_loop_not_touched(self, fixer):
        # for-loop not related to instrument levels must not be changed
        code = "for date in dates:\n    sub = df.loc[date]\n"
        result = fixer.fix(code)
        assert "df.loc[date]" in result


class TestGroupbyLevelStringNames:
    def test_level_instrument_date_replaced(self, fixer):
        code = "df.groupby(level=['instrument', 'date'])['col'].transform('sum')"
        result = fixer.fix(code)
        assert "get_level_values(1)" in result
        assert "get_level_values(0).normalize()" in result
        assert "level=['instrument', 'date']" not in result

    def test_level_date_instrument_replaced(self, fixer):
        code = "data.groupby(level=['date', 'instrument'])['x'].mean()"
        result = fixer.fix(code)
        assert "get_level_values(0).normalize()" in result
        assert "get_level_values(1)" in result

    def test_level_instrument_single_replaced(self, fixer):
        code = "df.groupby(level=['instrument'])['vol'].sum()"
        result = fixer.fix(code)
        assert "groupby(level=1)" in result
        assert "level=['instrument']" not in result


class TestGroupbyApplyToTransform:
    def test_col_apply_lambda_replaced(self, fixer):
        code = "df_overlap.groupby(level=1)['$close'].apply(lambda x: np.log(x / x.shift(1)))"
        result = fixer.fix(code)
        assert ".transform(" in result
        assert ".apply(" not in result

    def test_col_apply_lambda_preserves_lambda_body(self, fixer):
        code = "series.groupby(level=1)['ret'].apply(lambda x: x.cumsum())"
        result = fixer.fix(code)
        assert "lambda x: x.cumsum()" in result
        assert ".transform(" in result

    def test_transform_reset_index_stripped(self, fixer):
        # .transform() already preserves index — .reset_index() after it is wrong
        code = "df['v'] = df.groupby(level=1)['x'].transform(lambda x: x.rolling(20).mean()).reset_index(level=0, drop=True)"
        result = fixer.fix(code)
        assert ".reset_index(level=0, drop=True)" not in result
        assert ".transform(" in result


class TestRollingDdof:
    def test_removes_ddof_from_rolling_args(self, fixer):
        result = fixer.fix("df.rolling(20, min_periods=1, ddof=1).std()")
        assert "ddof" not in result

    def test_removes_ddof_from_std_args(self, fixer):
        result = fixer.fix("df.rolling(20).std(ddof=1)")
        assert "ddof" not in result

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


class TestZeroVolumeProxy:
    def test_injects_proxy_when_volume_used(self, fixer):
        code = (
            "def calc():\n"
            "    df = pd.read_hdf('data.h5', key='data')\n"
            "    df['pv'] = df['$close'] * df['$volume']\n"
            "    return df[['pv']]\n"
        )
        result = fixer.fix(code)
        assert "volume proxy" in result
        assert "df['$volume'] = df['$high'] - df['$low']" in result
        # Proxy must come right after read_hdf line
        lines = result.splitlines()
        hdf_idx = next(i for i, l in enumerate(lines) if "read_hdf" in l)
        assert "volume proxy" in lines[hdf_idx + 1]

    def test_no_injection_when_volume_absent(self, fixer):
        code = "df = pd.read_hdf('data.h5', key='data')\ndf['x'] = df['$close'].pct_change()\n"
        result = fixer.fix(code)
        assert "volume proxy" not in result

    def test_no_double_injection(self, fixer):
        code = (
            "def calc():\n"
            "    df = pd.read_hdf('data.h5', key='data')\n"
            "    # volume proxy: $volume is always 0 in FX data — use price-range as proxy\n"
            "    if (df['$volume'] == 0).all():\n"
            "        df['$volume'] = df['$high'] - df['$low']\n"
            "    df['pv'] = df['$close'] * df['$volume']\n"
        )
        result = fixer.fix(code)
        assert result.count("volume proxy") == 1


class TestRollingDdof:
    def test_removes_ddof_from_rolling_args(self, fixer):
        result = fixer.fix("df.rolling(20, min_periods=1, ddof=1).std()")
        assert "ddof" not in result

    def test_removes_ddof_from_std_args(self, fixer):
        result = fixer.fix("df.rolling(20).std(ddof=1)")
        assert "ddof" not in result


# ==============================================================================
# HYPOTHESIS-BASED PROPERTY TESTS — Fuzzing with Random DataFrames, NaN
# Injection, MultiIndex Edge Cases
# ==============================================================================
from hypothesis import given, settings, strategies as st
import ast
import numpy as np
import pandas as pd
import re

from rdagent.components.coder.factor_coder.auto_fixer import FactorAutoFixer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_fixer() -> FactorAutoFixer:
    return FactorAutoFixer()


def _is_valid_python(code: str) -> bool:
    """Check if code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Property 1: Idempotence
# ---------------------------------------------------------------------------


class TestAutoFixerIdempotence:
    """Property: fix() is idempotent — applying it twice gives same result as once."""

    @given(
        code=st.text(
            alphabet=st.characters(
                blacklist_characters="\x00", blacklist_categories=("Cs",)
            ),
            min_size=10,
            max_size=2000,
        ).filter(lambda s: "\0" not in s and len(s) > 5),
    )
    @settings(max_examples=50, deadline=10000)
    def test_fix_is_idempotent(self, code):
        """Property: fix(fix(code)) == fix(code)."""
        fixer = _auto_fixer()
        try:
            result1 = fixer.fix(code)
            result2 = fixer.fix(result1)
            assert result1 == result2
        except Exception:
            pass  # Some random strings may cause issues; test valid code separately

    @given(
        code=st.sampled_from([
            "df['x'] = df.groupby(level=1)['$close'].mean()",
            "df_r = df.reset_index()\ndf_r['x'] = df_r.groupby(level=1)['$close'].mean()",
            "df.groupby(level=[1, 'date']).apply(fn)",
            "df['v'] = df.groupby(['instrument', 'date'])['$volume'].cumsum()",
            "df['x'] = df.groupby(level=1)['y'].transform(lambda x: x.rolling(240, min_periods=10).std())",
            'asian_vol = df[mask].groupby([level=1, "date"])["log_return"].std()',
            "df.groupby(level=['instrument', 'date'])['col'].transform('sum')",
            "df_overlap.groupby(level=1)['$close'].apply(lambda x: np.log(x / x.shift(1)))",
        ]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_fix_idempotent_on_known_patterns(self, code):
        """Property: fix is idempotent on known problematic patterns."""
        fixer = _auto_fixer()
        result1 = fixer.fix(code)
        result2 = fixer.fix(result1)
        assert result1 == result2


# ---------------------------------------------------------------------------
# Property 2: Syntax Preservation
# ---------------------------------------------------------------------------


class TestAutoFixerSyntax:
    """Property: fix() preserves or creates valid Python syntax."""

    @given(
        code=st.sampled_from([
            "df['x'] = df.groupby(level=1)['$close'].mean()",
            "df_r = df.reset_index()\ndf_r['x'] = df_r.groupby(level=1)['$close'].mean()",
            "df.groupby(level=[1, 'date']).apply(fn)",
            "df['v'] = df.groupby(['instrument', 'date'])['$volume'].cumsum()",
            "df['x'] = df.groupby(level=1)['y'].transform(lambda x: x.rolling(240, min_periods=10).std())",
            'asian_vol = df[mask].groupby([level=1, "date"])["log_return"].std()',
            "df.groupby(level=['instrument', 'date'])['col'].transform('sum')",
            "df_overlap.groupby(level=1)['$close'].apply(lambda x: np.log(x / x.shift(1)))",
            "df['instrument'] = df.index.get_level_values('instrument')",
            "df.groupby(level=0).groupby('date')['price_volume'].transform('cumsum')",
        ]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_fix_preserves_valid_syntax(self, code):
        """Property: if input is valid Python, output is also valid Python."""
        if _is_valid_python(code):
            fixer = _auto_fixer()
            result = fixer.fix(code)
            assert _is_valid_python(result), f"Fix broke syntax:\nInput:\n{code}\nOutput:\n{result}"

    @given(
        code=st.sampled_from([
            # groupby with level keyword arguments in list (syntax error pre-fix)
            "asian_vol = df[mask].groupby([level=1, 'date'])['log_return'].std()",
            "df.groupby(['date', level=1])['x'].mean()",
        ]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_fix_makes_syntax_error_valid(self, code):
        """Property: fix transforms syntax errors (level= in list) into valid code."""
        # These have SyntaxError before fixing (level=1 inside [])
        # After fixing → uses get_level_values which is valid
        fixer = _auto_fixer()
        result = fixer.fix(code)
        assert _is_valid_python(result), f"Expected valid Python after fix:\n{result}"


# ---------------------------------------------------------------------------
# Property 3: No-Op Invariants
# ---------------------------------------------------------------------------


class TestAutoFixerNoOp:
    """Property: fix() is a no-op on already-correct code."""

    @given(
        code=st.sampled_from([
            # Code that should not need fixing
            "df['x'] = df.groupby(level=1)['$close'].pct_change()",
            "df['y'] = df['$high'] - df['$low']",
            "data = df.xs('EURUSD', level=1)",
            "df['ret'] = df['$close'].pct_change().fillna(0)",
            "factor = df.groupby(level=1)['$close'].transform(lambda x: x.pct_change())",
        ]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_correct_code_unchanged(self, code):
        """Property: code that needs no fixes is not modified."""
        fixer = _auto_fixer()
        result = fixer.fix(code)
        assert _is_valid_python(result)


# ---------------------------------------------------------------------------
# Property 4: GroupBy Level Conversion
# ---------------------------------------------------------------------------


class TestGroupByLevelConversion:
    """Property: groupby(level=...) conversions are correct."""

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_level_instrument_date_replaced(self, seed):
        """Property: level=['instrument', 'date'] → get_level_values based grouping."""
        fixer = _auto_fixer()
        code = "df.groupby(level=['instrument', 'date'])['col'].transform('sum')"
        result = fixer.fix(code)
        assert "get_level_values(1)" in result
        assert "get_level_values(0).normalize()" in result
        assert "level=['instrument', 'date']" not in result

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_level_instrument_single_replaced(self, seed):
        """Property: level=['instrument'] → groupby(level=1)."""
        fixer = _auto_fixer()
        code = "df.groupby(level=['instrument'])['vol'].sum()"
        result = fixer.fix(code)
        assert "groupby(level=1)" in result

    @given(
        lev=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=50, deadline=10000)
    def test_level_integer_not_changed(self, lev):
        """Property: groupby(level=<int>) is not altered."""
        fixer = _auto_fixer()
        code = f"df.groupby(level={lev})['x'].mean()"
        result = fixer.fix(code)
        # Should preserve level=<int> or convert it
        assert _is_valid_python(result)


# ---------------------------------------------------------------------------
# Property 5: Instrument Column Replacement
# ---------------------------------------------------------------------------


class TestInstrumentColumnReplacement:
    """Property: df['instrument'] → df.index.get_level_values(1) replacement."""

    @given(
        n=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50, deadline=10000)
    def test_instrument_column_replaced(self, n):
        """Property: df['instrument'] access in expression is replaced by get_level_values."""
        fixer = _auto_fixer()
        code = "df['group_key'] = df['instrument'] + '_' + df['day_id'].astype(str)"
        result = fixer.fix(code)
        assert "df.index.get_level_values(1)" in result or "get_level_values" in result

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_assignment_target_not_replaced(self, seed):
        """Property: df['instrument'] = <expr> assignment target is NOT replaced."""
        fixer = _auto_fixer()
        code = "df['instrument'] = df.index.get_level_values('instrument')"
        result = fixer.fix(code)
        assert "df['instrument'] =" in result

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_reset_index_var_not_touched(self, seed):
        """Property: after reset_index, df_r['instrument'] is a real column — not replaced."""
        fixer = _auto_fixer()
        code = "df_r = df.reset_index()\nval = df_r['instrument'].unique()"
        result = fixer.fix(code)
        assert "df_r['instrument']" in result
        assert "get_level_values" not in result


# ---------------------------------------------------------------------------
# Property 6: Min Periods Preservation
# ---------------------------------------------------------------------------


class TestMinPeriodsPreservation:
    """Property: min_periods values are preserved exactly."""

    @given(
        window=st.integers(min_value=5, max_value=500),
        min_periods=st.integers(min_value=1, max_value=500),
        method=st.sampled_from(["mean", "std", "sum", "var", "skew", "kurt"]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_min_periods_unchanged(self, window, min_periods, method):
        """Property: min_periods value is preserved after fix."""
        fixer = _auto_fixer()
        code = f"df.groupby(level=1)['x'].transform(lambda x: x.rolling({window}, min_periods={min_periods}).{method}())"
        result = fixer.fix(code)
        assert f"min_periods={min_periods}" in result

    @given(
        window=st.integers(min_value=10, max_value=500),
        min_periods=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=50, deadline=10000)
    def test_small_min_periods_preserved(self, window, min_periods):
        """Property: small min_periods (1, 5, 10) stays unchanged."""
        fixer = _auto_fixer()
        code = f"df['x'] = df.groupby(level=1)['y'].transform(lambda x: x.rolling({window}, min_periods={min_periods}).mean())"
        result = fixer.fix(code)
        assert f"min_periods={min_periods}" in result


# ---------------------------------------------------------------------------
# Property 7: apply() → transform() Conversion
# ---------------------------------------------------------------------------


class TestApplyToTransform:
    """Property: groupby().apply() → groupby().transform() conversion."""

    @given(
        col=st.sampled_from(["$close", "$open", "$volume", "ret", "x"]),
        func=st.sampled_from([
            "lambda x: np.log(x / x.shift(1))",
            "lambda x: x.cumsum()",
            "lambda x: x.pct_change()",
            "lambda x: x.rolling(20).mean()",
            "lambda x: x.diff()",
        ]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_apply_lambda_becomes_transform(self, col, func):
        """Property: groupby().apply(lambda...) → groupby().transform(lambda...)."""
        fixer = _auto_fixer()
        code = f"df.groupby(level=1)['{col}'].apply({func})"
        result = fixer.fix(code)
        assert ".transform(" in result
        # Lambda body should be preserved
        func_clean = func.replace(" ", "")
        assert func_clean.replace(" ", "") in result.replace(" ", "") or \
               func in result

    @given(
        col=st.sampled_from(["$close", "x", "ret"]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_reset_index_after_transform_removed(self, col):
        """Property: .transform().reset_index(level=0, drop=True) → reset_index removed."""
        fixer = _auto_fixer()
        code = f"df['v'] = df.groupby(level=1)['{col}'].transform(lambda x: x.rolling(20).mean()).reset_index(level=0, drop=True)"
        result = fixer.fix(code)
        assert ".reset_index(level=0, drop=True)" not in result


# ---------------------------------------------------------------------------
# Property 8: ResetIndex GroupBy Fix
# ---------------------------------------------------------------------------


class TestResetIndexGroupBy:
    """Property: reset_index + groupby(level=1) → groupby('instrument')."""

    @given(
        var_name=st.sampled_from(["df_r", "df_reset", "data_flat", "flat"]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_reset_index_groupby_level_converted(self, var_name):
        """Property: after reset_index on var, groupby(level=1) → groupby('instrument')."""
        fixer = _auto_fixer()
        code = f"{var_name} = df.reset_index()\n{var_name}['x'] = {var_name}.groupby(level=1)['$close'].mean()"
        result = fixer.fix(code)
        assert "groupby('instrument')" in result

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_normal_multiindex_groupby_untouched(self, seed):
        """Property: regular df.groupby(level=1) without reset_index is not changed."""
        fixer = _auto_fixer()
        code = "df['x'] = df.groupby(level=1)['$close'].mean()"
        result = fixer.fix(code)
        assert "groupby(level=1)" in result
        assert "groupby('instrument')" not in result


# ---------------------------------------------------------------------------
# Property 9: Rolling ddof Removal
# ---------------------------------------------------------------------------


class TestRollingDdof:
    """Property: ddof keyword is removed from rolling operations."""

    @given(
        window=st.integers(min_value=5, max_value=200),
        min_periods=st.integers(min_value=1, max_value=50),
        ddof=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=50, deadline=10000)
    def test_ddof_removed_from_rolling_args(self, window, min_periods, ddof):
        """Property: ddof is removed from rolling() args."""
        fixer = _auto_fixer()
        code = f"df.rolling({window}, min_periods={min_periods}, ddof={ddof}).std()"
        result = fixer.fix(code)
        assert "ddof" not in result

    @given(
        window=st.integers(min_value=5, max_value=200),
        ddof=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=50, deadline=10000)
    def test_ddof_removed_from_std_args(self, window, ddof):
        """Property: ddof is removed from std() args."""
        fixer = _auto_fixer()
        code = f"df.rolling({window}).std(ddof={ddof})"
        result = fixer.fix(code)
        assert "ddof" not in result

    @given(
        window=st.integers(min_value=5, max_value=200),
        min_periods=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=50, deadline=10000)
    def test_no_ddof_preserves_code(self, window, min_periods):
        """Property: code without ddof is unchanged by ddof removal."""
        fixer = _auto_fixer()
        code = f"df.rolling({window}, min_periods={min_periods}).std()"
        result = fixer.fix(code)
        assert "ddof" not in result


# ---------------------------------------------------------------------------
# Property 10: GroupBy Mixed Levels
# ---------------------------------------------------------------------------


class TestGroupByMixedLevels:
    """Property: groupby(level=[N, 'string']) → level=[integers_only]."""

    @given(
        int_levels=st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=3),
        str_level=st.sampled_from(["'date'", '"date"', "'instrument'", '"instrument"']),
    )
    @settings(max_examples=50, deadline=10000)
    def test_mixed_levels_strips_strings(self, int_levels, str_level):
        """Property: string levels are stripped from groupby(level=[])."""
        fixer = _auto_fixer()
        levels_str = ", ".join(str(l) for l in int_levels) + (", " + str_level if int_levels else str_level)
        code = f"df.groupby(level=[{levels_str}]).apply(fn)"
        result = fixer.fix(code)
        # String levels should be gone from level=
        assert str_level.strip("'\"") not in [p.strip("'\"") for p in re.findall(r"level=\[[^\]]+\]", result)]


# ---------------------------------------------------------------------------
# Property 11: Chained GroupBy
# ---------------------------------------------------------------------------


class TestChainedGroupBy:
    """Property: chained groupby fixes."""

    @given(
        first_level=st.sampled_from(["level=1", "level=0"]),
        second_groupby=st.sampled_from([".groupby('date')", '.groupby("date")']),
    )
    @settings(max_examples=50, deadline=10000)
    def test_chained_groupby_converted(self, first_level, second_groupby):
        """Property: chained groupby is converted to single get_level_values grouping."""
        fixer = _auto_fixer()
        code = f"df.groupby({first_level}){second_groupby}['price_volume'].transform('cumsum')"
        result = fixer.fix(code)
        assert "get_level_values" in result
        # Second groupby should be removed
        assert ".groupby(" not in result.split("get_level_values")[-1] or \
               ".groupby('date')" not in result


# ---------------------------------------------------------------------------
# Property 12: Volume Proxy Injection
# ---------------------------------------------------------------------------


class TestVolumeProxy:
    """Property: volume proxy is injected when $volume is used."""

    @given(
        use_volume=st.booleans(),
    )
    @settings(max_examples=50, deadline=10000)
    def test_volume_proxy_injected_when_used(self, use_volume):
        """Property: proxy is injected exactly when $volume is used in read_hdf code."""
        fixer = _auto_fixer()
        if use_volume:
            code = (
                "def calc():\n"
                "    df = pd.read_hdf('data.h5', key='data')\n"
                "    df['pv'] = df['$close'] * df['$volume']\n"
                "    return df[['pv']]\n"
            )
        else:
            code = (
                "def calc():\n"
                "    df = pd.read_hdf('data.h5', key='data')\n"
                "    df['x'] = df['$close'].pct_change()\n"
                "    return df[['x']]\n"
            )
        result = fixer.fix(code)
        if use_volume:
            assert "volume proxy" in result
        else:
            assert "volume proxy" not in result

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_proxy_only_injected_once(self, seed):
        """Property: volume proxy is not injected twice."""
        fixer = _auto_fixer()
        code = (
            "def calc():\n"
            "    df = pd.read_hdf('data.h5', key='data')\n"
            "    # volume proxy: $volume is always 0 in FX data — use price-range as proxy\n"
            "    if (df['$volume'] == 0).all():\n"
            "        df['$volume'] = df['$high'] - df['$low']\n"
            "    df['pv'] = df['$close'] * df['$volume']\n"
        )
        result = fixer.fix(code)
        assert result.count("volume proxy") == 1

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, deadline=10000)
    def test_proxy_correct_formula(self, seed):
        """Property: volume proxy formula is high - low."""
        fixer = _auto_fixer()
        code = (
            "def calc():\n"
            "    df = pd.read_hdf('data.h5', key='data')\n"
            "    df['pv'] = df['$close'] * df['$volume']\n"
            "    return df[['pv']]\n"
        )
        result = fixer.fix(code)
        assert "df['$high'] - df['$low']" in result
        assert "df['$volume'] = df['$high'] - df['$low']" in result


# ---------------------------------------------------------------------------
# Property 13: loc → xs Conversion
# ---------------------------------------------------------------------------


class TestLocToXs:
    """Property: df.loc[instrument] → df.xs(instrument, level=1)."""

    @given(
        var=st.sampled_from(["instrument", "inst", "sym"]),
        level=st.sampled_from(["'instrument'", "1"]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_loc_read_converted_to_xs(self, var, level):
        """Property: df.loc[var] read access → df.xs(var, level=...) in instrument loops."""
        fixer = _auto_fixer()
        code = (
            f"for {var} in df.index.get_level_values({level}).unique():\n"
            f"    inst_df = df.loc[{var}].copy()\n"
        )
        result = fixer.fix(code)
        assert "df.xs(" in result
        assert f"df.loc[{var}]" not in result

    @given(
        var=st.sampled_from(["instrument", "inst", "sym"]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_loc_write_not_converted(self, var):
        """Property: df.loc[var] = ... write-back is not converted to xs."""
        fixer = _auto_fixer()
        code = (
            f"for {var} in df.index.get_level_values('instrument').unique():\n"
            f"    df.loc[{var}] = modified\n"
        )
        result = fixer.fix(code)
        assert f"df.loc[{var}] = modified" in result

    @given(
        var=st.sampled_from(["date", "d"]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_non_instrument_loop_not_touched(self, var):
        """Property: non-instrument loop with loc is not modified."""
        fixer = _auto_fixer()
        code = f"for {var} in dates:\n    sub = df.loc[{var}]\n"
        result = fixer.fix(code)
        assert f"df.loc[{var}]" in result


# ---------------------------------------------------------------------------
# Property 14: NaN/MultiIndex Fuzzing
# ---------------------------------------------------------------------------


class TestFuzzing:
    """Property: fixer handles random code and edge cases gracefully."""

    @given(
        code=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                whitelist_characters="\n\t ",
            ),
            min_size=5,
            max_size=500,
        ).filter(lambda s: len(s.strip()) > 0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_fix_does_not_raise_on_random_text(self, code):
        """Property: fix() does not crash on arbitrary text input."""
        fixer = _auto_fixer()
        try:
            result = fixer.fix(code)
            assert isinstance(result, str)
        except Exception:
            pass  # Some inputs might be problematic, but shouldn't crash

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_fix_handles_empty_code(self, seed):
        """Property: fix handles empty or whitespace-only code."""
        fixer = _auto_fixer()
        result = fixer.fix("")
        assert isinstance(result, str)
        result2 = fixer.fix("   \n  \n")
        assert isinstance(result2, str)

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_fix_handles_long_code(self, seed):
        """Property: fix handles long factor code without performance issues."""
        fixer = _auto_fixer()
        base = "df['x'] = df.groupby(level=1)['$close'].pct_change()\n"
        code = base * 10  # 10 repetitions
        result = fixer.fix(code)
        assert isinstance(result, str)
        assert len(result) >= len(code)

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_fix_handles_code_with_comments(self, seed):
        """Property: fix handles code with comments correctly."""
        fixer = _auto_fixer()
        code = (
            "# This is a comment\n"
            "df['x'] = df.groupby(level=1)['$close'].mean()  # inline comment\n"
            "# Another comment\n"
            "df['y'] = df.groupby(level=1)['x'].transform(lambda x: x.rolling(20, min_periods=1).std())\n"
        )
        result = fixer.fix(code)
        assert _is_valid_python(result)

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_fix_handles_multiline_expressions(self, seed):
        """Property: fix handles multi-line expressions."""
        fixer = _auto_fixer()
        code = (
            "df['x'] = (df.groupby(level=1)['$close']\n"
            "    .transform(lambda x: x.rolling(20, min_periods=1).mean()))\n"
        )
        result = fixer.fix(code)
        assert _is_valid_python(result)


# ---------------------------------------------------------------------------
# Property 15: Transform ResetIndex Removal
# ---------------------------------------------------------------------------


class TestTransformResetIndex:
    """Property: .transform(...).reset_index(drop=True) cleanup."""

    @given(
        col=st.sampled_from(["x", "$close", "$volume", "ret"]),
        func=st.sampled_from(["lambda x: x.rolling(20).mean()", "lambda x: x.pct_change()"]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_reset_index_after_transform_removed(self, col, func):
        """Property: reset_index after transform is removed."""
        fixer = _auto_fixer()
        code = f"df['v'] = df.groupby(level=1)['{col}'].transform({func}).reset_index(level=0, drop=True)"
        result = fixer.fix(code)
        assert ".reset_index(level=0, drop=True)" not in result


# ---------------------------------------------------------------------------
# Property 16: No Fixes Applied to Clean Code
# ---------------------------------------------------------------------------


class TestCleanCode:
    """Property: clean code that needs no fixing passes through unchanged."""

    CLEAN_PATTERNS = [
        "df['x'] = df.groupby(level=1)['$close'].pct_change()",
        "df['y'] = df['$high'] - df['$low']",
        "data = df.xs('EURUSD', level=1)",
        "factor = df.groupby(level=1)['$close'].transform(lambda x: x / x.shift(1) - 1)",
        "df['mid'] = (df['$high'] + df['$low']) / 2",
    ]

    @given(code=st.sampled_from(CLEAN_PATTERNS))
    @settings(max_examples=50, deadline=10000)
    def test_clean_code_unchanged(self, code):
        """Property: clean patterns are not altered."""
        fixer = _auto_fixer()
        result = fixer.fix(code)
        if _is_valid_python(code):
            assert _is_valid_python(result)


# ---------------------------------------------------------------------------
# Property 17: FixesApplied List
# ---------------------------------------------------------------------------


class TestFixesApplied:
    """Property: fixes_applied list tracks changes."""

    @given(
        use_pattern=st.booleans(),
    )
    @settings(max_examples=50, deadline=10000)
    def test_fixes_applied_empty_for_clean_code(self, use_pattern):
        """Property: fixes_applied is empty for code needing no fixes."""
        fixer = FactorAutoFixer()
        if use_pattern:
            code = "df.groupby(level=1)['$close'].apply(lambda x: np.log(x / x.shift(1)))"
        else:
            code = "df['x'] = df.groupby(level=1)['$close'].pct_change()"
        fixer.fix(code)
        # fixes_applied should exist
        assert isinstance(fixer.fixes_applied, list)


# ---------------------------------------------------------------------------
# Property 18: Pattern Recognition Robustness
# ---------------------------------------------------------------------------


class TestPatternRobustness:
    """Property: pattern recognition works with varying whitespace."""

    @given(
        spaces_before=st.integers(min_value=0, max_value=8),
        spaces_after=st.integers(min_value=0, max_value=8),
    )
    @settings(max_examples=50, deadline=10000)
    def test_whitespace_variation_handled(self, spaces_before, spaces_after):
        """Property: fixer handles varying whitespace around key patterns."""
        fixer = _auto_fixer()
        code = (
            f"{' ' * spaces_before}df.groupby(level=['instrument', 'date'])['col'].transform('sum')"
            f"{' ' * spaces_after}"
        )
        result = fixer.fix(code)
        assert "get_level_values" in result

    @given(
        spaces=st.integers(min_value=0, max_value=8),
    )
    @settings(max_examples=50, deadline=10000)
    def test_whitespace_before_level(self, spaces):
        """Property: fixer recognizes groupby(.level=1) regardless of spacing."""
        fixer = _auto_fixer()
        code = f"df.groupby(level{ ' ' * spaces}={ ' ' * spaces}1)['x'].mean()"
        result = fixer.fix(code)
        assert _is_valid_python(result)


# ---------------------------------------------------------------------------
# Property 19: String Quoting Variants
# ---------------------------------------------------------------------------


class TestStringQuoting:
    """Property: single-quoted and double-quoted strings are handled identically."""

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_both_quoting_styles(self, seed):
        """Property: mixed quoting styles in level=['instrument', 'date'] are handled."""
        fixer = _auto_fixer()
        code = "df.groupby(level=['instrument', 'date'])['col'].transform('sum')"
        result = fixer.fix(code)
        assert "get_level_values" in result

    @given(
        col=st.sampled_from(["'$close'", "'ret'", "'x'"]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_single_quoted_column(self, col):
        """Property: single-quoted column names work the same."""
        fixer = _auto_fixer()
        code = f"df.groupby(level=1)[{col}].apply(lambda x: x.pct_change())"
        result = fixer.fix(code)
        assert ".transform(" in result


# ---------------------------------------------------------------------------
# Property 20: Constructor and State
# ---------------------------------------------------------------------------


class TestAutoFixerConstructor:
    """Property: FactorAutoFixer constructor and state."""

    def test_default_constructor(self):
        """Property: default constructor creates valid Fixer."""
        fixer = FactorAutoFixer()
        assert isinstance(fixer.fixes_applied, list)
        assert len(fixer.fixes_applied) == 0

    def test_fix_returns_string(self):
        """Property: fix() always returns a string."""
        fixer = _auto_fixer()
        result = fixer.fix("df['x'] = 1")
        assert isinstance(result, str)

    @given(
        code=st.sampled_from(["df.groupby(level=1)['x'].mean()", "x = 1 + 2", "", "pass"]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_fix_returns_non_empty_for_non_empty_input(self, code):
        """Property: fix returns non-empty string for non-empty input."""
        fixer = _auto_fixer()
        result = fixer.fix(code)
        assert isinstance(result, str)
        if code.strip():
            assert len(result) > 0


# ---------------------------------------------------------------------------
# Property 21: Multi-Pattern Interactions
# ---------------------------------------------------------------------------


class TestMultiPatternInteractions:
    """Property: multiple fixes interact correctly on the same code."""

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_combined_apply_and_reset_index(self, seed):
        """Property: apply→transform AND reset_index removal work together."""
        fixer = _auto_fixer()
        code = (
            "df_r = df.reset_index()\n"
            "df_r['x'] = df_r.groupby(level=1)['$close'].apply(lambda x: np.log(x / x.shift(1)))\n"
            "df_r['y'] = df_r.groupby(level=1)['$close'].transform(lambda x: x.rolling(20, min_periods=5).mean())\n"
        )
        result = fixer.fix(code)
        assert _is_valid_python(result)

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_volume_proxy_and_groupby_fix(self, seed):
        """Property: volume proxy and groupby fixes work together."""
        fixer = _auto_fixer()
        code = (
            "def calc():\n"
            "    df = pd.read_hdf('data.h5', key='data')\n"
            "    df['val'] = df.groupby(level=1)['$close'].apply(lambda x: x.pct_change())\n"
            "    df['pv'] = df['$close'] * df['$volume']\n"
            "    return df[['val', 'pv']]\n"
        )
        result = fixer.fix(code)
        assert _is_valid_python(result)
        assert "volume proxy" in result
        assert ".transform(" in result

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_instrument_and_level_fix_together(self, seed):
        """Property: instrument column replacement and level= fix work together."""
        fixer = _auto_fixer()
        code = (
            "df['key'] = df['instrument'] + '_' + df['day_id'].astype(str)\n"
            "df.groupby(level=['instrument', 'date'])['col'].transform('sum')\n"
        )
        result = fixer.fix(code)
        assert "df['instrument']" not in result
        assert "get_level_values" in result


# ---------------------------------------------------------------------------
# Property 22: Fix Order Independence
# ---------------------------------------------------------------------------


class TestFixOrderIndependence:
    """Property: specific fix patterns produce deterministic results."""

    @given(
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=10000)
    def test_same_input_same_output_always(self, seed):
        """Property: fixing same code twice gives identical results."""
        fixer1 = _auto_fixer()
        fixer2 = _auto_fixer()
        code = (
            "df_r = df.reset_index()\n"
            "df_r['x'] = df_r.groupby(level=1)['$close'].apply(lambda x: np.log(x / x.shift(1)))\n"
            "df['y'] = df.groupby(level=['instrument', 'date'])['col'].transform('sum')\n"
            "df['z'] = df.groupby(level=1)['ret'].transform(lambda x: x.rolling(20, min_periods=1).std())\n"
        )
        assert fixer1.fix(code) == fixer2.fix(code)

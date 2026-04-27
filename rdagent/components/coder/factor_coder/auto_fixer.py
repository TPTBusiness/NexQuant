"""
Predix Factor Auto-Fixer - Automatically patches common factor code issues.

This module intercepts LLM-generated factor code and automatically fixes known problems:
1. min_periods mismatch in rolling window calculations
2. Missing inf/NaN handling for division by zero
3. groupby().apply() instead of groupby().transform()
4. Incomplete data range processing
5. Missing groupby for MultiIndex dataframes

Usage:
    auto_fixer = FactorAutoFixer()
    fixed_code = auto_fixer.fix(original_code, factor_task_info)
"""

import ast
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class FactorAutoFixer:
    """
    Automatically patches common factor code issues before execution.

    This runs AFTER LLM code generation but BEFORE execution, ensuring
    known patterns are fixed without requiring another LLM iteration.
    """

    def __init__(self):
        self.fixes_applied = []

    def fix(self, code: str, factor_task_info: Optional[str] = None) -> str:
        """
        Apply all auto-fixes to generated factor code.

        Parameters
        ----------
        code : str
            LLM-generated factor code
        factor_task_info : str, optional
            Factor task information for context-aware fixes

        Returns
        -------
        str
            Patched factor code
        """
        self.fixes_applied = []
        fixed_code = code

        # Apply fixes in order
        # NOTE: _fix_min_periods is intentionally excluded — it increased min_periods to
        # match window size, which causes all-NaN output for intraday data with 96 bars/day
        # (window=240 > 96 means zero valid bars per day). The LLM sets its own min_periods.
        fix_methods = [
            self._fix_instrument_column_access,     # First: fix df['instrument'] on MultiIndex
            self._fix_instrument_loc_multiindex,    # Second: fix df.loc[instrument_var] on MultiIndex
            self._fix_reset_index_groupby,          # Third: fix groupby(level=N) after reset_index()
            self._fix_groupby_mixed_levels,         # Fourth: fix groupby(level=[int, str])
            self._fix_groupby_column_on_multiindex, # Fifth: fix groupby(['instrument','date']) on MultiIndex
            self._fix_chained_groupby,              # Sixth: fix groupby(level=N).groupby('date') chain
            self._fix_rolling_ddof,                 # Seventh: remove unsupported ddof kwarg
            self._fix_groupby_apply_to_transform,   # Eighth: fix groupby patterns
            self._fix_inf_nan_handling,             # Ninth: add inf/nan handling
            self._fix_data_range_processing,        # Tenth: ensure full data range
            self._fix_multiindex_groupby,           # Eleventh: ensure groupby on MultiIndex
        ]

        for fix_method in fix_methods:
            try:
                fixed_code = fix_method(fixed_code)
            except Exception as e:
                logger.debug(f"Auto-fixer {fix_method.__name__} failed: {e}")
                continue

        if self.fixes_applied:
            logger.info(
                f"[AutoFix] Applied {len(self.fixes_applied)} fix(es) for {factor_task_info or 'unknown'}: "
                f"{', '.join(self.fixes_applied)}"
            )

        return fixed_code

    def _fix_instrument_column_access(self, code: str) -> str:
        """
        Fix: df['instrument'] raises KeyError on a MultiIndex DataFrame because
        'instrument' is an index level (level 1), not a column.

        Replace df['instrument'] with df.index.get_level_values('instrument')
        but only when the DataFrame has a MultiIndex (not after reset_index which
        would have promoted it to a real column).

        Also fixes df.reset_index()['instrument'] correctly since after reset_index
        the column exists.
        """
        fixed_code = code

        # Skip if already fixed or if reset_index() is being used before the access
        # We only fix bare df['instrument'] where df is the original MultiIndex frame.
        # Heuristic: if the assignment lhs or context shows reset_index, leave it alone.

        # Pattern: <varname>['instrument'] where varname is NOT a reset_index result
        reset_vars = set(re.findall(r'(\w+)\s*=\s*\w[^=\n]*\.reset_index\(', fixed_code))

        def _replace_instrument_access(m: re.Match) -> str:
            var = m.group(1)
            if var in reset_vars:
                return m.group(0)  # leave reset_index vars alone — column exists
            self.fixes_applied.append(f"instrument_column: {var}['instrument'] → get_level_values(1)")
            return f"{var}.index.get_level_values(1)"

        fixed_code = re.sub(r"(\w+)\['instrument'\]", _replace_instrument_access, fixed_code)

        return fixed_code

    def _fix_instrument_loc_multiindex(self, code: str) -> str:
        """
        Fix: df.loc[instrument_var] raises DateParseError on a (datetime, instrument)
        MultiIndex because pandas tries to match the instrument string against the
        datetime level (level 0).

        Pattern detected: for-loops iterating over get_level_values('instrument') or
        get_level_values(1) where the loop variable is then used as df.loc[loop_var].

        Replacement: df.loc[instrument_var] → df.xs(instrument_var, level=1)
        """
        fixed_code = code

        # Find variables iterated from get_level_values('instrument') or get_level_values(1)
        inst_vars = set(
            re.findall(
                r"for\s+(\w+)\s+in\s+.+?\.get_level_values\s*\(\s*(?:1|['\"]instrument['\"])\s*\)[^:\n]*:",
                code,
            )
        )

        if not inst_vars:
            return fixed_code

        for var in inst_vars:
            # Replace DF.loc[var] (read) with DF.xs(var, level=1)
            # Exclude write-back patterns (DF.loc[var] = ...) — leave those as-is
            def _make_replacer(v: str):
                def _replace(m: re.Match) -> str:
                    df_var = m.group(1)
                    self.fixes_applied.append(
                        f"instrument_loc: {df_var}.loc[{v}] → {df_var}.xs({v}, level=1)"
                    )
                    return f"{df_var}.xs({v}, level=1)"

                return _replace

            # Only match when NOT followed by ' =' (assignment)
            fixed_code = re.sub(
                rf"(\w+)\.loc\[\s*{re.escape(var)}\s*\](?!\s*=)",
                _make_replacer(var),
                fixed_code,
            )

        return fixed_code

    def _fix_reset_index_groupby(self, code: str) -> str:
        """
        Fix: groupby(level=N) on a variable created by .reset_index() fails because
        reset_index() converts the MultiIndex into regular columns, leaving a plain
        RangeIndex.  Replace groupby(level=N) on such variables with
        groupby('instrument').

        Detected pattern:
            varname = <anything>.reset_index(...)
            ...
            varname.groupby(level=0|1)
        """
        fixed_code = code

        # Find all variables assigned via reset_index()
        reset_vars = set(re.findall(r'(\w+)\s*=\s*\w[^=\n]*\.reset_index\(', fixed_code))

        for var in reset_vars:
            # Replace var.groupby(level=N) with var.groupby('instrument')
            pattern = rf'{re.escape(var)}\.groupby\(level\s*=\s*\d+\)'
            if re.search(pattern, fixed_code):
                fixed_code = re.sub(pattern, f"{var}.groupby('instrument')", fixed_code)
                self.fixes_applied.append(f"reset_index_groupby: {var}.groupby(level=N) → groupby('instrument')")

        return fixed_code

    def _fix_groupby_mixed_levels(self, code: str) -> str:
        """
        Fix: groupby(level=[int, 'str']) raises AssertionError because string level
        names don't exist on an unnamed MultiIndex.  Keep only integer levels.

        Pattern:  .groupby(level=[0, 'date'])  →  .groupby(level=0)
                  .groupby(level=[1, 'date'])  →  .groupby(level=1)
        """
        fixed_code = code

        def _keep_int_levels(m):
            inner = m.group(1)
            ints = re.findall(r'\b(\d+)\b', inner)
            if not ints:
                return m.group(0)
            replacement = f'.groupby(level={ints[0]})' if len(ints) == 1 else f'.groupby(level=[{", ".join(ints)}])'
            self.fixes_applied.append(f"mixed_levels: groupby(level=[...,str]) → {replacement}")
            return replacement

        fixed_code = re.sub(r'\.groupby\(level=\[([^\]]+)\]\)', _keep_int_levels, fixed_code)
        return fixed_code

    def _fix_groupby_column_on_multiindex(self, code: str) -> str:
        """
        Fix: groupby(['instrument', 'date']) on a MultiIndex (datetime, instrument)
        DataFrame fails with KeyError because those are index levels, not columns.

        Correct replacement preserves BOTH dimensions so intraday calculations reset
        per day:
            var.groupby(['instrument', 'date'])
            → var.groupby([var.index.get_level_values(1), var.index.get_level_values(0).normalize()])

        Single-column groupby(['instrument']) is correctly replaced with groupby(level=1).
        Note: do NOT convert groupby('instrument') → groupby(level=1) here — that would
        undo the reset_index_groupby fix which correctly emits groupby('instrument').
        """
        fixed_code = code

        def _replace_two_col_groupby(m: re.Match, order: str) -> str:
            var = m.group(1)
            if order == "instrument_date":
                repl = (
                    f"{var}.groupby([{var}.index.get_level_values(1), "
                    f"{var}.index.get_level_values(0).normalize()])"
                )
            else:  # date_instrument
                repl = (
                    f"{var}.groupby([{var}.index.get_level_values(0).normalize(), "
                    f"{var}.index.get_level_values(1)])"
                )
            self.fixes_applied.append(f"multiindex_groupby: {m.group(0)[:60]} → two-level")
            return repl

        # groupby(['instrument', 'date']) — capture variable name before .groupby
        fixed_code = re.sub(
            r'(\w+)\.groupby\(\[\'instrument\',\s*\'date\'\]\)',
            lambda m: _replace_two_col_groupby(m, "instrument_date"),
            fixed_code,
        )
        # groupby(['date', 'instrument'])
        fixed_code = re.sub(
            r'(\w+)\.groupby\(\[\'date\',\s*\'instrument\'\]\)',
            lambda m: _replace_two_col_groupby(m, "date_instrument"),
            fixed_code,
        )
        # single: groupby(['instrument']) → groupby(level=1)
        if re.search(r"\.groupby\(\['instrument'\]\)", fixed_code):
            fixed_code = re.sub(r"\.groupby\(\['instrument'\]\)", ".groupby(level=1)", fixed_code)
            self.fixes_applied.append("multiindex_groupby: groupby(['instrument']) → groupby(level=1)")

        return fixed_code

    def _fix_chained_groupby(self, code: str) -> str:
        """
        Fix two broken patterns the LLM generates when trying to group by (instrument, date):

        Pattern A — chained groupby (runtime AttributeError):
            var.groupby(level=1).groupby('date')
            → var.groupby([var.index.get_level_values(1),
                           var.index.get_level_values(0).normalize()])

        Pattern B — keyword arg inside list (SyntaxError):
            var.groupby([level=1, 'date'])
            → same two-level replacement
        """
        fixed_code = code

        def _two_level(var: str, tag: str) -> str:
            self.fixes_applied.append(f"chained_groupby: {tag} → two-level")
            return (
                f"{var}.groupby([{var}.index.get_level_values(1), "
                f"{var}.index.get_level_values(0).normalize()])"
            )

        # Pattern A: var.groupby(level=N).groupby('date')
        fixed_code = re.sub(
            r'(\w+)\.groupby\(level=\d+\)\.groupby\(["\']date["\']\)',
            lambda m: _two_level(m.group(1), m.group(0)[:60]),
            fixed_code,
        )

        # Pattern B: .groupby([level=N, 'date']) — SyntaxError in Python.
        # The variable before .groupby may be complex (e.g. df[mask]) so we don't
        # try to capture it; we use df as the index reference (always correct since
        # all filtered frames share df's MultiIndex structure).
        def _two_level_df(tag: str) -> str:
            self.fixes_applied.append(f"chained_groupby: {tag} → two-level")
            return ".groupby([df.index.get_level_values(1), df.index.get_level_values(0).normalize()])"

        fixed_code = re.sub(
            r'\.groupby\(\[\s*level\s*=\s*\d+\s*,\s*["\']?date["\']?\s*\]\)',
            lambda m: _two_level_df(m.group(0)[:60]),
            fixed_code,
        )
        # Also handle reversed order: ['date', level=N]
        fixed_code = re.sub(
            r'\.groupby\(\[\s*["\']?date["\']?\s*,\s*level\s*=\s*\d+\s*\]\)',
            lambda m: _two_level_df(m.group(0)[:60]),
            fixed_code,
        )

        return fixed_code

    def _fix_rolling_ddof(self, code: str) -> str:
        """
        Fix: pandas rolling() does not accept a ddof kwarg — raises TypeError.
        Remove ddof from both rolling(..., ddof=N) and rolling(...).std(ddof=N).
        """
        fixed_code = code

        # Form 1: ddof inside rolling() — .rolling(window=N, min_periods=M, ddof=K)
        def _strip_ddof_from_rolling(m):
            inner = re.sub(r',?\s*ddof\s*=\s*\d+', '', m.group(1))
            inner = inner.strip(', ')
            self.fixes_applied.append("rolling_ddof: removed ddof from rolling()")
            return f'.rolling({inner})'

        fixed_code = re.sub(r'\.rolling\(([^)]*ddof\s*=\s*\d+[^)]*)\)', _strip_ddof_from_rolling, fixed_code)

        # Form 2: ddof inside .std() / .var() — .std(ddof=N)
        if re.search(r'\.(std|var)\([^)]*ddof\s*=\s*\d+', fixed_code):
            fixed_code = re.sub(r'\.(std|var)\([^)]*ddof\s*=\s*\d+[^)]*\)', r'.\1()', fixed_code)
            self.fixes_applied.append("rolling_ddof: removed ddof from std()/var()")

        return fixed_code

    def _fix_min_periods(self, code: str) -> str:
        """
        Fix: Ensure min_periods matches window size in rolling calculations.

        Problem: LLM often sets min_periods=1 or min_periods=2 for rolling windows,
        which creates inconsistent feature definitions.

        Fix: Set min_periods equal to window size.
        """
        fixed_code = code

        # Pattern 1: .rolling(window=N, min_periods=M) where M < N
        # Replace with min_periods=N
        pattern1 = r'\.rolling\(window=(\d+),\s*min_periods=(\d+)\)'

        def replace_min_periods1(match):
            window_size = int(match.group(1))
            min_periods = int(match.group(2))
            if min_periods < window_size:
                self.fixes_applied.append(f"min_periods: {min_periods}→{window_size}")
                return f'.rolling(window={window_size}, min_periods={window_size})'
            return match.group(0)

        fixed_code = re.sub(pattern1, replace_min_periods1, fixed_code)

        # Pattern 2: .rolling(N).mean() or .rolling(N).std() without min_periods
        # Add min_periods=N
        pattern2 = r'\.rolling\((\d+)\)\.(mean|std|var|sum|count|median|skew|kurt|quantile|min|max)\(\)'

        def replace_min_periods2(match):
            window_size = int(match.group(1))
            method = match.group(2)
            self.fixes_applied.append(f"min_periods: added {window_size} for {method}")
            return f'.rolling({window_size}, min_periods={window_size}).{method}()'

        fixed_code = re.sub(pattern2, replace_min_periods2, fixed_code)

        # Pattern 3: .rolling(window=N).method() without min_periods
        pattern3 = r'\.rolling\(window=(\d+)\)\.(mean|std|var|sum|count|median|skew|kurt|quantile|min|max)\(\)'

        def replace_min_periods3(match):
            window_size = int(match.group(1))
            method = match.group(2)
            self.fixes_applied.append(f"min_periods: added {window_size} for {method}")
            return f'.rolling(window={window_size}, min_periods={window_size}).{method}()'

        fixed_code = re.sub(pattern3, replace_min_periods3, fixed_code)

        return fixed_code

    def _fix_inf_nan_handling(self, code: str) -> str:
        """
        Fix: Add inf/NaN handling after division operations.

        Problem: Z-score and ratio calculations can produce inf values when
        denominator (std, volatility) is zero.

        Fix: Add .replace([np.inf, -np.inf], np.nan) after result calculation.
        """
        fixed_code = code

        # Check if inf handling already exists
        if 'replace([np.inf, -np.inf]' in fixed_code or 'replace([np.inf,-np.inf]' in fixed_code:
            if 'np.nan' in fixed_code or 'np.NaN' in fixed_code:
                return fixed_code  # Already handled

        # Pattern 1: Division operation that could produce inf
        # Look for patterns like: df['zscore'] = ... / df['sigma_20bar']
        # or: df['ratio'] = df['sigma_5bar'] / df['sigma_60bar']

        # Find the result column assignment (last major assignment before save)
        # Pattern: result = df[['column_name']] or df['column_name'] = ...

        # Add inf handling before the save operation
        save_pattern = r'(\s*result\s*=\s*df\[\[.*?\]\])'
        match = re.search(save_pattern, fixed_code, re.DOTALL)

        if match:
            insert_pos = match.start()
            # Extract column name from the result assignment
            col_match = re.search(r"result\s*=\s*df\[\[(.*?)\]\]", match.group(0))
            if col_match:
                col_name = col_match.group(1).strip().strip("'\"")
                inf_fix = f"\n    # Auto-fix: Handle infinite values\n    df['{col_name}'] = df['{col_name}'].replace([np.inf, -np.inf], np.nan)\n"
                fixed_code = fixed_code[:insert_pos] + inf_fix + fixed_code[insert_pos:]
                self.fixes_applied.append("inf/nan: added replace for inf values")
                return fixed_code

        # Pattern 2: Direct assignment to result variable
        # Add inf handling before dropna or save
        dropna_pattern = r'(\s*\.dropna\(\))'
        match = re.search(dropna_pattern, fixed_code)

        if match:
            insert_pos = match.start()
            # Find the column being processed
            # Look backwards for the last assignment
            lines_before = fixed_code[:insert_pos].split('\n')
            for line in reversed(lines_before):
                col_match = re.search(r"df\['(.+?)'\]\s*=", line.strip())
                if col_match:
                    col_name = col_match.group(1)
                    inf_fix = f"    # Auto-fix: Handle infinite values\n    df['{col_name}'] = df['{col_name}'].replace([np.inf, -np.inf], np.nan)\n"
                    fixed_code = fixed_code[:insert_pos] + inf_fix + fixed_code[insert_pos:]
                    self.fixes_applied.append("inf/nan: added replace for inf values")
                    return fixed_code

        # Pattern 3: Generic fallback - add inf handling before any .to_hdf call
        hdf_pattern = r'(\s*\.to_hdf\()'
        match = re.search(hdf_pattern, fixed_code)

        if match:
            insert_pos = match.start()
            inf_fix = "    # Auto-fix: Handle infinite values\n    result = result.replace([np.inf, -np.inf], np.nan)\n"
            fixed_code = fixed_code[:insert_pos] + inf_fix + fixed_code[insert_pos:]
            self.fixes_applied.append("inf/nan: added replace for inf values on result")

        return fixed_code

    def _fix_groupby_apply_to_transform(self, code: str) -> str:
        """
        Fix: Convert groupby().apply() to groupby().transform() where appropriate.

        Problem: groupby().apply() returns a DataFrame structure that cannot be
        assigned to a single column, causing ValueError.

        Fix: Use groupby().transform() which preserves original DataFrame structure.
        """
        fixed_code = code

        # === CRITICAL FIX: groupby().rolling() on MultiIndex creates extra index level ===
        # Pattern: df.groupby(level=N)['col'].rolling(window=W, min_periods=M).method()
        # When assigned back to df['new_col'], it causes:
        #   AssertionError: Length of new_levels (3) must be <= self.nlevels (2)
        # Fix: Add .reset_index(level=-1, drop=True) after rolling operation

        # Pattern: df.groupby(level=N)['col_A'].rolling(window=W, min_periods=M).corr(x['col_B'])
        rolling_corr_pattern = (
            r"df\.groupby\(level=(\d+)\)\['([^']+)'\]\.rolling\(\s*window=(\d+)\s*,\s*min_periods=(\d+)\s*\)"
            r"\.corr\(x\['([^']+)'\]\)"
        )
        match = re.search(rolling_corr_pattern, fixed_code)
        if match:
            level = match.group(1)
            col_a = match.group(2)
            window = match.group(3)
            min_periods = match.group(4)
            col_b = match.group(5)

            old_code = match.group(0)
            new_code = (
                f"df.groupby(level={level}).apply(\n"
                f"        lambda x: x['{col_a}'].rolling(window={window}, min_periods={min_periods}).corr(x['{col_b}'])\n"
                f"    ).reset_index(level={level}, drop=True)"
            )
            fixed_code = fixed_code.replace(old_code, new_code)
            self.fixes_applied.append(f"groupby: fixed rolling correlation with reset_index (window={window})")
            # Continue to check for more patterns below

        # Pattern: df.groupby(level=N)['col'].rolling(window=W, min_periods=M).method()
        # This is the MOST COMMON pattern that causes failures
        # Matches multi-line expressions too
        groupby_rolling_pattern = (
            r"df\.groupby\(level=(\d+)\)\['([^']+)'\]\.rolling\(\s*([^)]+)\s*\)\.(\w+)\(\)"
        )

        for match in re.finditer(groupby_rolling_pattern, fixed_code, re.DOTALL):
            full_expr = match.group(0)
            level = match.group(1)
            col_name = match.group(2)
            rolling_args = match.group(3).strip()
            # Normalize rolling_args to single line
            rolling_args = ' '.join(rolling_args.split())
            method = match.group(4)

            # Check if this expression is being assigned to df[...]
            # Since full_expr may contain newlines, use a flexible pattern
            # Look for: df['xxx'] = df.groupby(level=N)['col'].rolling(...)
            # We need to match even with whitespace/newlines between tokens
            escaped_parts = []
            for token in ["df", r"\.groupby\(level=" + level + r"\)\['" + re.escape(col_name) + r"'\]", r"\.rolling\("]:
                escaped_parts.append(re.escape(token) if not token.startswith(r"\\") else token)

            # Simpler approach: search for assignment before the match position
            match_start = match.start()
            preceding_text = fixed_code[max(0, match_start-50):match_start]
            assign_match = re.search(r"df\['[^']+'\]\s*=\s*$", preceding_text)

            if assign_match:
                # Direct assignment - use transform pattern
                new_expr = f"df.groupby(level={level})['{col_name}'].transform(lambda x: x.rolling({rolling_args}).{method}())"
                fixed_code = fixed_code[:match.start()] + new_expr + fixed_code[match.end():]
                self.fixes_applied.append(f"groupby: converted rolling {method} to transform pattern")
            else:
                # Not direct assignment but still needs fix
                new_expr = f"df.groupby(level={level})['{col_name}'].rolling({rolling_args}).{method}().reset_index(level=-1, drop=True)"
                fixed_code = fixed_code[:match.start()] + new_expr + fixed_code[match.end():]
                self.fixes_applied.append(f"groupby: added reset_index for rolling {method}")

        # === GENERAL FIX: ANY series.groupby(level=N).rolling() pattern ===
        # Catches patterns like: sigma_60 = returns.groupby(level=1).rolling(...).std()
        # or: mu_30 = volume_price_product.groupby(level=1).rolling(...).mean()
        # These create MultiIndex issues when used in arithmetic with original series
        general_groupby_rolling = (
            r"(\w+)\.groupby\(level=(\d+)\)\.rolling\(\s*([^)]+)\s*\)\.(\w+)\(\)"
        )

        for match in re.finditer(general_groupby_rolling, fixed_code, re.DOTALL):
            full_expr = match.group(0)
            series_name = match.group(1)
            level = match.group(2)
            rolling_args = match.group(3).strip()
            rolling_args = ' '.join(rolling_args.split())
            method = match.group(4)

            # Check if this already has reset_index
            if 'reset_index' not in full_expr and 'transform' not in full_expr:
                # Check if this is assigned to a variable
                assign_pattern = rf"(\w+)\s*=\s*{re.escape(full_expr)}"
                if re.search(assign_pattern, fixed_code):
                    new_expr = f"{series_name}.groupby(level={level}).rolling({rolling_args}).{method}().reset_index(level=-1, drop=True)"
                    fixed_code = fixed_code.replace(full_expr, new_expr)
                    self.fixes_applied.append(f"groupby: added reset_index for {series_name}.rolling().{method}()")

        # Pattern: Rolling correlation with groupby().apply() - CRITICAL FIX
        # df.groupby(level=N).apply(lambda x: x['A'].rolling(window=W).corr(x['B']))
        corr_pattern = r"df\.groupby\(level=(\d+)\)\.apply\(\s*lambda\s+x:\s+x\['([^']+)'\]\.rolling\(window=(\d+)[^)]*\)\.corr\(x\['([^']+)'\]\)\)"

        match = re.search(corr_pattern, fixed_code)
        if match:
            level = match.group(1)
            col_a = match.group(2)
            window = match.group(3)

            # Find the actual second column name
            full_match = match.group(0)
            col_b_match = re.search(r"corr\(x\['([^']+)'\]\)", full_match)
            if col_b_match:
                col_b = col_b_match.group(1)

            # Replace with proper rolling correlation per group
            old_code = match.group(0)
            new_code = (
                f"df.groupby(level={level}).apply(\n"
                f"        lambda x: x['{col_a}'].rolling(window={window}, min_periods={window}).corr(x['{col_b}'])\n"
                f"    ).reset_index(level={level}, drop=True)"
            )
            fixed_code = fixed_code.replace(old_code, new_code)
            self.fixes_applied.append(f"groupby: fixed rolling correlation (window={window}) with reset_index")

        # Pattern: Simple groupby().apply() with rolling().method()
        # df.groupby(level=N).apply(lambda x: x['col'].rolling(...).method())
        apply_pattern = r"df\.groupby\(level=(\d+)\)\.apply\(\s*lambda\s+x:\s+x\['([^']+)'\]\.rolling\([^)]+\)\.(\w+)\([^)]*\)\s*\)"

        match = re.search(apply_pattern, fixed_code)
        if match:
            level = match.group(1)
            col_name = match.group(2)
            method = match.group(3)

            # Replace with transform pattern
            old_code = match.group(0)
            # Extract window size from the rolling call
            window_match = re.search(r"rolling\(window=(\d+)", old_code)
            window = window_match.group(1) if window_match else "20"

            new_code = f"df.groupby(level={level})['{col_name}'].transform(lambda x: x.rolling(window={window}, min_periods={window}).{method}())"
            fixed_code = fixed_code.replace(old_code, new_code)
            self.fixes_applied.append(f"groupby: converted apply() to transform() for {method}")

        return fixed_code

    def _fix_data_range_processing(self, code: str) -> str:
        """
        Fix: Ensure full data range (2020-2026) is processed, not just a subset.

        Problem: Some factors only process a subset of data (e.g., 2024-2024).

        Fix: Remove any date filtering and ensure full range processing.
        """
        fixed_code = code

        # Remove date filtering patterns
        date_filter_patterns = [
            r"df\s*=\s*df\.loc\[[^:]*20\d\d[^]]*\]",
            r"df\s*=\s*df\[df\.index\.get_level_values\('datetime'\)\s*>=\s*['\"]20\d\d",
            r"df\s*=\s*df\[(df\.)?index\.get_level_values\(0\)\s*>=\s*",
        ]

        for pattern in date_filter_patterns:
            match = re.search(pattern, fixed_code)
            if match:
                # Comment out the date filter instead of removing
                self.fixes_applied.append("data_range: removed date filter")
                fixed_code = fixed_code.replace(match.group(0), f"# Date filter removed to process full range: {match.group(0)}")

        return fixed_code

    def _fix_multiindex_groupby(self, code: str) -> str:
        """
        Fix: Ensure rolling operations use groupby(level=1) for MultiIndex dataframes.

        Problem: Without groupby, rolling calculations mix instruments together.

        Fix: Add groupby(level=1) before rolling operations if not already present.
        """
        fixed_code = code

        # Check if code already has groupby
        if 'groupby(level=' in fixed_code or 'groupby("instrument")' in fixed_code:
            return fixed_code

        # Check if code uses MultiIndex (has 'instrument' in index)
        if 'level=1' not in fixed_code and 'level=' not in fixed_code:
            # Check if there are rolling operations that should be grouped
            rolling_pattern = r"\.rolling\(\d+\)"
            if re.search(rolling_pattern, fixed_code):
                # The code might need groupby, but we can't safely add it without
                # understanding the full context. Log a warning instead.
                logger.warning(
                    f"[AutoFix] Code uses rolling without groupby - may need manual review"
                )

        return fixed_code


# Module-level convenience function
def auto_fix_factor_code(code: str, factor_task_info: Optional[str] = None) -> str:
    """
    Apply all auto-fixes to factor code.

    Parameters
    ----------
    code : str
        LLM-generated factor code
    factor_task_info : str, optional
        Factor task information

    Returns
    -------
    str
        Patched factor code
    """
    fixer = FactorAutoFixer()
    return fixer.fix(code, factor_task_info)

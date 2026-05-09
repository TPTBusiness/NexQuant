"""
Session Analyst — analysiert welche FX Session gerade aktiv ist
und gibt Momentum vs Mean-Reversion Empfehlung
"""
from datetime import datetime, timezone
import pandas as pd


def create_session_analyst(llm):
    def session_analyst_node(state):
        factor_report = state.get("factor_report", "")
        current_date = state.get("trade_date", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))

        # Aktuelle UTC Stunde bestimmen
        try:
            dt = datetime.fromisoformat(current_date.replace("Z", "+00:00"))
            hour_utc = dt.hour
        except Exception:
            hour_utc = datetime.now(timezone.utc).hour

        # Session bestimmen
        if 0 <= hour_utc < 8:
            session = "Asian"
            regime = "Mean Reversion"
            session_note = "Low volume, ranging market. Mean reversion factors perform better. Avoid momentum strategies."
        elif 8 <= hour_utc < 13:
            session = "London"
            regime = "Momentum/Trending"
            session_note = "High volume, trending market. Momentum factors perform better. London open breakouts common."
        elif 13 <= hour_utc < 16:
            session = "London-NY Overlap"
            regime = "Strong Momentum"
            session_note = "Highest volume of the day. Strong directional moves. Best session for momentum factors."
        elif 16 <= hour_utc < 21:
            session = "NY"
            regime = "Momentum/Reversal"
            session_note = "Moderate volume. Watch for reversals after London close at 16:00."
        else:
            session = "After Hours"
            regime = "Low Liquidity"
            session_note = "Very low volume. Spreads widen. Avoid trading."

        prompt = f"""You are an FX Session Analyst specialized in EURUSD intraday dynamics.

Current UTC time: {current_date}
Active Session: {session}
Expected Regime: {regime}
Session Notes: {session_note}

Factor Report from NexQuant RD-Agent:
{factor_report}

Analyze whether the proposed factor is suitable for the current session regime.
Provide a detailed report covering:
1. Session characteristics and expected price behavior
2. Whether the factor aligns with the current session regime
3. Recommended adjustments if needed (e.g., apply only during London hours)
4. Risk considerations for current session

End with a markdown table summarizing your findings.
"""
        response = llm.invoke(prompt)
        report = response.content if hasattr(response, 'content') else str(response)

        return {
            "messages": [response],
            "session_report": report,
            "current_session": session,
            "current_regime": regime,
        }

    return session_analyst_node

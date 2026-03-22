"""
FX Macro Analyst — analysiert makroökonomische FX-Faktoren
statt Aktien-Fundamentals
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def get_fx_macro_data() -> str:
    """Holt DXY, EUR/USD, Volatilität als Makro-Kontext"""
    try:
        end = datetime.now()
        start = end - timedelta(days=5)

        eurusd = yf.download("EURUSD=X", start=start, end=end, interval="1h", progress=False)
        dxy = yf.download("DX-Y.NYB", start=start, end=end, interval="1h", progress=False)

        eurusd_last = eurusd['Close'].iloc[-1] if not eurusd.empty else "N/A"
        eurusd_change = ((eurusd['Close'].iloc[-1] / eurusd['Close'].iloc[-24]) - 1) * 100 if len(eurusd) > 24 else "N/A"
        dxy_last = dxy['Close'].iloc[-1] if not dxy.empty else "N/A"

        # Realized volatility (24h)
        if len(eurusd) > 1:
            returns = eurusd['Close'].pct_change().dropna()
            vol_24h = returns.tail(24).std() * (24**0.5) * 100
        else:
            vol_24h = "N/A"

        return f"""
EURUSD Current: {eurusd_last:.5f if isinstance(eurusd_last, float) else eurusd_last}
EURUSD 24h Change: {eurusd_change:.3f}% if isinstance(eurusd_change, float) else eurusd_change}
DXY (Dollar Index): {dxy_last:.2f if isinstance(dxy_last, float) else dxy_last}
Realized Volatility 24h: {vol_24h:.4f}% if isinstance(vol_24h, float) else vol_24h}
"""
    except Exception as e:
        return f"Macro data unavailable: {e}"


def create_macro_analyst(llm):
    def macro_analyst_node(state):
        factor_report = state.get("factor_report", "")
        current_date = state.get("trade_date", "")

        macro_data = get_fx_macro_data()

        prompt = f"""You are an FX Macro Analyst specialized in EURUSD trading.

Current Date: {current_date}

Live Macro Data:
{macro_data}

Factor Report from Predix RD-Agent:
{factor_report}

Analyze the macro environment and its impact on the proposed factor:
1. DXY trend and correlation with EURUSD
2. Current volatility regime (low/normal/high) and what it means for the factor
3. Any macro risks that could invalidate the factor signal
4. Recommended position sizing given current volatility

End with a markdown table of key macro indicators and their signal implications.
"""
        response = llm.invoke(prompt)
        report = response.content if hasattr(response, 'content') else str(response)

        return {
            "messages": [response],
            "macro_report": report,
        }

    return macro_analyst_node

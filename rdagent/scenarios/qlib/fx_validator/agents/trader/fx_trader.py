"""
FX Trader — trifft finale BUY/HOLD/SELL Entscheidung
basierend auf allen Analyst- und Researcher-Reports
"""


def create_fx_trader(llm):
    def trader_node(state):
        factor_report = state.get("factor_report", "")
        session_report = state.get("session_report", "")
        macro_report = state.get("macro_report", "")
        debate_state = state.get("fx_debate_state", {})
        debate_history = debate_state.get("history", "")
        risk_report = state.get("risk_report", "")

        prompt = f"""You are an FX Trading Decision Agent for EURUSD 1min intraday trading.

You have received reports from your team:

FACTOR ANALYSIS (NexQuant RD-Agent):
{factor_report}

SESSION ANALYSIS:
{session_report}

MACRO ANALYSIS:
{macro_report}

BULL vs BEAR DEBATE:
{debate_history}

RISK ASSESSMENT:
{risk_report}

Based on all available information, make a final trading decision:
- APPROVE: Factor is valid, deploy it in the live loop
- REJECT: Factor has critical flaws, do not deploy
- CONDITIONAL: Deploy only under specific conditions (specify which)

Consider:
1. IC > 0.02 threshold
2. ARR > 9.62% target
3. Spread costs vs signal strength
4. Session-specific validity
5. Macro regime alignment

End your response with exactly one of:
FINAL DECISION: **APPROVE**
FINAL DECISION: **REJECT**
FINAL DECISION: **CONDITIONAL: [conditions]**
"""
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        # Decision extrahieren
        decision = "UNKNOWN"
        if "FINAL DECISION: **APPROVE**" in content:
            decision = "APPROVE"
        elif "FINAL DECISION: **REJECT**" in content:
            decision = "REJECT"
        elif "FINAL DECISION: **CONDITIONAL" in content:
            decision = "CONDITIONAL"

        return {
            "messages": [response],
            "trader_decision": content,
            "final_decision": decision,
        }

    return trader_node

"""
Bear FX Researcher — argumentiert GEGEN den vorgeschlagenen Faktor
"""


def create_fx_bear_researcher(llm):
    def bear_node(state):
        debate_state = state.get("fx_debate_state", {})
        history = debate_state.get("history", "")
        current_response = debate_state.get("current_response", "")

        factor_report = state.get("factor_report", "")
        session_report = state.get("session_report", "")
        macro_report = state.get("macro_report", "")

        prompt = f"""You are a Bear FX Analyst arguing AGAINST using this trading factor on EURUSD.

Factor Analysis: {factor_report}
Session Analysis: {session_report}
Macro Analysis: {macro_report}
Debate History: {history}
Bull's Last Argument: {current_response}

Build a critical case AGAINST this factor:
1. Overfitting risk — does IC hold out-of-sample?
2. Spread cost impact — does 1.5 bps erode the edge?
3. Session limitations — does the factor fail in Asian session?
4. Macro regime risk — does the factor break in high-volatility environments?
5. Counter the bull's specific claims with data

Be specific and critical. Challenge every assumption.
"""
        response = llm.invoke(prompt)
        argument = f"Bear Analyst: {response.content if hasattr(response, 'content') else str(response)}"

        new_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": debate_state.get("bear_history", "") + "\n" + argument,
            "bull_history": debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": debate_state.get("count", 0) + 1,
        }

        return {"fx_debate_state": new_debate_state}

    return bear_node

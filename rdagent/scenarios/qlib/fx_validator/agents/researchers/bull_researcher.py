"""
Bull FX Researcher — argumentiert FÜR den vorgeschlagenen Faktor
"""


def create_fx_bull_researcher(llm):
    def bull_node(state):
        debate_state = state.get("fx_debate_state", {})
        history = debate_state.get("history", "")
        current_response = debate_state.get("current_response", "")

        factor_report = state.get("factor_report", "")
        session_report = state.get("session_report", "")
        macro_report = state.get("macro_report", "")

        prompt = f"""You are a Bull FX Analyst advocating FOR using this trading factor on EURUSD.

Factor Analysis: {factor_report}
Session Analysis: {session_report}
Macro Analysis: {macro_report}
Debate History: {history}
Bear's Last Argument: {current_response}

Build a strong evidence-based case FOR this factor:
1. Why the IC and ARR justify using this factor
2. How session timing makes this factor especially effective
3. Why spread costs are manageable given the signal strength
4. Counter the bear's specific concerns with data

Be specific, use numbers from the reports. Engage directly with bear arguments.
"""
        response = llm.invoke(prompt)
        argument = f"Bull Analyst: {response.content if hasattr(response, 'content') else str(response)}"

        new_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": debate_state.get("bull_history", "") + "\n" + argument,
            "bear_history": debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": debate_state.get("count", 0) + 1,
        }

        return {"fx_debate_state": new_debate_state}

    return bull_node

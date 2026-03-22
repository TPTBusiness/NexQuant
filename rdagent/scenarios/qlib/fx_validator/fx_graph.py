"""
FX Validator Graph — Multi-Agent Validierung für Predix Faktoren
Inspiriert von TradingAgents, angepasst für EURUSD 15min
"""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import os

from .agents.analysts.session_analyst import create_session_analyst
from .agents.analysts.macro_analyst import create_macro_analyst
from .agents.researchers.bull_researcher import create_fx_bull_researcher
from .agents.researchers.bear_researcher import create_fx_bear_researcher
from .agents.trader.fx_trader import create_fx_trader
from .config import FX_CONFIG


class FXValidatorState(TypedDict):
    factor_report: str
    trade_date: str
    session_report: str
    macro_report: str
    fx_debate_state: dict
    risk_report: str
    trader_decision: str
    final_decision: str
    messages: list


def create_fx_validator(config: dict = None):
    cfg = config or FX_CONFIG

    llm = ChatOpenAI(
        model=cfg["deep_think_llm"].replace("openai/", ""),
        base_url=cfg["backend_url"],
        api_key=cfg["api_key"],
        temperature=0.5,
    )

    # Agenten erstellen
    session_analyst = create_session_analyst(llm)
    macro_analyst = create_macro_analyst(llm)
    bull_researcher = create_fx_bull_researcher(llm)
    bear_researcher = create_fx_bear_researcher(llm)
    fx_trader = create_fx_trader(llm)

    # Debate Loop
    def should_continue_debate(state):
        count = state.get("fx_debate_state", {}).get("count", 0)
        max_rounds = cfg.get("max_debate_rounds", 2) * 2
        if count >= max_rounds:
            return "trader"
        return "bear" if count % 2 == 0 else "bull"

    # Graph bauen
    graph = StateGraph(FXValidatorState)

    graph.add_node("session_analyst", session_analyst)
    graph.add_node("macro_analyst", macro_analyst)
    graph.add_node("bull", bull_researcher)
    graph.add_node("bear", bear_researcher)
    graph.add_node("trader", fx_trader)

    graph.set_entry_point("session_analyst")
    graph.add_edge("session_analyst", "macro_analyst")
    graph.add_edge("macro_analyst", "bull")

    graph.add_conditional_edges(
        "bull",
        should_continue_debate,
        {"bear": "bear", "bull": "bull", "trader": "trader"}
    )
    graph.add_conditional_edges(
        "bear",
        should_continue_debate,
        {"bear": "bear", "bull": "bull", "trader": "trader"}
    )

    graph.add_edge("trader", END)

    return graph.compile()


def validate_factor(factor_report: str, trade_date: str = None) -> dict:
    """
    Hauptfunktion — validiert einen Predix-Faktor durch Multi-Agent Debatte

    Args:
        factor_report: Der Faktor-Report von Predix RD-Agent
        trade_date: Datum/Zeit in ISO Format (default: jetzt)

    Returns:
        dict mit final_decision (APPROVE/REJECT/CONDITIONAL) und Reports
    """
    from datetime import datetime, timezone

    if trade_date is None:
        trade_date = datetime.now(timezone.utc).isoformat()

    validator = create_fx_validator()

    initial_state = {
        "factor_report": factor_report,
        "trade_date": trade_date,
        "session_report": "",
        "macro_report": "",
        "fx_debate_state": {"history": "", "bull_history": "", "bear_history": "", "current_response": "", "count": 0},
        "risk_report": "",
        "trader_decision": "",
        "final_decision": "",
        "messages": [],
    }

    result = validator.invoke(initial_state)
    return result

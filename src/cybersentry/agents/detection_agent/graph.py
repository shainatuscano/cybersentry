"""LangGraph workflow of the Detection Agent.

    START -> receive -> validate --(valid)--> evaluate_signals -> decide -> END
                                 \\-(invalid)-> reject ------------------> END

The graph has no tools and makes no network or LLM calls. It reads one structured detection result and
writes one AgentDecision.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from ...detection.schemas import DetectionResult
from .policy import Thresholds, consistency_errors, decide, evaluate_signals, reject
from .schemas import AgentDecision, DetectionAgentState


def _format_errors(e: ValidationError) -> list[str]:
    out = []
    for err in e.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "detection_result"
        out.append(f"{loc}: {err['msg']}")
    return out


def build_graph(thresholds: Thresholds | None = None):
    t = thresholds or Thresholds()

    def receive(state: DetectionAgentState) -> dict:
        raw = state.get("raw_input")
        if isinstance(raw, DetectionResult):
            raw = raw.model_dump()
        return {"raw_input": raw, "validation_errors": [], "detection": None, "signals": None, "decision": None}

    def validate(state: DetectionAgentState) -> dict:
        raw = state["raw_input"]
        if not isinstance(raw, Mapping):
            return {"validation_errors": [f"detection result must be an object, got {type(raw).__name__}"]}
        try:
            d = DetectionResult.model_validate(dict(raw))
        except ValidationError as e:
            return {"validation_errors": _format_errors(e)}
        errors = consistency_errors(d)
        return {"detection": None if errors else d, "validation_errors": errors}

    def route(state: DetectionAgentState) -> str:
        return "reject" if state["validation_errors"] else "evaluate_signals"

    def evaluate(state: DetectionAgentState) -> dict:
        return {"signals": evaluate_signals(state["detection"], t)}

    def decide_node(state: DetectionAgentState) -> dict:
        return {"decision": decide(state["detection"], state["signals"])}

    def reject_node(state: DetectionAgentState) -> dict:
        raw = state["raw_input"] if isinstance(state["raw_input"], Mapping) else {}
        return {"decision": reject(dict(raw), state["validation_errors"])}

    g = StateGraph(DetectionAgentState)
    g.add_node("receive", receive)
    g.add_node("validate", validate)
    g.add_node("evaluate_signals", evaluate)
    g.add_node("decide", decide_node)
    g.add_node("reject", reject_node)
    g.add_edge(START, "receive")
    g.add_edge("receive", "validate")
    g.add_conditional_edges("validate", route, {"evaluate_signals": "evaluate_signals", "reject": "reject"})
    g.add_edge("evaluate_signals", "decide")
    g.add_edge("decide", END)
    g.add_edge("reject", END)
    return g.compile()


@lru_cache(maxsize=1)
def _default_graph():
    return build_graph()


def run_detection_agent(detection_result: Mapping[str, Any] | DetectionResult,
                        thresholds: Thresholds | None = None) -> AgentDecision:
    """Run the Detection Agent on one structured detection result and return its decision."""
    graph = build_graph(thresholds) if thresholds else _default_graph()
    final = graph.invoke({"raw_input": detection_result})
    return final["decision"]

"""Detection Agent: decides whether an ML detection result should enter the investigation workflow.

The ML models classify; this agent interprets their structured output with a deterministic policy,
orchestrated as a LangGraph workflow. It has no tools, takes no actions and makes no LLM calls.
"""
from .graph import build_graph, run_detection_agent
from .policy import Thresholds
from .schemas import AgentDecision, NextStep, Priority, Status

__all__ = ["AgentDecision", "NextStep", "Priority", "Status", "Thresholds", "build_graph",
           "run_detection_agent"]

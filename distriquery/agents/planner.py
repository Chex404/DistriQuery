"""Query planner (Theory Primer, Part 3.4 — Routing).

Classifies each incoming query into a strategy BEFORE any expensive work
happens:
- "tool": a tool can answer this directly (arithmetic, today's date) —
  no retrieval needed at all.
- "multi_hop": the query has multiple parts or asks for a comparison,
  needing evidence chained across more than one retrieval step.
- "direct": everything else — the existing hybrid+rerank pipeline.

HeuristicQueryPlanner is fast, deterministic, and needs no LLM call for
routing itself — this is the default and what tests use.
"""

import re
from dataclasses import dataclass
from typing import Optional

_ARITHMETIC_RE = re.compile(r"\d+(?:\.\d+)?\s*[\+\-\*/]\s*\d+(?:\.\d+)?")
_DATE_KEYWORDS = ("today's date", "current date", "what day is it", "what's today")
_MULTI_HOP_KEYWORDS = ("compare", "difference between", " vs ", " versus ", "both")


@dataclass
class PlanDecision:
    strategy: str  # "direct" | "multi_hop" | "tool"
    tool_name: Optional[str] = None


class HeuristicQueryPlanner:
    def plan(self, query: str) -> PlanDecision:
        lowered = query.lower()

        if _ARITHMETIC_RE.search(query):
            return PlanDecision(strategy="tool", tool_name="calculator")
        if any(keyword in lowered for keyword in _DATE_KEYWORDS):
            return PlanDecision(strategy="tool", tool_name="current_date")
        if any(keyword in lowered for keyword in _MULTI_HOP_KEYWORDS):
            return PlanDecision(strategy="multi_hop")

        return PlanDecision(strategy="direct")
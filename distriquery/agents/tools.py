"""Tools (Theory Primer, Part 3.5).

Each tool is one action the agent can take instead of retrieval.
"""

import re
from abc import ABC, abstractmethod
from datetime import date

_CALC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([\+\-\*/])\s*(\d+(?:\.\d+)?)")


class Tool(ABC):
    name: str

    @abstractmethod
    def run(self, query: str) -> str:
        raise NotImplementedError


class CalculatorTool(Tool):
    name = "calculator"

    def run(self, query: str) -> str:
        match = _CALC_RE.search(query)
        if not match:
            return "I couldn't find a calculation in that question."

        a, op, b = float(match.group(1)), match.group(2), float(match.group(3))
        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        elif op == "*":
            result = a * b
        else:
            if b == 0:
                return "Cannot divide by zero."
            result = a / b

        return f"{match.group(0)} = {result}"


class CurrentDateTool(Tool):
    name = "current_date"

    def run(self, query: str) -> str:
        return f"Today's date is {date.today().isoformat()}."


_TOOLS = {
    "calculator": CalculatorTool(),
    "current_date": CurrentDateTool(),
}


def get_tool(name: str) -> Tool:
    if name not in _TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    return _TOOLS[name]
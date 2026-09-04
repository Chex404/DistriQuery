import pytest

from distriquery.agents.tools import CalculatorTool, CurrentDateTool, get_tool


def test_calculator_addition():
    tool = CalculatorTool()
    assert tool.run("What is 12 + 7?") == "12 + 7 = 19.0"


def test_calculator_subtraction():
    tool = CalculatorTool()
    assert "= 5.0" in tool.run("10 - 5")


def test_calculator_multiplication():
    tool = CalculatorTool()
    assert "= 20.0" in tool.run("4 * 5")


def test_calculator_division():
    tool = CalculatorTool()
    assert "= 2.5" in tool.run("5 / 2")


def test_calculator_division_by_zero():
    tool = CalculatorTool()
    assert "Cannot divide by zero" in tool.run("5 / 0")


def test_calculator_no_expression_found():
    tool = CalculatorTool()
    result = tool.run("What is the capital of France?")
    assert "couldn't find a calculation" in result


def test_current_date_tool_returns_iso_format():
    from datetime import date

    tool = CurrentDateTool()
    result = tool.run("what's today's date?")

    assert date.today().isoformat() in result


def test_get_tool_returns_correct_instances():
    assert isinstance(get_tool("calculator"), CalculatorTool)
    assert isinstance(get_tool("current_date"), CurrentDateTool)


def test_get_tool_unknown_name_raises():
    with pytest.raises(ValueError):
        get_tool("not-a-real-tool")
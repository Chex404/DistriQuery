from distriquery.agents.planner import HeuristicQueryPlanner


def test_arithmetic_routes_to_calculator_tool():
    planner = HeuristicQueryPlanner()

    decision = planner.plan("What is 12 + 7?")

    assert decision.strategy == "tool"
    assert decision.tool_name == "calculator"


def test_date_question_routes_to_current_date_tool():
    planner = HeuristicQueryPlanner()

    decision = planner.plan("What is today's date?")

    assert decision.strategy == "tool"
    assert decision.tool_name == "current_date"


def test_comparison_question_routes_to_multi_hop():
    planner = HeuristicQueryPlanner()

    decision = planner.plan("Compare Kafka and RabbitMQ for event streaming.")

    assert decision.strategy == "multi_hop"


def test_versus_phrasing_routes_to_multi_hop():
    planner = HeuristicQueryPlanner()

    decision = planner.plan("Qdrant vs Milvus for vector search")

    assert decision.strategy == "multi_hop"


def test_plain_factual_question_routes_to_direct():
    planner = HeuristicQueryPlanner()

    decision = planner.plan("What does the query planner decide between?")

    assert decision.strategy == "direct"
    assert decision.tool_name is None


def test_arithmetic_takes_priority_over_other_signals():
    planner = HeuristicQueryPlanner()

    # contains both an arithmetic expression AND could look ambiguous —
    # arithmetic should win since it's checked first
    decision = planner.plan("If I have 5 + 3 apples, how many do I have?")

    assert decision.strategy == "tool"
    assert decision.tool_name == "calculator"
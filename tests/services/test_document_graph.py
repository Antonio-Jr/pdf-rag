"""Tests for the LangGraph routing and structure."""

from langgraph.graph import StateGraph

from src.services.graphs.document_graph import generate_graph


def test_generate_graph_structure():
    """Test that the generated graph has the expected nodes and edges."""
    workflow = generate_graph()

    # Verify it returns a StateGraph object
    assert isinstance(workflow, StateGraph)

    # Verify nodes
    assert "chatbot" in workflow.nodes
    assert "tools" in workflow.nodes
    assert "summarizer" in workflow.nodes

    # Compile checking verifies edge validity
    app = workflow.compile()

    # We can't easily introspect internal edges in LangGraph 1.0 without
    # traversing the compiled graph schemas, but verifying it compiles
    # confirms the conditional edge logic is syntactically sound.
    assert app is not None

from langchain_core.messages import AIMessage

from app.services.nodes.discovery_node import discovery_node
from app.services.prompts.registry import registry
from app.services.states.graph_state import GraphState
from app.utils.llm_factory import get_model


def extraction_node(state: GraphState) -> dict:
    meta = state.discovery

    prompt = registry.extaction_processor.format(
        "universal_extraction",
        discovery_summary=meta.doc_purpose,
        tracking_points=", ".join(meta.data_points),
        content=state.raw_content,
    )

    model = get_model()
    result = model.invoke(prompt)

    final_response = (
        f"### Document Analysis: ({meta.tone})\n"
        f"**Purpose:** {meta.doc_purpose}\n"
        f"{result.content}"
    )

    return {
        "extracted_data": {"raw_json_or_text": result.content},
        "response": final_response,
        "messages": [AIMessage(content=final_response)]
    }
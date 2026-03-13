from typing import Dict, Any

from app.services.prompts.registry import registry
from app.services.rag.rag_service import get_retriever
from app.services.states.graph_state import GraphState, DocumentMetadata
from app.utils.llm_factory import get_model


def classifier_node(state: GraphState) -> Dict[str, Any]:
    retriever = get_retriever()
    docs = retriever.invoke(state=state.query)
    content = "\n\n".join([doc.page_content for doc in docs])

    model = get_model()
    prompt = registry.classifier_default.format("identify_type", content=content[:2000])

    result = model.invoke(prompt=prompt)
    doc_type = result.content.strip()

    return {
        "raw_content": content,
        "doc_context": DocumentMetadata(doc_type=doc_type),
    }
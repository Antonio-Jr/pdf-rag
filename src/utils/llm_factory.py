from langchain.chat_models import init_chat_model

from src.utils.log_wrapper import log_execution
from src.core.settings import settings


@log_execution
def get_model(
    model_provider: str | None = None,
    model_name: str | None = None,
    structured_schema: type | None = None,
    temperature: int | None = None,
    **kwargs,
):
    provider = (model_provider or settings.LLM_PROVIDER).lower()
    m_name = model_name or settings.LLM_MODEL_NAME

    target_base_url = None
    if provider in ["ollama", "openai"]:
        target_base_url = settings.LLM_BASE_URL

    llm = init_chat_model(
        model=m_name,
        model_provider=provider,
        base_url=target_base_url,
        temperature=temperature or settings.LLM_TEMPERATURE,
        api_key=settings.LLM_API_KEY,
        **kwargs,
    )

    if structured_schema:
        return llm.with_structured_output(structured_schema)

    return llm

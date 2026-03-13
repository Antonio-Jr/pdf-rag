"""LLM chat model factory.

Creates a LangChain chat model instance using ``init_chat_model``,
with optional structured output binding.  Supports Google GenAI,
Ollama, and OpenAI providers via a unified interface.
"""

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
    """Create and return a LangChain chat model instance.

    Falls back to application settings for any parameter not explicitly
    provided.  When a ``structured_schema`` is given, wraps the model
    with ``with_structured_output`` for typed responses.

    Args:
        model_provider: LLM provider override (e.g. ``"google_genai"``).
        model_name: Model name override (e.g. ``"gemini-1.5-pro"``).
        structured_schema: Optional Pydantic model class for structured
                           output parsing.
        temperature: Sampling temperature override.
        **kwargs: Additional keyword arguments forwarded to
                  ``init_chat_model``.

    Returns:
        A LangChain ``BaseChatModel`` instance, optionally wrapped
        for structured output.
    """
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

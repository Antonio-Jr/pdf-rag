from fastapi import APIRouter

from src.services.prompts.registry import registry
from src.core.settings import settings

router = APIRouter(prefix="/admin", tags=["Ops"])


@router.post("/prompts/refresh")
async def refresh_prompts():
    registry.reload()

    return {
        "status": "success",
        "external_path": str(settings.EXTERNAL_PROMPTS_DIR),
    }


@router.post("/config/status")
async def get_status():
    return {
        "provider": settings.LLM_PROVIDER,
        "model": settings.LLM_MODEL_NAME,
        "prompts_loaded": list(registry._namespaces.keys()),
    }

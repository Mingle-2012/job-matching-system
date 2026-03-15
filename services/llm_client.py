from openai import OpenAI

from config.settings import get_settings

settings = get_settings()


def create_openai_client() -> OpenAI | None:
    if not settings.openai_api_key:
        return None

    kwargs: dict = {
        "api_key": settings.openai_api_key,
        "timeout": settings.openai_timeout_seconds,
    }

    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url

    return OpenAI(**kwargs)


def llm_config_summary() -> dict[str, str | int | bool]:
    return {
        "api_key_configured": bool(settings.openai_api_key),
        "base_url": settings.openai_base_url,
        "chat_model": settings.openai_model,
        "embedding_model": settings.openai_embedding_model,
        "timeout_seconds": settings.openai_timeout_seconds,
    }

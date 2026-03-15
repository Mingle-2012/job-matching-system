from config.settings import get_settings
from services.llm_client import create_openai_client, llm_config_summary


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "***" + value[-4:]


def main() -> None:
    settings = get_settings()
    summary = llm_config_summary()
    print("=== LLM Config ===")
    print(f"api_key_configured: {summary['api_key_configured']}")
    print(f"api_key(masked): {_mask(settings.openai_api_key)}")
    print(f"base_url: {summary['base_url']}")
    print(f"chat_model: {summary['chat_model']}")
    print(f"embedding_model: {summary['embedding_model']}")
    print(f"timeout_seconds: {summary['timeout_seconds']}")

    client = create_openai_client()
    if not client:
        print("client: not initialized (missing api key)")
        return

    print("\n=== Connectivity Test ===")

    try:
        chat_resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Reply with exactly: ok"},
            ],
            temperature=0,
            max_tokens=8,
        )
        chat_text = chat_resp.choices[0].message.content if chat_resp.choices else ""
        print(f"chat: OK -> {chat_text}")
    except Exception as exc:
        print(f"chat: FAILED -> {type(exc).__name__}: {str(exc)[:300]}")

    model_flag = (settings.openai_embedding_model or "").strip().lower()
    if model_flag in {"", "none", "disabled", "off", "local"}:
        print(f"embedding: DISABLED -> model={settings.openai_embedding_model}")
        return

    try:
        emb_resp = client.embeddings.create(
            model=settings.openai_embedding_model,
            input="embedding health check",
        )
        dim = len(emb_resp.data[0].embedding)
        print(f"embedding: OK -> dim={dim}")
    except Exception as exc:
        print(f"embedding: FAILED -> {type(exc).__name__}: {str(exc)[:300]}")


if __name__ == "__main__":
    main()

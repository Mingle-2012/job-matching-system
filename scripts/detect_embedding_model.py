import argparse
import re
from pathlib import Path

from openai import OpenAI

from config.settings import get_settings

CANDIDATE_MODELS = [
    "ecnu-embedding",
    "ecnu_embedding",
    "ecnu-plus-embedding",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-v3",
    "embedding-3-large",
    "embedding-3-small",
    "bge-m3",
    "bge-large-zh-v1.5",
    "jina-embeddings-v2-base-zh",
]


def _create_client() -> OpenAI:
    settings = get_settings()
    kwargs: dict = {
        "api_key": settings.openai_api_key,
        "timeout": settings.openai_timeout_seconds,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


def _candidate_pool(client: OpenAI, current_model: str) -> list[str]:
    settings = get_settings()
    pool: list[str] = []
    if current_model:
        pool.append(current_model)
    if settings.openai_model:
        pool.append(settings.openai_model)

    try:
        model_resp = client.models.list()
        for item in model_resp.data:
            model_id = str(getattr(item, "id", "")).strip()
            if not model_id:
                continue
            if re.search(r"embed|embedding|bge|jina", model_id, flags=re.IGNORECASE):
                pool.append(model_id)
    except Exception:
        pass

    pool.extend(CANDIDATE_MODELS)

    # de-duplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for model in pool:
        key = model.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(model)
    return result


def detect_embedding_model() -> tuple[str | None, int | None, list[str]]:
    settings = get_settings()
    client = _create_client()

    failed: list[str] = []
    for model in _candidate_pool(client, settings.openai_embedding_model):
        try:
            resp = client.embeddings.create(model=model, input="embedding model probe")
            dim = len(resp.data[0].embedding)
            return model, dim, failed
        except Exception as exc:
            failed.append(f"{model}: {type(exc).__name__}")

    return None, None, failed


def update_env(env_path: Path, model: str) -> None:
    lines = env_path.read_text(encoding="utf-8").splitlines()
    updated = False
    for idx, line in enumerate(lines):
        if line.startswith("OPENAI_EMBEDDING_MODEL="):
            lines[idx] = f"OPENAI_EMBEDDING_MODEL={model}"
            updated = True
            break

    if not updated:
        lines.append(f"OPENAI_EMBEDDING_MODEL={model}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect available embedding model from current OpenAI-compatible endpoint")
    parser.add_argument("--env-path", default=".env", help="Path to env file")
    parser.add_argument("--update-env", action="store_true", help="Update env file with detected model")
    args = parser.parse_args()

    model, dim, failed = detect_embedding_model()

    if model:
        print(f"DETECTED_MODEL={model}")
        print(f"EMBEDDING_DIM={dim}")
        if args.update_env:
            env_path = Path(args.env_path)
            update_env(env_path, model)
            print(f"ENV_UPDATED={env_path}")
    else:
        print("DETECTED_MODEL=")
        print("EMBEDDING_DIM=")
        print("FAILED_TRIALS=")
        for item in failed:
            print(item)
        if args.update_env:
            env_path = Path(args.env_path)
            update_env(env_path, "disabled")
            print(f"ENV_UPDATED={env_path}")


if __name__ == "__main__":
    main()

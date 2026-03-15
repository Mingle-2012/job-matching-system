import hashlib
import math
from typing import List, Sequence

from config.settings import get_settings
from services.llm_client import create_openai_client

settings = get_settings()


class EmbeddingService:
    def __init__(self) -> None:
        self.client = create_openai_client()
        model_flag = (settings.openai_embedding_model or "").strip().lower()
        self.llm_enabled = self.client is not None and model_flag not in {"", "none", "disabled", "off", "local"}

    def embed_text(self, text: str) -> List[float]:
        if self.llm_enabled and self.client:
            try:
                response = self.client.embeddings.create(
                    model=settings.openai_embedding_model,
                    input=text[:20000],
                )
                vector = list(response.data[0].embedding)
                return self._align_vector_dim(vector)
            except Exception:
                self.llm_enabled = False
        return self._deterministic_embedding(text)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.llm_enabled and self.client:
            try:
                response = self.client.embeddings.create(
                    model=settings.openai_embedding_model,
                    input=[text[:20000] for text in texts],
                )
                return [self._align_vector_dim(list(item.embedding)) for item in response.data]
            except Exception:
                self.llm_enabled = False

        return [self._deterministic_embedding(text) for text in texts]

    @staticmethod
    def _align_vector_dim(vector: List[float]) -> List[float]:
        target_dim = settings.vector_size
        current_dim = len(vector)

        if current_dim == target_dim:
            return vector
        if current_dim > target_dim:
            return vector[:target_dim]
        return vector + [0.0] * (target_dim - current_dim)

    @staticmethod
    def _deterministic_embedding(text: str) -> List[float]:
        dim = settings.vector_size
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
        values = [((digest[i % len(digest)] / 255.0) - 0.5) for i in range(dim)]
        norm = math.sqrt(sum(v * v for v in values)) + 1e-12
        return [v / norm for v in values]


embedding_service = EmbeddingService()

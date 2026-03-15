from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from config.settings import get_settings

settings = get_settings()


class QdrantVectorDB:
    def __init__(self) -> None:
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    def init_collections(self) -> None:
        for collection_name in (settings.candidate_collection, settings.job_collection):
            if self.client.collection_exists(collection_name=collection_name):
                continue
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qm.VectorParams(size=settings.vector_size, distance=qm.Distance.COSINE),
            )

    def upsert_candidate_chunks(
        self,
        candidate_id: int,
        vectors: Sequence[list[float]],
        chunks: Sequence[str],
    ) -> None:
        points: list[qm.PointStruct] = []
        for idx, (vector, chunk) in enumerate(zip(vectors, chunks)):
            point_id = candidate_id * 1_000_000 + idx
            points.append(
                qm.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "candidate_id": candidate_id,
                        "chunk_index": idx,
                        "text": chunk,
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=settings.candidate_collection, points=points, wait=True)

    def upsert_job_chunks(
        self,
        job_id: int,
        vectors: Sequence[list[float]],
        chunks: Sequence[str],
    ) -> None:
        points: list[qm.PointStruct] = []
        for idx, (vector, chunk) in enumerate(zip(vectors, chunks)):
            point_id = job_id * 1_000_000 + idx
            points.append(
                qm.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "job_id": job_id,
                        "chunk_index": idx,
                        "text": chunk,
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=settings.job_collection, points=points, wait=True)

    def search_candidates(
        self,
        query_vector: Sequence[float],
        top_k: int,
        candidate_ids: Sequence[int] | None = None,
    ) -> list[dict[str, Any]]:
        query_filter = None
        if candidate_ids:
            query_filter = qm.Filter(
                must=[qm.FieldCondition(key="candidate_id", match=qm.MatchAny(any=list(candidate_ids)))]
            )

        results = self.client.search(
            collection_name=settings.candidate_collection,
            query_vector=list(query_vector),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        merged: dict[int, float] = {}
        for point in results:
            candidate_id = int(point.payload.get("candidate_id"))
            score = float(point.score)
            if score > merged.get(candidate_id, float("-inf")):
                merged[candidate_id] = score

        return [
            {"candidate_id": candidate_id, "score": score}
            for candidate_id, score in sorted(merged.items(), key=lambda x: x[1], reverse=True)
        ]

    def search_jobs(
        self,
        query_vector: Sequence[float],
        top_k: int,
        job_ids: Sequence[int] | None = None,
    ) -> list[dict[str, Any]]:
        query_filter = None
        if job_ids:
            query_filter = qm.Filter(
                must=[qm.FieldCondition(key="job_id", match=qm.MatchAny(any=list(job_ids)))]
            )

        results = self.client.search(
            collection_name=settings.job_collection,
            query_vector=list(query_vector),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        merged: dict[int, float] = {}
        for point in results:
            job_id = int(point.payload.get("job_id"))
            score = float(point.score)
            if score > merged.get(job_id, float("-inf")):
                merged[job_id] = score

        return [
            {"job_id": job_id, "score": score}
            for job_id, score in sorted(merged.items(), key=lambda x: x[1], reverse=True)
        ]


vector_client = QdrantVectorDB()

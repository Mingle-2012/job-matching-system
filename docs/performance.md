# Performance Plan (1M Candidates / 100K Jobs)

## 1. Vector Index Strategy
- Use Qdrant HNSW index with cosine distance.
- Separate collections:
  - candidate_chunks
  - job_chunks
- Store only compact payload for hot path (ids and chunk index).
- For high throughput:
  - Batch upsert embeddings.
  - Use quantization for memory savings at scale.

## 2. Neo4j Query Optimization
- Keep unique constraints on Candidate.id, Job.id, Skill.name.
- Normalize skill names to lowercase to avoid cardinality explosion.
- Restrict traversal depth for SUB_SKILL_OF (0..2 or 0..3 max).
- Avoid returning full nodes in retrieval path; return ids + aggregate score only.

## 3. Cache Strategy
- Redis key patterns:
  - search:candidates:{job_id}:{filter_hash}
  - search:jobs:{candidate_id}:{filter_hash}
- TTL recommendation:
  - 60-180s for active searching.
  - Invalidate on profile/job update events.

## 4. Batch Embedding
- Process chunks in batches (e.g., 64-256 texts per API call).
- Queue ingestion using a worker system (Celery/RQ/Kafka consumer).
- Retry transient API failures with exponential backoff.

## 5. Async Search
- Run graph retrieval and vector retrieval concurrently.
- Keep RRF in-memory and O(n) in result size.
- Run LLM rerank only on top 10-20 pairs.
- Use fallback heuristic reranker when LLM quota is exhausted.

## 6. Data Partitioning and Lifecycle
- Shard by region or business unit if required.
- Archive inactive candidates/jobs to cold collections.
- Keep recent active entities in hot index for lower latency.

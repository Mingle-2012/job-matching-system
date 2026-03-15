# Example Search Flow

## A. Search Candidates by Job (POST /search/candidates)
1. Input: job_id=2001.
2. MySQL pre-filter:
   - location match
   - salary expectation <= job salary_max
   - degree and status filters
   Output: candidate_ids (e.g., 35,000 from 1M)
3. Dual-track retrieval:
   - Graph track (Neo4j): top 100 by skill match and sub-skill inference.
   - Vector track (Qdrant): top 100 by semantic similarity.
4. RRF fusion:
   score = sum(1 / (60 + rank))
   Output: fused top 10 candidate ids.
5. LLM rerank:
   - Evaluate each candidate/job pair
   - Return match_score + reason
6. API response example:
[
  {
    "candidate_id": 456,
    "score": 0.88,
    "reason": "Strong overlap on Python, FastAPI, Docker and backend architecture projects."
  }
]

## B. Search Jobs by Candidate (POST /search/jobs)
1. Input: candidate_id=456.
2. MySQL pre-filter jobs.
3. Neo4j + Qdrant retrieval in parallel.
4. RRF fusion.
5. LLM rerank and explanation.
6. Return ranked jobs.

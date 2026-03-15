# Hybrid Graph-Vector RAG Recruiting Architecture

## 1. Data Acquisition Layer
- Inputs: candidate resume uploads, job posting creation, company metadata sync.
- Source channels: admin portal, ATS events, HR batch import.
- Output contracts:
  - candidate raw text bundle
  - job raw text bundle
  - business metadata payload

## 2. Data Processing and Parsing Layer
- Text parser normalizes resume/job text blocks.
- Skill extractor calls LLM to output strict JSON:
  {
    "skills": ["Python", "FastAPI", "Docker"]
  }
- Chunker splits long text into semantic chunks for vector indexing.
- Embedding service batches chunk vectors.

## 3. Storage Layer
- MySQL (transactional metadata + business state)
  - Candidate, Job, Application, Company.
  - Structured filters and lifecycle states.
- Neo4j (recruitment knowledge graph)
  - Nodes: Candidate, Job, Skill, Company.
  - Relations:
    - (Candidate)-[:HAS_SKILL]->(Skill)
    - (Job)-[:REQUIRES_SKILL]->(Skill)
    - (Skill)-[:SUB_SKILL_OF]->(Skill)
    - (Candidate)-[:WORKED_AT]->(Company)
    - (Job)-[:POSTED_BY]->(Company)
  - Supports sub-skill inference and multi-hop graph retrieval.
- Qdrant (vector retrieval)
  - Candidate chunks and Job chunks.
  - Payload metadata includes candidate_id or job_id.

## 4. Retrieval and Matching Layer
- Step A: MySQL pre-filter for hard constraints.
- Step B: Dual-track retrieval.
  - Graph track: skill and hierarchy matches from Neo4j.
  - Vector track: semantic similarity from Qdrant.
- Step C: RRF fusion merges graph/vector ranked lists.
  - score = sum(1 / (k + rank)), k=60.

## 5. LLM Explanation and Rerank Layer
- Top-N pairs from RRF are sent to LLM for pairwise evaluation:
  1. Skill match
  2. Experience relevance
  3. Seniority alignment
  4. Soft skills
- Returns structured JSON:
  {
    "match_score": 0-100,
    "reason": "..."
  }
- Produces final explainable ranking.

## 6. API Service Layer (FastAPI)
- Ingestion APIs:
  - POST /ingest/candidate
  - POST /ingest/job
- Search APIs:
  - POST /search/candidates
  - POST /search/jobs
- Health:
  - GET /health
- Redis cache stores short-lived search responses and reduces repeated compute.

## End-to-End Text Diagram
Client/ATS -> FastAPI Ingestion -> Parser/Skill Extractor/Embedder
-> MySQL + Neo4j + Qdrant

Search Request -> FastAPI Search
-> MySQL Pre-filter
-> Neo4j Retrieval + Qdrant Retrieval (parallel)
-> RRF Fusion
-> LLM Rerank + Explanation
-> Response + Redis cache

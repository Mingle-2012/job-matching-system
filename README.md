# Hybrid Graph-Vector RAG Job Matching System

Production-ready baseline for bidirectional recruitment matching:
- Job -> Candidate search
- Candidate -> Job search
- Match score
- Explainable reasons

## Tech Stack
- FastAPI (Python backend)
- MySQL (business metadata)
- Neo4j (knowledge graph)
- Qdrant (vector retrieval)
- Redis (cache)
- OpenAI API (skill extraction + reranking)

## Project Structure
job-matching-system/
|- api/
|  |- main.py
|  |- routes.py
|- services/
|  |- graph_search.py
|  |- vector_search.py
|  |- hybrid_scoring.py
|  |- gt_rule_store.py
|  |- rrf.py
|  |- reranker.py
|  |- embedding.py
|  |- cache.py
|- ingestion/
|  |- parser.py
|  |- skill_extractor.py
|- database/
|  |- mysql.py
|  |- neo4j.py
|  |- vector_db.py
|- config/
|  |- settings.py
|- docs/
|  |- architecture.md
|  |- schema.sql
|  |- neo4j_examples.cypher
|  |- performance.md
|  |- deployment.md
|  |- example_search_flow.md
|- scripts/
|  |- bootstrap_skill_graph.py
|  |- evaluate_cv_dataset.py
|  |- learn_gt_rules.py
|- requirements.txt
|- docker-compose.yml
|- Dockerfile

## Quick Start
1. Copy environment variables:
   - cp .env.example .env
2. Fill OPENAI_API_KEY in .env.
3. Start all services:
   - docker compose up -d --build
4. Check API health:
   - GET http://localhost:8000/health

## Ingestion Pipeline
1. Upload candidate/job payload.
2. Parse text blocks.
3. Extract skills via LLM.
4. Chunk + embed text.
5. Write graph relations into Neo4j.
6. Write vectors to Qdrant.
7. Persist metadata into MySQL.

### Candidate Ingestion API
POST /ingest/candidate

Example body:
{
  "name": "Alice",
  "location": "Shanghai",
  "years_experience": 6,
  "salary_expectation": 45,
  "degree": "master",
  "job_status": "open_to_work",
  "resume_summary": "Backend engineer with Python and FastAPI",
  "project_experience": "Built microservices and event-driven platform",
  "achievements": "Led 5-person team",
  "skills": ["Python", "FastAPI", "Docker"],
  "company_name": "Acme"
}

### Job Ingestion API
POST /ingest/job

Example body:
{
  "title": "Senior Backend Engineer",
  "company_name": "Acme",
  "location": "Shanghai",
  "salary_range": "35-55",
  "degree_required": "bachelor",
  "status": "open",
  "job_description": "Build backend platform with Python and microservices",
  "responsibilities": "Design architecture and mentor team",
  "preferred_qualifications": "FastAPI, Docker, Kubernetes",
  "skills": ["Python", "FastAPI", "Docker", "Kubernetes"]
}

## Search Pipeline
1. Hard filter in MySQL + rule filter:
  - location / degree / experience / category
2. Dual-track retrieval in parallel:
   - Graph track (Neo4j)
   - Vector track (Qdrant)
3. RRF fusion (graph + vector).
4. Hybrid rule scoring (skill/vector/role/experience/domain).
5. Top-N rerank and explanation via LLM (with GT learned rules).

## GT Learning Workflow
Recommended loop:
1. Learn GT rules with LLM.
2. Generate per-job scoring weights.
3. Inject rules into online scoring + reranking.
4. Re-run benchmark.

Generate rules (Prompt1 + Prompt3):

python -m scripts.learn_gt_rules --dataset-dir ../dataset --output ../dataset/gt_learned_rules.json

Generate rules + per-resume labels (Prompt2):

python -m scripts.learn_gt_rules --dataset-dir ../dataset --output ../dataset/gt_learned_rules_with_labels.json --include-labels

Enable learned rules in `.env`:

GT_RULE_ENABLED=true
GT_RULE_FILE=/dataset/gt_learned_rules.json

For cross-discipline lexical expansion (optional):

DOMAIN_TAXONOMY_FILE=/dataset/domain_taxonomy.json

Reference format:

- docs/domain_taxonomy.example.json

### Search Candidates
POST /search/candidates

Input:
{
  "job_id": 123
}

Output:
[
  {
    "candidate_id": 456,
    "score": 0.88,
    "reason": "Strong overlap on Python, FastAPI and backend architecture experience."
  }
]

### Search Jobs
POST /search/jobs

Input:
{
  "candidate_id": 456
}

Output:
[
  {
    "job_id": 123,
    "score": 0.92,
    "reason": "Candidate has skills and experience matching role requirements and seniority."
  }
]

## RRF Algorithm
Implemented in services/rrf.py

score(d) = sum(1 / (k + rank_i(d)))

Recommended k = 60.

## Neo4j Skill Inheritance
Example:
- React -> JavaScript
- PyTorch -> Python

Bootstrap examples:
- python scripts/bootstrap_skill_graph.py

## Performance Notes
See docs/performance.md for scaling to:
- 1,000,000 candidates
- 100,000 jobs

## Deployment
See docs/deployment.md and docker-compose.yml.

## Dataset Evaluation (cv_gt.csv)
If your workspace has:
- ../dataset/cv_gt.csv
- ../dataset/<岗位名称>/*.pdf

You can run an end-to-end benchmark:

python -m scripts.evaluate_cv_dataset --api-base http://localhost:8000 --top-k 10

The script will:
1. Ingest resume PDFs.
2. Ingest jobs from csv responsibilities.
3. Query /search/candidates for each job.
4. Compute Jaccard@|GT|, Precision@K, Recall@K, F1@K, MAP@K, nDCG@K.

Report output:
- ../dataset/eval_report.json

See details in docs/evaluation.md.

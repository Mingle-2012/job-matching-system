# Evaluation Guide for cv_gt Dataset

## Dataset Assumption
- dataset/cv_gt.csv
- dataset/<岗位名称>/*.pdf

Each folder contains resumes that applied to that role.

## Recommended Evaluation Flow
1. Ingest all resumes from dataset folders.
2. Ingest each job from cv_gt.csv using 岗位职责 as description text.
3. Query backend with POST /search/candidates by job_id.
4. Compare predicted candidates with ground truth from 简历初筛通过人员.
5. Compute both set-based and ranking-based metrics.

## Why Not Only Jaccard
Jaccard is useful for overlap quality but ignores rank order.
For retrieval systems, ranking quality matters, so combine:
- Jaccard@|GT|: set overlap
- Precision@K, Recall@K, F1@K: hit quality at top K
- MAP@K: precision across relevant hit positions
- nDCG@K: emphasizes putting relevant resumes higher
- Hired Recall@K: ability to cover 入职人员 in top K

## Formula
- Jaccard:
  score = |P ∩ G| / |P ∪ G|
- Precision@K:
  |TopK ∩ G| / K
- Recall@K:
  |TopK ∩ G| / |G|
- nDCG@K:
  DCG@K / IDCG@K

## Run
python scripts/evaluate_cv_dataset.py --api-base http://localhost:8000 --top-k 10

Output report JSON:
- dataset/eval_report.json

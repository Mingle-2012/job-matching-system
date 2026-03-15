from typing import List

from config.settings import get_settings
from database.neo4j import graph_client

settings = get_settings()


class GraphSearchService:
    def retrieve_candidates_for_job(self, job_id: int, candidate_ids: List[int] | None = None) -> List[dict]:
        job_skills = graph_client.get_job_skills(job_id)
        if not job_skills:
            return []
        return graph_client.search_candidates_by_job_skills(
            job_skills=job_skills,
            candidate_ids=candidate_ids,
            limit=settings.graph_top_k,
        )

    def retrieve_jobs_for_candidate(self, candidate_id: int, job_ids: List[int] | None = None) -> List[dict]:
        candidate_skills = graph_client.get_candidate_skills(candidate_id)
        if not candidate_skills:
            return []
        return graph_client.search_jobs_by_candidate_skills(
            candidate_skills=candidate_skills,
            job_ids=job_ids,
            limit=settings.graph_top_k,
        )


graph_search_service = GraphSearchService()

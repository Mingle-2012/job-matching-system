from collections.abc import Sequence
from typing import Any

from neo4j import GraphDatabase

from config.settings import get_settings

settings = get_settings()


class Neo4jGraph:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def init_constraints(self) -> None:
        constraints = [
            "CREATE CONSTRAINT candidate_id_unique IF NOT EXISTS FOR (c:Candidate) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT job_id_unique IF NOT EXISTS FOR (j:Job) REQUIRE j.id IS UNIQUE",
            "CREATE CONSTRAINT skill_name_unique IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT company_id_unique IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for cql in constraints:
                session.run(cql)

    def upsert_candidate_skills(self, candidate_id: int, skills: Sequence[str]) -> None:
        query = """
        MERGE (c:Candidate {id: $candidate_id})
        WITH c, $skills AS skill_list
        UNWIND skill_list AS skill_name
        WITH c, toLower(trim(skill_name)) AS normalized_skill
        WHERE normalized_skill <> ''
        MERGE (s:Skill {name: normalized_skill})
        MERGE (c)-[:HAS_SKILL]->(s)
        """
        with self.driver.session() as session:
            session.run(query, candidate_id=candidate_id, skills=list(skills))

    def upsert_job_skills(self, job_id: int, skills: Sequence[str]) -> None:
        query = """
        MERGE (j:Job {id: $job_id})
        WITH j, $skills AS skill_list
        UNWIND skill_list AS skill_name
        WITH j, toLower(trim(skill_name)) AS normalized_skill
        WHERE normalized_skill <> ''
        MERGE (s:Skill {name: normalized_skill})
        MERGE (j)-[:REQUIRES_SKILL]->(s)
        """
        with self.driver.session() as session:
            session.run(query, job_id=job_id, skills=list(skills))

    def link_candidate_company(self, candidate_id: int, company_id: int, company_name: str) -> None:
        query = """
        MERGE (c:Candidate {id: $candidate_id})
        MERGE (co:Company {id: $company_id})
        SET co.name = $company_name
        MERGE (c)-[:WORKED_AT]->(co)
        """
        with self.driver.session() as session:
            session.run(query, candidate_id=candidate_id, company_id=company_id, company_name=company_name)

    def link_job_company(self, job_id: int, company_id: int, company_name: str) -> None:
        query = """
        MERGE (j:Job {id: $job_id})
        MERGE (co:Company {id: $company_id})
        SET co.name = $company_name
        MERGE (j)-[:POSTED_BY]->(co)
        """
        with self.driver.session() as session:
            session.run(query, job_id=job_id, company_id=company_id, company_name=company_name)

    def add_skill_hierarchy(self, child_skill: str, parent_skill: str) -> None:
        query = """
        MERGE (child:Skill {name: toLower(trim($child_skill))})
        MERGE (parent:Skill {name: toLower(trim($parent_skill))})
        MERGE (child)-[:SUB_SKILL_OF]->(parent)
        """
        with self.driver.session() as session:
            session.run(query, child_skill=child_skill, parent_skill=parent_skill)

    def get_job_skills(self, job_id: int) -> list[str]:
        query = """
        MATCH (j:Job {id: $job_id})-[:REQUIRES_SKILL]->(s:Skill)
        RETURN collect(DISTINCT s.name) AS skills
        """
        with self.driver.session() as session:
            record = session.run(query, job_id=job_id).single()
            if not record:
                return []
            return list(record.get("skills", []))

    def get_candidate_skills(self, candidate_id: int) -> list[str]:
        query = """
        MATCH (c:Candidate {id: $candidate_id})-[:HAS_SKILL]->(s:Skill)
        RETURN collect(DISTINCT s.name) AS skills
        """
        with self.driver.session() as session:
            record = session.run(query, candidate_id=candidate_id).single()
            if not record:
                return []
            return list(record.get("skills", []))

    def search_candidates_by_job_skills(
        self,
        job_skills: Sequence[str],
        limit: int,
        candidate_ids: Sequence[int] | None = None,
    ) -> list[dict[str, Any]]:
        query = """
        UNWIND $job_skills AS job_skill_name
        MATCH (target:Skill {name: toLower(trim(job_skill_name))})
        OPTIONAL MATCH (target)<-[:SUB_SKILL_OF*0..2]-(expanded:Skill)
        WITH collect(DISTINCT target.name) + collect(DISTINCT expanded.name) AS expanded_skills
        UNWIND expanded_skills AS skill_name
        MATCH (c:Candidate)-[:HAS_SKILL]->(s:Skill)
        WHERE s.name = skill_name
        AND ($candidate_ids IS NULL OR c.id IN $candidate_ids)
        RETURN c.id AS candidate_id, count(DISTINCT s) AS skill_match
        ORDER BY skill_match DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(
                query,
                job_skills=list(job_skills),
                candidate_ids=list(candidate_ids) if candidate_ids else None,
                limit=limit,
            )
            return [
                {"candidate_id": int(record["candidate_id"]), "score": float(record["skill_match"])}
                for record in result
            ]

    def search_jobs_by_candidate_skills(
        self,
        candidate_skills: Sequence[str],
        limit: int,
        job_ids: Sequence[int] | None = None,
    ) -> list[dict[str, Any]]:
        query = """
        UNWIND $candidate_skills AS candidate_skill_name
        MATCH (source:Skill {name: toLower(trim(candidate_skill_name))})
        OPTIONAL MATCH (source)-[:SUB_SKILL_OF*0..2]->(expanded:Skill)
        WITH collect(DISTINCT source.name) + collect(DISTINCT expanded.name) AS expanded_skills
        UNWIND expanded_skills AS skill_name
        MATCH (j:Job)-[:REQUIRES_SKILL]->(s:Skill)
        WHERE s.name = skill_name
        AND ($job_ids IS NULL OR j.id IN $job_ids)
        RETURN j.id AS job_id, count(DISTINCT s) AS skill_match
        ORDER BY skill_match DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(
                query,
                candidate_skills=list(candidate_skills),
                job_ids=list(job_ids) if job_ids else None,
                limit=limit,
            )
            return [{"job_id": int(record["job_id"]), "score": float(record["skill_match"])} for record in result]


graph_client = Neo4jGraph()

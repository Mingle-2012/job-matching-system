from sqlalchemy import delete

from config.settings import get_settings
from database.mysql import Application, Candidate, Company, Job, SessionLocal
from database.neo4j import graph_client
from database.vector_db import vector_client
from services.cache import cache_client

settings = get_settings()

SKILL_ONTOLOGY_EDGES = [
    ("creo", "cad"),
    ("ug", "cad"),
    ("模具", "结构设计"),
    ("结构设计", "机械设计"),
]


def reset_mysql() -> None:
    db = SessionLocal()
    try:
        db.execute(delete(Application))
        db.execute(delete(Job))
        db.execute(delete(Candidate))
        db.execute(delete(Company))
        db.commit()
    finally:
        db.close()


def reset_neo4j() -> None:
    with graph_client.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    graph_client.init_constraints()
    for child, parent in SKILL_ONTOLOGY_EDGES:
        graph_client.add_skill_hierarchy(child, parent)


def reset_qdrant() -> None:
    for collection_name in [settings.candidate_collection, settings.job_collection]:
        if vector_client.client.collection_exists(collection_name=collection_name):
            vector_client.client.delete_collection(collection_name=collection_name)
    vector_client.init_collections()


def reset_cache() -> None:
    try:
        cache_client.client.flushdb()
    except Exception:
        pass


def main() -> None:
    reset_mysql()
    reset_neo4j()
    reset_qdrant()
    reset_cache()
    print("benchmark state reset complete")


if __name__ == "__main__":
    main()

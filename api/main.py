import time
from typing import Callable

from fastapi import FastAPI

from api.routes import router
from config.settings import get_settings
from database.mysql import init_mysql
from database.neo4j import graph_client
from database.vector_db import vector_client

settings = get_settings()

app = FastAPI(title=settings.app_name)
app.include_router(router)

SKILL_ONTOLOGY_EDGES = [
    ("creo", "cad"),
    ("ug", "cad"),
    ("模具", "结构设计"),
    ("结构设计", "机械设计"),
]


def _retry_startup_step(name: str, fn: Callable[[], None], attempts: int = 30, delay_seconds: int = 2) -> None:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            fn()
            return
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(delay_seconds)
    raise RuntimeError(f"startup step failed: {name}") from last_error


def _init_skill_ontology() -> None:
    for child, parent in SKILL_ONTOLOGY_EDGES:
        graph_client.add_skill_hierarchy(child, parent)


@app.on_event("startup")
def on_startup() -> None:
    _retry_startup_step("mysql init", init_mysql)
    _retry_startup_step("neo4j init", graph_client.init_constraints)
    _retry_startup_step("neo4j ontology init", _init_skill_ontology)
    _retry_startup_step("qdrant init", vector_client.init_collections)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

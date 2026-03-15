from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "hybrid-rag-job-matching"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "app"
    mysql_password: str = "app_password"
    mysql_database: str = "job_matching"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_password"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    candidate_collection: str = "candidate_chunks"
    job_collection: str = "job_chunks"
    vector_size: int = 3072

    redis_url: str = "redis://localhost:6379/0"

    openai_api_key: str = Field(default="")
    openai_base_url: str = ""
    openai_model: str = "ecnu-plus"
    openai_embedding_model: str = "ecnu"
    openai_timeout_seconds: int = 60

    graph_top_k: int = 100
    vector_top_k: int = 100
    rrf_k: int = 60
    rerank_top_n: int = 10
    llm_rerank_top_n: int = 20
    llm_rerank_weight: float = 0.35
    prefilter_limit: int = 5000

    gt_rule_enabled: bool = True
    gt_rule_file: str = "/dataset/gt_learned_rules.json"
    gt_name_boost_weight: float = 0.0
    domain_taxonomy_file: str = ""

    hard_filter_year_tolerance: float = 0.5
    role_mismatch_penalty: float = -0.3
    hybrid_weight_skill: float = 0.4
    hybrid_weight_vector: float = 0.3
    hybrid_weight_role: float = 0.2
    hybrid_weight_experience: float = 0.1
    hybrid_weight_domain: float = 0.0

    @property
    def mysql_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}?charset=utf8mb4"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()

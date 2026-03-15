import json
from typing import Any

import redis

from config.settings import get_settings

settings = get_settings()


class RedisCache:
    def __init__(self) -> None:
        self.client = redis.Redis.from_url(settings.redis_url, decode_responses=True)

    def get_json(self, key: str) -> Any | None:
        try:
            payload = self.client.get(key)
            if not payload:
                return None
            return json.loads(payload)
        except Exception:
            return None

    def set_json(self, key: str, value: Any, ttl_seconds: int = 120) -> None:
        try:
            self.client.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=True))
        except Exception:
            return


cache_client = RedisCache()

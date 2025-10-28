import time
from typing import Dict, Any, List

from src.optimizers.base import BaseOptimizer


class PostgresOptimizer(BaseOptimizer):
    def __init__(self):
        super().__init__(optimizer_name="PostgreSQL_Default")

    def get_plan(self, query_sql: str) -> Dict[str, Any]:
        start_time = time.perf_counter()

        plan_sql = query_sql

        settings: List[str] = []

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        return {
            "plan_sql": plan_sql,
            "settings": settings,
            "inference_time_ms": inference_time_ms,
        }


class PostgresGEQOOptimizer(BaseOptimizer):
    def __init__(self, geqo_threshold: int = 2):
        super().__init__(optimizer_name=f"PostgreSQL_GEQO(t={geqo_threshold})")
        self._geqo_threshold = geqo_threshold

    def get_plan(self, query_sql: str) -> Dict[str, Any]:
        start_time = time.perf_counter()

        plan_sql = query_sql

        settings: List[str] = [
            "SET geqo = 'on';",
            f"SET geqo_threshold = '{self._geqo_threshold}';",
        ]

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        return {
            "plan_sql": plan_sql,
            "settings": settings,
            "inference_time_ms": inference_time_ms,
        }

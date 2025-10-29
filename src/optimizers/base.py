import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable


class BaseOptimizer(ABC):
    def __init__(self, optimizer_name: str):
        self.optimizer_name = optimizer_name

        self._model_path: str = ""
        self._featurizer_path: str = ""

        os.makedirs("models", exist_ok=True)

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def featurizer_path(self) -> str:
        return self._featurizer_path

    def train(self, training_data_path: str, config: Optional[Dict[str, Any]] = None):
        print(f"[{self.optimizer_name}] Training not applicable")
        pass

    @abstractmethod
    def get_plan(self, query_sql: str) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _time_inference(func: Callable, *args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        inference_time_ms = (end_time - start_time) * 1000
        return result, inference_time_ms

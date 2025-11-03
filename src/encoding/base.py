from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any

import joblib
import numpy as np


class BaseFeaturizer(ABC):
    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, plan_jsons: List[Dict[str, Any]], query_sqls: List[str]):
        raise NotImplementedError

    @abstractmethod
    def transform(self, plan_json: Dict[str, Any], query_sql: str) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(
        self, plan_jsons: List[Dict[str, Any]], query_sqls: List[str]
    ) -> np.ndarray:
        self.fit(plan_jsons, query_sqls)

        vectors = [self.transform(p, q) for p, q in zip(plan_jsons, query_sqls)]
        return np.array(vectors)

    def save(self, filepath: str):
        if not self.is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted featurizer. " "Call .fit() first."
            )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Featurizer saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> BaseFeaturizer:
        print(f"Loading featurizer from {filepath}")
        featurizer = joblib.load(filepath)
        if not featurizer.is_fitted:
            print("Warning: Loaded featurizer is not fitted")

        return featurizer

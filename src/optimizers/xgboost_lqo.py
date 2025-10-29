import os
import time
from typing import Dict, Any, Optional

import numpy as np
import xgboost as xgb
from tqdm import tqdm

from src.db.connector import DatabaseConnector
from src.db.executor import get_plan_json
from src.encoding.base import BaseFeaturizer
from src.encoding.featurizers import SimpleFeaturizer
from src.optimizers.base import BaseOptimizer
from src.optimizers.baselines import PostgresOptimizer
from src.pipeline.data_generator import get_plan_forcing_configs
from src.pipeline.data_utils import load_training_data


class XGBoostLQO(BaseOptimizer):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(optimizer_name="XGBoostLQO")

        self._config = config or {}

        # 1. Model & Featurizer Artifacts
        self._model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self._config.get("n_estimators", 100),
            learning_rate=self._config.get("learning_rate", 0.1),
            max_depth=self._config.get("max_depth", 6),
            n_jobs=-1,  # Use all cores
            random_state=42,
        )
        self._featurizer: BaseFeaturizer = SimpleFeaturizer()
        self._is_trained = False

        # 2. Paths for saving/loading
        self._model_path = "models/xgboost_lqo.xgb"
        self._featurizer_path = "models/xgboost_featurizer.joblib"

        # 3. DB Connector for inference
        self._db_connector = DatabaseConnector()

        # 4. Candidate Plan Generator
        self._plan_configs = list(get_plan_forcing_configs())

    def train(self, training_data_path: str, config: Optional[Dict[str, Any]] = None):
        if config:
            self._config.update(config)  # Update model params

        print(f"[{self.optimizer_name}] Starting training...")

        # 1. Load data
        plan_jsons, y_log_latencies = load_training_data(training_data_path)
        print(f"Loaded {len(plan_jsons)} training samples.")

        # 2. Fit featurizer
        print("Fitting featurizer...")
        self._featurizer.fit(plan_jsons)

        # 3. Transform plans
        print("Transforming plans into feature vectors...")
        X = np.array([self._featurizer.transform(p) for p in tqdm(plan_jsons)])
        y = y_log_latencies

        # 4. Train model
        print(
            f"Training XGBoost model ({self._config.get('n_estimators', 100)} estimators)..."
        )
        self._model.fit(X, y)
        print("Model training complete.")

        # 5. Save artifacts
        self._model.save_model(self._model_path)
        print(f"Model saved to {self._model_path}")
        self._featurizer.save(self._featurizer_path)

        self._is_trained = True

    def get_plan(self, query_sql: str) -> Dict[str, Any]:
        if not self._is_trained:
            self._load_model()

        start_time = time.perf_counter()

        # 1. Generate candidate plan JSONs
        candidate_plans = []  # Stores (settings, plan_json)
        for settings in self._plan_configs:
            # Use the lightweight get_plan_json
            plan_json = get_plan_json(self._db_connector, query_sql, list(settings))
            if plan_json:
                candidate_plans.append((settings, plan_json))

        if not candidate_plans:
            # Fallback to default PG if no valid plans found
            print(
                f"[{self.optimizer_name}] Warning: No valid candidate plans found. Falling back to default."
            )
            return PostgresOptimizer().get_plan(query_sql)

        # 2. Featurize all valid plans
        features_list = [
            self._featurizer.transform(p_json) for _, p_json in candidate_plans
        ]
        X_candidates = np.array(features_list)

        # 3. Predict latency for all candidates in a batch
        predicted_log_latencies = self._model.predict(X_candidates)

        # 4. Find the best plan
        # The index of the minimum predicted log-latency
        best_plan_index = np.argmin(predicted_log_latencies)

        # Get the 'settings' for that best plan
        best_settings = candidate_plans[best_plan_index][0]

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        return {
            "plan_sql": query_sql,
            "settings": list(best_settings),
            "inference_time_ms": inference_time_ms,
        }

    def _load_model(self):
        if not (
            os.path.exists(self._model_path) and os.path.exists(self._featurizer_path)
        ):
            raise FileNotFoundError(
                f"Model artifacts not found. "
                f"Run .train() or place files at {self._model_path} "
                f"and {self._featurizer_path}"
            )

        print(f"[{self.optimizer_name}] Loading pre-trained model...")
        self._model.load_model(self._model_path)
        self._featurizer = BaseFeaturizer.load(self._featurizer_path)
        self._is_trained = True

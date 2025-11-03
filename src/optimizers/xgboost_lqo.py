import concurrent.futures
import os
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import xgboost as xgb
from tqdm import tqdm

from src.db.connector import DatabaseConnector
from src.db.executor import get_plan_json
from src.encoding.base import BaseFeaturizer
from src.encoding.featurizers import SimpleFeaturizer
from src.optimizers.base import BaseOptimizer
from src.optimizers.baselines import PostgresOptimizer
from src.optimizers.plan_generator import get_strategic_plan_configs
from src.pipeline.data_generator import get_plan_forcing_configs
from src.pipeline.data_utils import load_training_data


class XGBoostLQO(BaseOptimizer):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(optimizer_name="XGBoostLQO")

        self._config = config or {}

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

        self._model_path = "models/xgboost_lqo.xgb"
        self._featurizer_path = "models/xgboost_featurizer.joblib"

        self._training_plan_configs = list(get_plan_forcing_configs())
        self._inference_plan_configs = get_strategic_plan_configs()

        self._max_workers = os.cpu_count() or 4

    def train(self, training_data_path: str, config: Optional[Dict[str, Any]] = None):
        if config:
            self._config.update(config)  # Update model params

        print(f"[{self.optimizer_name}] Starting training...")

        # 1. Load data
        plan_jsons, query_sqls, y_log_latencies = load_training_data(training_data_path)
        print(f"Loaded {len(plan_jsons)} training samples.")

        # 2. Fit featurizer
        print("Fitting featurizer...")
        self._featurizer.fit(plan_jsons, query_sqls)

        # 3. Transform plans
        print("Transforming plans into feature vectors...")
        X = np.array(
            [
                self._featurizer.transform(p, q)
                for p, q in tqdm(zip(plan_jsons, query_sqls), total=len(plan_jsons))
            ]
        )
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
            try:
                self._load_model()
            except FileNotFoundError:
                return PostgresOptimizer().get_plan(query_sql)

        start_time = time.perf_counter()

        candidate_data = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            jobs = {
                executor.submit(
                    self._get_plan_and_features, query_sql, settings
                ): settings
                for settings in self._inference_plan_configs
            }

            for future in concurrent.futures.as_completed(jobs):
                result = future.result()
                if result:
                    candidate_data.append(result)

        if not candidate_data:
            return PostgresOptimizer().get_plan(query_sql)

        settings_list = [data[0] for data in candidate_data]
        features_list = [data[2] for data in candidate_data]
        X_candidates = np.array(features_list)

        predicted_log_latencies = self._model.predict(X_candidates)

        best_plan_index = np.argmin(predicted_log_latencies)
        best_settings = settings_list[best_plan_index]

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

    def _get_plan_and_features(
        self, query_sql: str, settings: Tuple[str, ...]
    ) -> Optional[Tuple[Tuple[str, ...], Dict[str, Any], np.ndarray]]:
        db_conn = None
        try:
            db_conn = DatabaseConnector()

            plan_json = get_plan_json(db_conn, query_sql, list(settings))
            if not plan_json:
                return None

            features = self._featurizer.transform(plan_json, query_sql)
            return settings, plan_json, features
        except Exception as e:
            print(f"[Worker Error] {e}")
            return None
        finally:
            if db_conn:
                db_conn.release_connection(db_conn.get_connection())

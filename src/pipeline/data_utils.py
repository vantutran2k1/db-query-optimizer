from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd


def load_training_data(parquet_path: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    try:
        df = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        print(f"Error: Training data file not found at {parquet_path}")
        raise

    plan_jsons = df["plan_json"].tolist()

    latencies = df["actual_latency_ms"].values + 1e-6
    log_latencies = np.log(latencies)

    return plan_jsons, log_latencies

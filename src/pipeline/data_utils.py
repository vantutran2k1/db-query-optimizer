from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from src.benchmark import load_benchmark_queries


def load_training_data(
    parquet_path: str, benchmark_name: str = "job"
) -> Tuple[List[Dict[str, Any]], List[str], np.ndarray]:
    try:
        df = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        print(f"Error: Training data file not found at {parquet_path}")
        raise

    print(f"Loading query SQLs from benchmark '{benchmark_name}'")
    all_queries = load_benchmark_queries(benchmark_name)
    query_sql_map = {q.query_name: q.query_sql for q in all_queries}

    query_sqls = df["query_name"].map(query_sql_map).tolist()

    if any(q is None for q in query_sqls):
        missing_count = sum(1 for q in query_sqls if q is None)
        print(
            f"Error: Could not find SQL for {missing_count} queries. "
            "Is the benchmark_name correct?"
        )
        raise ValueError("Failed to map query names to SQL.")

    plan_jsons = df["plan_json"].tolist()

    latencies = df["actual_latency_ms"].values + 1e-6
    log_latencies = np.log(latencies)

    return plan_jsons, query_sqls, log_latencies

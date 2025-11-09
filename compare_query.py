import json
import os
import sys
from typing import Any, Dict, List

import click
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.db.connector import DatabaseConnector
from src.db.executor import QueryExecutionError, measure_query
from src.optimizers import PostgresGEQOOptimizer, PostgresOptimizer, XGBoostLQO
from src.optimizers.base import BaseOptimizer


def initialize_optimizers() -> List[BaseOptimizer]:
    """Initializes and returns a list of all optimizers to compare."""
    print("Initializing optimizers...")

    # 1. Initialize Baselines
    pg_default = PostgresOptimizer()
    pg_geqo = PostgresGEQOOptimizer(geqo_threshold=2)

    # 2. Initialize LQO and load its trained model
    lqo = XGBoostLQO()
    try:
        lqo.load_model()  # Load the .xgb and .joblib files
        print("Successfully loaded trained XGBoostLQO model.")
    except FileNotFoundError:
        print(f"Warning: Could not load trained model for {lqo.optimizer_name}.")
        print("Will run as untrained (if possible) or fail.")

    return [pg_default, pg_geqo, lqo]


def run_comparison(
    query_sql: str, optimizers: List[BaseOptimizer]
) -> List[Dict[str, Any]]:
    """
    Runs a head-to-head comparison for a single query.
    """
    db_conn = None
    results = []

    try:
        db_conn = DatabaseConnector()
        print(f"\n--- Comparing plans for query: ---\n{query_sql}\n")

        for optimizer in optimizers:
            print(f"Running {optimizer.optimizer_name}...")

            # 1. Get Plan (Inference)
            plan_dict = optimizer.get_plan(query_sql)

            # 2. Measure Plan (Execution)
            measurement = measure_query(
                db_connector=db_conn,
                query_sql=plan_dict["plan_sql"],
                settings=plan_dict["settings"],
            )

            # 3. Combine results
            if measurement:
                total_time = (
                    plan_dict["inference_time_ms"] + measurement["actual_latency_ms"]
                )
                result_row = {
                    "Optimizer": optimizer.optimizer_name,
                    "Inference Time (ms)": plan_dict["inference_time_ms"],
                    "Execution Latency (ms)": measurement["actual_latency_ms"],
                    "Total End-to-End (ms)": total_time,
                    "Estimated Cost": measurement["estimated_cost"],
                    "plan_json": measurement["plan_json"],
                    "full_explain_json": measurement["full_explain_json"],
                }
                results.append(result_row)

    except QueryExecutionError as e:
        print(f"  [ERROR] Failed to measure: {e.message}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if db_conn:
            db_conn.close_all()

    return results


def present_results(results: List[Dict[str, Any]]):
    """
    Prints a formatted table and saves plan files.
    """
    if not results:
        print("No results to display.")
        return

    print("\n--- 1. Performance Summary ---")

    df = pd.DataFrame(results)

    df_display = df.drop(columns=["plan_json", "full_explain_json"])
    print(df_display.to_string(index=False))

    print("\n--- 2. Plan JSON Files (for Visualization) ---")

    output_dir = "results/comparison_plans"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for res in results:
        optimizer_name = res["Optimizer"]
        full_json_to_save = res["full_explain_json"]

        filename = os.path.join(output_dir, f"{optimizer_name}_plan.json")
        with open(filename, "w") as f:
            json.dump(full_json_to_save, f, indent=2)
        print(f"Saved plan for {optimizer_name} to: {filename}")


@click.command()
@click.option(
    "--query", prompt="Enter your SQL query", help="The SQL query to compare."
)
def main(query: str):
    """
    Compares the plans chosen by all registered optimizers
    for a single input query.
    """
    if not query.strip():
        print("Query cannot be empty.")
        return

    optimizers = initialize_optimizers()
    results = run_comparison(query, optimizers)
    present_results(results)


if __name__ == "__main__":
    main()

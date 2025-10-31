import os
import sys
import time
from typing import List, Dict, Any, Type

import click
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.connector import DatabaseConnector
from src.db.executor import measure_query, QueryExecutionError
from src.benchmark.query_parser import load_benchmark_queries
from src.benchmark.splitter import get_splits
import src.optimizers as optimizers_module  # Import the package
from src.optimizers.base import BaseOptimizer


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the experiment YAML config file."""
    print(f"Loading experiment config from {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def initialize_optimizers(config: List[Dict[str, Any]]) -> List[BaseOptimizer]:
    print("Initializing optimizers...")
    optimizer_instances: List[BaseOptimizer] = []

    for opt_config in config:
        optimizer_name = opt_config["name"]
        optimizer_params = opt_config.get("config", {})

        try:
            # Get the class from the optimizers module
            OptimizerClass: Type[BaseOptimizer] = getattr(
                optimizers_module, optimizer_name
            )

            try:
                # 1. Try to pass config as a single 'config' argument
                #    This is for LQOs like XGBoostLQO(config={...})
                instance = OptimizerClass(config=optimizer_params)
            except TypeError as e:
                # 2. If that fails (e.g., "got an unexpected keyword argument 'config'"),
                #    fall back to unpacking the params.
                #    This is for baselines like PostgresOptimizer() or
                #    PostgresGEQOOptimizer(geqo_threshold=2)

                # Check if the error is the one we expect
                if "unexpected keyword argument 'config'" in str(e):
                    instance = OptimizerClass(**optimizer_params)
                else:
                    # It was a different TypeError, re-raise it
                    raise e
            optimizer_instances.append(instance)
            print(f"  Initialized: {instance.optimizer_name}")

        except AttributeError:
            print(
                f"Error: Optimizer class '{optimizer_name}' "
                "not found in src/optimizers."
            )
            sys.exit(1)

    return optimizer_instances


def run_training_phase(
    optimizer_list: List[BaseOptimizer],
    training_data_path: str,
    lqo_config: Dict[str, Any],
) -> Dict[str, float]:
    print("\n--- Starting Training Phase ---")
    training_times: Dict[str, float] = {}

    if not os.path.exists(training_data_path):
        print(
            f"Warning: Training data '{training_data_path}' "
            "not found. Skipping LQO training."
        )
        return training_times

    for optimizer in optimizer_list:
        # Check if the optimizer is an LQO by
        # checking if it's NOT a baseline.
        # A better way would be an 'is_lqo' attribute.
        if "Postgres" in optimizer.optimizer_name:
            training_times[optimizer.optimizer_name] = 0.0
            continue

        print(f"Training {optimizer.optimizer_name}...")
        start_time = time.perf_counter()

        # Find the config for this specific LQO
        lqo_specific_config = {}
        for cfg in lqo_config:
            if cfg["name"] == optimizer.__class__.__name__:
                lqo_specific_config = cfg.get("config", {})
                break

        try:
            optimizer.train(
                training_data_path=training_data_path, config=lqo_specific_config
            )
        except FileNotFoundError as e:
            print(f"Error during training for {optimizer.optimizer_name}: {e}")
            print("Ensure your training data path is correct.")
            sys.exit(1)

        end_time = time.perf_counter()
        train_time_s = end_time - start_time
        training_times[optimizer.optimizer_name] = train_time_s
        print(
            f"Training for {optimizer.optimizer_name} "
            f"complete in {train_time_s:.2f}s"
        )

    print("--- Training Phase Complete ---")
    return training_times


def run_evaluation_phase(
    optimizer_list: List[BaseOptimizer],
    test_queries: List,
    db_connector: DatabaseConnector,
) -> List[Dict[str, Any]]:
    """
    Runs the evaluation for all optimizers on all test queries.

    This is the core benchmarking loop.
    """
    print(f"\n--- Starting Evaluation Phase on {len(test_queries)} queries ---")
    results: List[Dict[str, Any]] = []

    pbar = tqdm(total=len(test_queries) * len(optimizer_list), desc="Evaluating Plans")

    for query in test_queries:
        for optimizer in optimizer_list:
            try:
                # 1. Get Plan (Inference)
                plan_dict = optimizer.get_plan(query.query_sql)

                # 2. Measure Plan (Execution)
                measurement = measure_query(
                    db_connector=db_connector,
                    query_sql=plan_dict["plan_sql"],
                    settings=plan_dict["settings"],
                )

                # 3. Combine Metrics
                if measurement:
                    end_to_end_time_ms = (
                        plan_dict["inference_time_ms"]
                        + measurement["actual_latency_ms"]
                    )

                    row = {
                        "query_name": query.query_name,
                        "query_family": query.query_family,
                        "optimizer_name": optimizer.optimizer_name,
                        "inference_time_ms": plan_dict["inference_time_ms"],
                        "actual_latency_ms": measurement["actual_latency_ms"],
                        "end_to_end_time_ms": end_to_end_time_ms,
                        "estimated_cost": measurement["estimated_cost"],
                        "plan_json_hash": hash(str(measurement["plan_json"])),
                    }
                    results.append(row)

            except QueryExecutionError as e:
                tqdm.write(
                    f"[FAIL] Query: {query.query_name}, "
                    f"Opt: {optimizer.optimizer_name}, "
                    f"Error: {e.message}"
                )
            except Exception as e:
                tqdm.write(
                    f"[FATAL] Query: {query.query_name}, "
                    f"Opt: {optimizer.optimizer_name}, "
                    f"Unexpected Error: {e}"
                )
            finally:
                pbar.update(1)

    pbar.close()
    print("--- Evaluation Phase Complete ---")
    return results


@click.command()
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
def main(config_file: str):
    """
    Runs a full benchmark experiment from a YAML config file.
    """
    config = load_config(config_file)
    exp_name = config.get("experiment_name", "default_experiment")
    print(f"*** Starting Experiment: {exp_name} ***")

    db_conn = None
    try:
        # 1. Setup DB
        db_conn = DatabaseConnector()

        # 2. Load & Split Queries
        all_queries = load_benchmark_queries(config["benchmark"])
        split_gen = get_splits(
            queries=all_queries,
            strategy=config["split_strategy"],
            test_size=config.get("split_test_size", 0.2),
            random_seed=config.get("split_random_seed", 42),
        )

        # 3. Initialize Optimizers
        optimizers = initialize_optimizers(config["optimizers"])

        # We will collect results across all splits
        # (e.g., for Leave-One-Out)
        all_results_df = pd.DataFrame()

        # 4. Run experiment for each split
        for i, (train_queries, test_queries) in enumerate(split_gen):
            split_name = f"{config['split_strategy']}_split_{i+1}"
            print(f"\n*** Running {split_name} ***")
            print(
                f"Train size: {len(train_queries)}, " f"Test size: {len(test_queries)}"
            )

            # 5. Training Phase
            # Note: We train *once* before all splits.
            # For LOO, this is technically "cheating",
            # as the test query was in the training set.
            # This matches the paper's "Easiest / Max Leakage"
            # definition.
            #
            # For Base/Random, this loop only runs once.
            if i == 0:  # Only train on the first split
                training_times = run_training_phase(
                    optimizers,
                    config["lqo_training_data"],
                    config["optimizers"],  # Pass optimizer config
                )

            # 6. Evaluation Phase
            results_list = run_evaluation_phase(optimizers, test_queries, db_conn)

            # 7. Process Results
            if not results_list:
                print(f"No results generated for {split_name}.")
                continue

            split_df = pd.DataFrame(results_list)
            split_df["split_name"] = split_name

            # Add training time to the df
            split_df["training_time_s"] = split_df["optimizer_name"].map(training_times)

            all_results_df = pd.concat([all_results_df, split_df], ignore_index=True)

        # 8. Save Final Results
        if not all_results_df.empty:
            output_file = config["output_file"]
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            all_results_df.to_csv(output_file, index=False)
            print(f"\n*** Experiment Complete ***")
            print(f"Results for {exp_name} saved to {output_file}")
        else:
            print(f"\n*** Experiment Complete: No results generated. ***")

    except Exception as e:
        print(f"\nAn unrecoverable error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if db_conn:
            db_conn.close_all()


if __name__ == "__main__":
    main()

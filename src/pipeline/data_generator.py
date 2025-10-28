import concurrent.futures
import glob
import hashlib
import itertools
import os
from typing import List, Dict, Any, Tuple, Generator, Optional

import pandas as pd
from tqdm import tqdm

from src.db.connector import DatabaseConnector
from src.db.executor import measure_query, QueryExecutionError


def _worker_measure_query(
    job: Tuple[str, str, Tuple[str, ...]],
) -> Optional[Dict[str, Any]]:
    query_name, query_sql, settings = job
    db = None
    try:
        db = DatabaseConnector()

        measurement = measure_query(
            db_connector=db, query_sql=query_sql, settings=list(settings)
        )

        if measurement:
            query_hash = hashlib.md5(query_sql.encode()).hexdigest()
            return {
                "query_name": query_name,
                "query_hash": query_hash,
                "settings": " ".join(settings),
                **measurement,
            }
    except QueryExecutionError:
        return None
    except Exception as e:
        print(f"[Worker Error] for {query_name}: {e}")
        return None
    finally:
        if db:
            db.close_all()
    return None


def load_queries(benchmark_name: str = "job") -> Dict[str, str]:
    print(f"Loading queries for benchmark: {benchmark_name}")
    query_dir = os.path.join("benchmarks", benchmark_name, "queries")

    sql_files = glob.glob(os.path.join(query_dir, "*.sql"))
    if not sql_files:
        raise FileNotFoundError(f"No .sql files found in {query_dir}.")

    queries = {}
    for sql_file in sql_files:
        query_name = os.path.basename(sql_file)
        with open(sql_file, "r") as f:
            queries[query_name] = f.read()

    print(f"Loaded {len(queries)} queries.")
    return queries


def get_plan_forcing_configs() -> Generator[Tuple[str, ...], None, None]:
    # Define the 'on'/'off' options for each plan operator
    # We can expand this list significantly
    setting_options = {
        "enable_hashjoin": ["on", "off"],
        "enable_mergejoin": ["on", "off"],
        "enable_nestloop": ["on", "off"],
        "enable_bitmapscan": ["on", "off"],
        "enable_indexscan": ["on", "off"],
        "enable_seqscan": ["on", "off"],
    }

    # Get the keys and value lists
    # e.g., keys = ['enable_hashjoin', 'enable_mergejoin', ...]
    # e.g., value_sets = [['on', 'off'], ['on', 'off'], ...]
    keys = list(setting_options.keys())
    value_sets = list(setting_options.values())

    # Create the Cartesian product
    # e.g., ('on', 'on', 'on', 'on', 'on', 'on')
    # e.g., ('on', 'on', 'on', 'on', 'on', 'off')
    # ...
    for value_tuple in itertools.product(*value_sets):
        # Format as SQL commands
        # e.g., ("SET enable_hashjoin = 'on';", "SET enable_mergejoin = 'on';", ...)
        yield tuple(f"SET {key} = '{value}';" for key, value in zip(keys, value_tuple))


def run_data_generation(benchmark_name: str, output_file: str) -> None:
    all_results: List[Dict[str, Any]] = []

    try:
        queries = load_queries(benchmark_name)
        plan_configs = list(get_plan_forcing_configs())

        # Determine number of parallel workers
        max_workers = os.cpu_count() or 4
        print(f"Starting data generation using up to {max_workers} processes.")

        total_queries = len(queries)
        print(
            f"Processing {total_queries} queries with {len(plan_configs)} configs each."
        )
        print(f"Total potential plans: {total_queries * len(plan_configs)}")
        print(f"Results will be saved to: {output_file}")

        # Outer loop is SEQUENTIAL (one query at a time)
        for i, (query_name, query_sql) in enumerate(queries.items()):

            print(f"\n--- Processing Query {i+1}/{total_queries}: {query_name} ---")

            # 1. Create the list of jobs for *this query only*
            jobs = [(query_name, query_sql, settings) for settings in plan_configs]

            # 2. Create a ProcessPoolExecutor to run jobs in parallel
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:

                # 3. Use executor.map to run jobs and show progress with tqdm
                # tqdm will wrap the executor.map iterator
                results_iterator = executor.map(_worker_measure_query, jobs)

                # We use list() to force execution and wait for all jobs to complete
                query_results = list(
                    tqdm(
                        results_iterator,
                        total=len(jobs),
                        desc=f"  Plans for {query_name}",
                    )
                )

                # 4. Filter out failed jobs (which returned None)
                successful_results = [r for r in query_results if r is not None]
                all_results.extend(successful_results)

                print(
                    f"  Completed {query_name}: "
                    f"{len(successful_results)} successful plans "
                    f"({len(jobs) - len(successful_results)} failed/invalid)."
                )

        print("\nData generation complete.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        # No main DB connection to close, as it's all in the workers

        if all_results:
            print(f"Saving {len(all_results)} total successful plan measurements...")
            df = pd.DataFrame(all_results)

            # Ensure results directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Save as Parquet for efficiency
            df.to_parquet(output_file, index=False)
            print(f"Successfully saved data to {output_file}")
        else:
            print("No results were generated.")


if __name__ == "__main__":
    if not os.path.exists("src") or not os.path.exists("benchmarks"):
        print("Error: This script must be run from the root project directory.")
        exit(1)

    run_data_generation(
        benchmark_name="job", output_file="results/job_training_data_cache.parquet"
    )

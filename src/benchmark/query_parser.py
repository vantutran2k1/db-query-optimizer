import glob
import os
import re
from typing import List, NamedTuple


class BenchmarkQuery(NamedTuple):
    query_name: str  # e.g., "1a.sql"
    query_family: str  # e.g., "1"
    query_sql: str  # "SELECT ..."


def load_benchmark_queries(benchmark_name: str = "job") -> List[BenchmarkQuery]:
    print(f"Loading queries for benchmark: {benchmark_name}")
    query_dir = os.path.join("benchmarks", benchmark_name, "queries")

    sql_files = glob.glob(os.path.join(query_dir, "*.sql"))
    if not sql_files:
        raise FileNotFoundError(
            f"No .sql files found in {query_dir}. "
            "Please download the JOB benchmark queries."
        )

    queries: List[BenchmarkQuery] = []

    # Regex to capture the 'family' (e.g., '1', '10')
    # from '1a.sql' or '10b.sql'
    family_regex = re.compile(r"^(\d+)[a-zA-Z]*\.sql$")

    for sql_file in sql_files:
        query_name = os.path.basename(sql_file)

        match = family_regex.match(query_name)
        if not match:
            print(f"Skipping non-query file: {query_name}")
            continue

        query_family = match.group(1)

        with open(sql_file, "r") as f:
            query_sql = f.read()

        queries.append(
            BenchmarkQuery(
                query_name=query_name, query_family=query_family, query_sql=query_sql
            )
        )

    print(f"Loaded {len(queries)} queries.")
    return queries

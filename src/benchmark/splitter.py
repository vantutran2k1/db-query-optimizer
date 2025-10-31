from typing import List, Tuple, Generator, Dict

import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneOut

from src.benchmark.query_parser import BenchmarkQuery

Split = Tuple[List[BenchmarkQuery], List[BenchmarkQuery]]
SplitGenerator = Generator[Split, None, None]


def get_splits(
    queries: List[BenchmarkQuery],
    strategy: str,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> SplitGenerator:
    if strategy == "random":
        yield _get_random_split(queries, test_size, random_seed)
    elif strategy == "leave_one_out":
        yield from _get_leave_one_out_split(queries)
    elif strategy == "base_query":
        yield _get_base_query_split(queries, test_size, random_seed)
    else:
        raise ValueError(
            f"Unknown split strategy: {strategy}. "
            "Must be 'random', 'leave_one_out', or 'base_query'."
        )


def _get_random_split(
    queries: List[BenchmarkQuery], test_size: float, random_seed: int
) -> Split:
    """
    Strategy 1: Random Sampling
    Splits queries randomly, ignoring families.
    """
    print(f"Splitting strategy: Random (test_size={test_size})")
    train, test = train_test_split(
        queries, test_size=test_size, random_state=random_seed
    )
    return train, test


def _get_leave_one_out_split(queries: List[BenchmarkQuery]) -> SplitGenerator:
    """
    Strategy 2: Leave One Out Sampling
    Generates N splits, each holding out one query.
    """
    print("Splitting strategy: Leave-One-Out")
    loo = LeaveOneOut()
    query_indices = np.arange(len(queries))

    for train_idx, test_idx in loo.split(query_indices):
        train = [queries[i] for i in train_idx]
        test = [queries[i] for i in test_idx]
        yield train, test


def _get_base_query_split(
    queries: List[BenchmarkQuery], test_size: float, random_seed: int
) -> Split:
    """
    Strategy 3: Base Query Sampling
    Splits by query *family* to prevent leakage of
    join structures.
    """
    print(f"Splitting strategy: Base Query (test_size={test_size})")

    # 1. Group queries by family
    families: Dict[str, List[BenchmarkQuery]] = {}
    for q in queries:
        if q.query_family not in families:
            families[q.query_family] = []
        families[q.query_family].append(q)

    family_names = sorted(list(families.keys()))

    # 2. Split the list of *family names*
    train_families, test_families = train_test_split(
        family_names, test_size=test_size, random_state=random_seed
    )

    # 3. Reconstruct train/test lists from the families
    train_queries: List[BenchmarkQuery] = []
    for family in train_families:
        train_queries.extend(families[family])

    test_queries: List[BenchmarkQuery] = []
    for family in test_families:
        test_queries.extend(families[family])

    print(
        f"  Split: {len(train_families)} train families "
        f"({len(train_queries)} queries) vs. "
        f"{len(test_families)} test families "
        f"({len(test_queries)} queries)"
    )

    return train_queries, test_queries

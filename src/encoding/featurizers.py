import re
from typing import List, Dict, Any, Set, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

from src.encoding.base import BaseFeaturizer


class SimpleFeaturizer(BaseFeaturizer):
    NUMERICAL_FEATURES = ["Total Cost", "Plan Rows", "Plan Width", "Startup Cost"]
    AGGREGATES = ["sum", "avg", "max", "min"]

    def __init__(self):
        super().__init__()

        self.node_type_encoder: Optional[OneHotEncoder] = None
        self.node_type_vocab: List[str] = []

        self.table_vectorizer: Optional[CountVectorizer] = None
        self.operator_vectorizer: Optional[CountVectorizer] = None

        self.feature_names: List[str] = []

    def fit(self, plan_jsons: List[Dict[str, Any]], query_sqls: List[str]) -> None:
        """
        Learns vocabularies from both plans (Node Types)
        and queries (Tables, Operators).
        """
        print("Fitting SimpleFeaturizer...")

        # 1. --- Fit Plan Featurizer ---
        all_node_types: Set[str] = set()
        for plan in plan_jsons:
            nodes = self._walk_tree(plan)
            for node in nodes:
                all_node_types.add(node["Node Type"])

        self.node_type_vocab = sorted(list(all_node_types))
        self.node_type_encoder = OneHotEncoder(
            categories=[self.node_type_vocab],
            handle_unknown="ignore",
            sparse_output=False,
        )
        self.node_type_encoder.fit(np.array(self.node_type_vocab).reshape(-1, 1))

        # 2. --- Fit Query Featurizers ---
        print("Fitting query vocabularies (tables, operators)...")
        # Table Vectorizer
        self.table_vectorizer = CountVectorizer(token_pattern=r"[a-zA-Z0-9_]+")
        table_corpus = [self._extract_tables(q) for q in query_sqls]
        self.table_vectorizer.fit(table_corpus)

        # Operator Vectorizer
        self.operator_vectorizer = CountVectorizer()
        operator_corpus = [self._extract_operators(q) for q in query_sqls]
        self.operator_vectorizer.fit(operator_corpus)

        # 3. --- Generate Feature Names ---
        self.feature_names = [f"NodeType_{t}" for t in self.node_type_vocab]
        for num_feat in self.NUMERICAL_FEATURES:
            for agg in self.AGGREGATES:
                self.feature_names.append(f"{agg}_{num_feat.replace(' ', '_')}")

        # Add new query feature names
        self.feature_names.extend(["query_len", "num_joins", "num_predicates"])
        self.feature_names.extend(
            [f"Tbl_{t}" for t in self.table_vectorizer.get_feature_names_out()]
        )
        self.feature_names.extend(
            [f"Op_{o}" for o in self.operator_vectorizer.get_feature_names_out()]
        )

        self.is_fitted = True
        print(f"Fit complete. Vocabulary size: {len(self.node_type_vocab)} nodes.")
        print(f"Total output vector size: {len(self.feature_names)}")

    def transform(self, plan_json: Dict[str, Any], query_sql: str) -> np.ndarray:
        """
        Transforms a single plan/query pair into a 1D numpy vector.
        """
        if not self.is_fitted:
            raise RuntimeError("Featurizer is not fitted. Call .fit() first.")

        nodes = self._walk_tree(plan_json)
        node_counts: Dict[str, int] = {nt: 0 for nt in self.node_type_vocab}
        for node in nodes:
            node_type = node["Node Type"]
            if node_type in self.node_type_vocab:
                node_counts[node_type] += 1

        plan_ohe_features = np.array(
            [node_counts[node_type] for node_type in self.node_type_vocab]
        )

        plan_numerical_features = []
        for feat_name in self.NUMERICAL_FEATURES:
            values = [n[feat_name] for n in nodes if feat_name in n]
            if not values:
                values = [0.0]
            plan_numerical_features.append(np.sum(values))
            plan_numerical_features.append(np.mean(values))
            plan_numerical_features.append(np.max(values))
            plan_numerical_features.append(np.min(values))

        plan_numerical_vector = np.array(plan_numerical_features)

        query_len = len(query_sql)
        num_joins = query_sql.upper().count("JOIN ")
        num_predicates = query_sql.upper().count("WHERE ") + query_sql.upper().count(
            "AND "
        )

        query_scalar_vector = np.array([query_len, num_joins, num_predicates])

        # Token count features
        table_counts = (
            self.table_vectorizer.transform([self._extract_tables(query_sql)])
            .toarray()
            .flatten()
        )

        operator_counts = (
            self.operator_vectorizer.transform([self._extract_operators(query_sql)])
            .toarray()
            .flatten()
        )

        # 3. --- Concatenate all features ---
        final_vector = np.concatenate(
            (
                plan_ohe_features,
                plan_numerical_vector,
                query_scalar_vector,
                table_counts,
                operator_counts,
            )
        ).astype(np.float32)

        return final_vector

    @staticmethod
    def _extract_tables(query_sql: str) -> str:
        tables = re.findall(
            r"\bFROM\s+([a-zA-Z0-9_]+)|JOIN\s+([a-zA-Z0-9_]+)", query_sql, re.IGNORECASE
        )
        return " ".join([t[0] or t[1] for t in tables])

    @staticmethod
    def _extract_operators(query_sql: str) -> str:
        operators = re.findall(
            r"\b([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)\s*(=|>|<|>=|<=|LIKE|IN)\b",
            query_sql,
            re.IGNORECASE,
        )
        return " ".join([op[1] for op in operators])

    @staticmethod
    def _walk_tree(node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Performs a DFS traversal to collect all nodes in the tree."""
        nodes = []

        stack = [node]
        while stack:
            current_node = stack.pop()
            if not current_node:
                continue

            nodes.append(current_node)

            if "Plans" in current_node and current_node["Plans"] is not None:
                for child in current_node["Plans"]:
                    stack.append(child)
        return nodes


class GNNFeaturizer(BaseFeaturizer):
    def __init__(self):
        super().__init__()
        # TODO: Add vocabs for node types, tables, columns,
        # predicates, etc.
        # self.node_type_vocab = {}
        # self.table_vocab = {}
        # self.predicate_vocab = {}
        pass

    def fit(self, plan_jsons: List[Dict[str, Any]]) -> None:
        print("[GNNFeaturizer] Fitting...")
        # TODO:
        # 1. Walk all trees
        # 2. Build vocabularies for:
        #    - 'Node Type' (e.g., 'Hash Join')
        #    - 'Relation Name' (e.g., 'title')
        #    - 'Filter' / 'Join Condition' (e.g., 't.id = mi.movie_id')
        # 3. We would use these vocabs to build embedding layers.
        #    This solves the invariance problem for filters/tables.
        print(
            "TODO: Fit GNN featurizer by building "
            "vocabularies for nodes, tables, and predicates."
        )
        self.is_fitted = True  # Mark as fitted for stub

    def transform(self, plan_json: Dict[str, Any]) -> Any:
        """
        Transforms a plan JSON into a PyTorch Geometric
        Data object (graph).
        """
        if not self.is_fitted:
            raise RuntimeError("Featurizer is not fitted.")

        # This is where the magic happens.
        # We would need:
        # 1. node_features (x): A [num_nodes, feature_dim] tensor.
        #    Each node's features would be an embedding of its
        #    type + its numerical features (cost, rows).
        #
        # 2. edge_index: A [2, num_edges] tensor defining the
        #    parent-child relationships in the tree.

        print(
            "TODO: Implement GNNFeaturizer.transform. "
            "This will return a torch_geometric.data.Data "
            "object, not a np.array."
        )

        # Placeholder:
        # try:
        #     import torch
        #     from torch_geometric.data import Data
        # except ImportError:
        #     raise ImportError(
        #         "Please install 'torch' and 'torch_geometric' "
        #         "to use GNNFeaturizer."
        #     )
        #
        # ... logic to build node_features and edge_index ...
        #
        # return Data(x=node_features, edge_index=edge_index)

        # For now, raise error as it's not implemented.
        raise NotImplementedError(
            "GNNFeaturizer.transform is not yet implemented. "
            "This is the advanced path."
        )

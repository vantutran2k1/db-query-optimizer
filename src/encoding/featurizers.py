from typing import List, Dict, Any, Set, Optional

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from src.encoding.base import BaseFeaturizer


class SimpleFeaturizer(BaseFeaturizer):
    NUMERICAL_FEATURES = ["Total Cost", "Plan Rows", "Plan Width", "Startup Cost"]
    AGGREGATES = ["sum", "avg", "max", "min"]

    def __init__(self):
        super().__init__()
        self.node_type_encoder: Optional[OneHotEncoder] = None
        self.vocabulary: List[str] = []
        self.feature_names: List[str] = []

    def fit(self, plan_jsons: List[Dict[str, Any]]) -> None:
        """
        Learns the vocabulary of all possible 'Node Type' values
        from the training plans.
        """
        print("Fitting SimpleFeaturizer...")
        all_node_types: Set[str] = set()

        for plan in plan_jsons:
            nodes = self._walk_tree(plan)
            for node in nodes:
                all_node_types.add(node["Node Type"])

        self.vocabulary = sorted(list(all_node_types))

        # Initialize the OneHotEncoder
        self.node_type_encoder = OneHotEncoder(
            categories=[self.vocabulary],
            handle_unknown="ignore",  # Ignore node types not seen in training
            sparse_output=False,
        )
        # Fit it so it's ready for transform
        # We fit it on an empty array of the correct shape
        self.node_type_encoder.fit(np.array(self.vocabulary).reshape(-1, 1))

        # Generate feature names for interpretability
        self.feature_names = [f"NodeType_{t}" for t in self.vocabulary]
        for num_feat in self.NUMERICAL_FEATURES:
            for agg in self.AGGREGATES:
                self.feature_names.append(f"{agg}_{num_feat.replace(' ', '_')}")

        self.is_fitted = True
        print(f"Fit complete. Vocabulary size: {len(self.vocabulary)} nodes.")
        print(f"Output vector size: {len(self.feature_names)}")

    def transform(self, plan_json: Dict[str, Any]) -> np.ndarray:
        """
        Transforms a single plan JSON into a 1D numpy vector.
        """
        if not self.is_fitted:
            raise RuntimeError("Featurizer is not fitted. Call .fit() first.")

        nodes = self._walk_tree(plan_json)

        # 1. --- Node Type Counts (Categorical) ---
        node_counts: Dict[str, int] = {node_type: 0 for node_type in self.vocabulary}
        for node in nodes:
            node_type = node["Node Type"]
            if node_type in self.vocabulary:
                node_counts[node_type] += 1

        # Create a 2D array for the encoder, then flatten
        # This is the "count vector"
        node_count_vector = np.array(
            [node_counts[node_type] for node_type in self.vocabulary]
        ).reshape(1, -1)

        # The OneHotEncoder here is a bit of a misnomer;
        # we're just using it to ensure a consistent vector structure.
        # A better way is just to use the count vector directly.
        # Let's simplify:
        ohe_features = np.array(
            [node_counts[node_type] for node_type in self.vocabulary]
        )

        # 2. --- Numerical Feature Aggregation ---
        numerical_features = []
        for feat_name in self.NUMERICAL_FEATURES:
            # Collect all values for this feature from all nodes
            values = [node[feat_name] for node in nodes if feat_name in node]
            if not values:
                # Handle case where no node has this feature
                values = [0.0]

            # Calculate aggregates
            numerical_features.append(np.sum(values))
            numerical_features.append(np.mean(values))
            numerical_features.append(np.max(values))
            numerical_features.append(np.min(values))

        numerical_vector = np.array(numerical_features)

        # 3. --- Concatenate all features ---
        final_vector = np.concatenate((ohe_features, numerical_vector)).astype(
            np.float32
        )

        return final_vector

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

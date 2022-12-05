from typing import List, Optional

import networkx as nx
import numpy as np

from helpers.graph import create_cooccurence_graph
from helpers.utils import get_ambiguous_ids, show_graph


class SimilarityMatrix():
    def __init__(
        self,
        matrix: np.ndarray,
        query_ids: List[int],
        target_ids: List[int],
    ):
        self._matrix = matrix
        self.query_ids = query_ids
        self.target_ids = target_ids


    @classmethod
    def from_data(
        cls,
        graph: nx.Graph,
        isnad_mention_ids: List[List[int]],
        disambiguated_ids: List[int],
        **hash_kwargs,
    ):
        disambiguated_ids = [id for id in disambiguated_ids if id in graph.nodes]
        ambiguous_ids = get_ambiguous_ids(isnad_mention_ids, disambiguated_ids)

        node_hashes = {
            int(node): hash_node(graph, node, **hash_kwargs)
            for node in graph.nodes
        }

        node_neighbors = {
            int(node): list(graph.successors(node)) + list(graph.predecessors(node))
            for node in graph.nodes
        }

        matrix = np.array([
            [
                0 if (
                    disambiguated_id in node_neighbors[ambiguous_id] or
                    ambiguous_id == disambiguated_id
                ) else (
                    cosine_similarity(
                        node_hashes[ambiguous_id],
                        node_hashes[disambiguated_id]
                    )
                )
                for disambiguated_id in disambiguated_ids
            ]
            for ambiguous_id in ambiguous_ids
        ])

        return cls(matrix, ambiguous_ids, disambiguated_ids)


    def __getitem__(self, ids):
        if isinstance(ids, tuple):
            query_id, target_id = ids

            # if both are targets (in the case of neighbors)
            if query_id in self.target_ids:
                return 1 if query_id == target_id else 0

            query_index = self.query_ids.index(query_id)
            target_index = self.target_ids.index(target_id)

            return self._matrix[query_index, target_index]

        else:
            query_id = ids
            query_index = self.query_ids.index(query_id)
            return self._matrix[query_index]


    def argsort(self):
        # TODO: Come back and fix this crap
        sorted_indexes = list(zip(*np.unravel_index(
            np.argsort(self._matrix, axis=None),
            self._matrix.shape
        )))
        return [
            (self.query_ids[query_index], self.target_ids[target_index])
            for query_index, target_index in sorted_indexes
        ]


    def argmax(self):
        # TODO: Come back and fix this crap
        query_index, target_index = np.unravel_index(
            np.argmax(self._matrix, axis=None),
            self._matrix.shape
        )
        return self.query_ids[query_index], self.target_ids[target_index]


    def __repr__(self):
        representation = np.zeros((self._matrix.shape[0] + 1, self._matrix.shape[1] + 1))

        representation[0, 1:] = self.target_ids
        representation[1:, 0] = self.query_ids
        representation[1:, 1:] = self._matrix

        representation_rounded = (representation * 10).astype(np.int)
        return str(representation_rounded)


def hash_node(
    graph: nx.Graph,
    node: "nx.Node",
    cooc_alpha: Optional[float] = 1.0,
    position_alpha: Optional[float] = 1.0,
    nlp_alpha: Optional[float] = 0.1
):
    num_nodes = graph.number_of_nodes()
    cooc_hash = np.zeros(num_nodes)
    position_hash = np.zeros(num_nodes)
    nlp_hash = np.array(graph.nodes[node]["embedding"])

    # used for indexing
    node_ids = list(graph.nodes())
    node_index = node_ids.index(node)

    #for all inbound edges
    for pred in graph.predecessors(node):
        pred_index = node_ids.index(pred)

        cooc_hash[pred_index] = graph[pred][node]["num_coocurrences"]
        position_hash[pred_index] = (
            -1 *
            graph[pred][node]["relative_position_sum"] /
            graph[pred][node]["num_coocurrences"]
        )

    #for outbound edges
    for succ in graph.successors(node):
        succ_index = node_ids.index(succ)

        cooc_hash[succ_index] = graph[node][succ]["num_coocurrences"]
        position_hash[succ_index] = (
            graph[node][succ]["relative_position_sum"] /
            graph[node][succ]["num_coocurrences"]
        )

    # concat relevant features
    hash = []
    if cooc_alpha is not None:
        hash = np.concatenate((hash, cooc_hash * cooc_alpha))
    if position_alpha is not None:
        hash = np.concatenate((hash, position_hash * position_alpha))
    if nlp_alpha is not None:
        hash = np.concatenate((hash, nlp_hash * nlp_alpha))

    return hash


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    graph, node_color = create_cooccurence_graph(
        "nameData/names_disambiguated.csv",
        "communities/goldStandard_goldTags.json",
        "nameData/namesWithEmbeddings_NER_strict.json",
        self_edges=True,
        max_isnads=1,
    )

    # used for splitting social hash into cooc and pos
    num_nodes = graph.number_of_nodes()

    print("cooc hashes:")
    for node_id in graph.nodes:
        social_hash = get_social_hash(graph, node_id)
        cooc_hash = social_hash[:num_nodes]
        print(f"{node_id}: {cooc_hash}")

    print("position hashes:")
    for node_id in graph.nodes:
        social_hash = get_social_hash(graph, node_id)
        position_hash = social_hash[num_nodes:]
        print(f"{node_id}: {position_hash}")

    show_graph(graph)

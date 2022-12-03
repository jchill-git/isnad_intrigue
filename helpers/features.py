from typing import List

import networkx as nx
import numpy as np

from graph import create_cooccurence_graph
from utils import get_ambiguous_ids, show_graph


class SimilarityMatrix():
    def __init__(
        self,
        matrix: np.ndarray,
        ambiguous_ids: List[int],
        disambiguated_ids: List[int]
    ):
        self.query_ids = ambiguous_ids
        self.target_ids = disambiguated_ids
        self._matrix = matrix


    def __getitem__(self, query_id, target_id=None):
        query_index = self.query_ids.index(query_id)

        if target_id is None:
            return self._matrix[query_index]

        else:
            target_index = self.target_ids.index(query_id)
            return self._matrix[query_index, target_index]


    def argsort(self):
        print(self._matrix.shape)
        sorted_indexes = np.argsort(self._matrix, axis=0)
        print(sorted_indexes)
        return [
            (self.query_ids[query_index], self.target_ids[target_index])
            for query_index, target_index in zip(*sorted_indexes)
        ]


    def argmax(self):
        query_index, target_index = np.argmax(self._matrix)
        return self.query_ids[query_index], self.target_ids[target_index]


def hash_node(graph, node, cooc_alpha=1.0, position_alpha=1.0, nlp_alpha=0.1):
    num_nodes = graph.number_of_nodes()
    cooc_hash = np.zeros(num_nodes)
    position_hash = np.zeros(num_nodes)
    nlp_hash = np.array([]) # TODO: make sure this is of type np

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
    if cooc_alpha != 0.0:
        hash = np.concatenate((hash, cooc_hash * cooc_alpha))
    if position_alpha != 0.0:
        hash = np.concatenate((hash, position_hash * position_alpha))
    if nlp_alpha != 0.0:
        hash = np.concatenate((hash, nlp_hash * nlp_alpha))

    return hash


def get_similarity_matrix(graph, isnad_mention_ids, disambiguated_ids):
    ambiguous_ids = get_ambiguous_ids(isnad_mention_ids, disambiguated_ids)

    matrix = np.array([
        [
            cosine_similarity(
                hash_node(graph, ambiguous_id),
                hash_node(graph, disambiguated_id)
            )
            for disambiguated_id in disambiguated_ids
        ]
        for ambiguous_id in ambiguous_ids
    ])

    return SimilarityMatrix(matrix, ambiguous_ids, disambiguated_ids)


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

from typing import List, Optional

import networkx as nx
import numpy as np

from helpers.graph import create_cooccurence_graph
from helpers.utils import get_ambiguous_ids, show_graph


class SimilarityMatrix():
    def __init__(
        self,
        matrix: np.ndarray,
        x_ids: List[int],
        y_ids: List[int],
    ):
        self._matrix = matrix
        self.x_index = {
            id: index
            for index, id in enumerate(x_ids)
        }
        self.x_id = x_ids
        self.y_index = {
            id: index
            for index, id in enumerate(y_ids)
        }
        self.y_id = y_ids


    @classmethod
    def from_data(
        cls,
        graph: nx.Graph,
        **hash_kwargs,
    ):
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
                0 if node_id_j in node_neighbors[node_id_i]
                else cosine_similarity(
                    node_hashes[node_id_i],
                    node_hashes[node_id_j]
                )
                for node_id_j in graph.nodes
            ]
            for node_id_i in graph.nodes
        ])

        node_ids = list(graph.nodes)

        return cls(matrix, node_ids, node_ids)


    def take_2d(self, x_ids, y_ids):
        x_indices = [self.x_index[id] for id in x_ids]
        y_indices = [self.y_index[id] for id in y_ids]
        matrix = self._matrix[x_indices][:, y_indices]

        return self.__class__(matrix, x_ids, y_ids)


    def __getitem__(self, ids):
        if isinstance(ids, tuple):
            x_id, y_id = ids

            x_index = self.x_index[x_id]
            y_index = self.y_index[y_id]

            return self._matrix[x_index, y_index]

        else:
            x_id = ids
            x_index = self.x_index[x_id]
            return self._matrix[x_index]


    def argsort(self):
        # TODO: Come back and fix this crap
        sorted_indexes = list(zip(*np.unravel_index(
            np.argsort(self._matrix, axis=None),
            self._matrix.shape
        )))
        return [
            (self.x_id[x_index], self.y_id[y_index])
            for x_index, y_index in sorted_indexes
        ]


    def argmax(self):
        # TODO: Come back and fix this crap
        x_index, y_index = np.unravel_index(
            np.argmax(self._matrix, axis=None),
            self._matrix.shape
        )
        return self.x_id[x_index], self.y_id[y_index]


    @property
    def shape(self):
        return self._matrix.shape

    def __repr__(self):
        representation = np.zeros((self._matrix.shape[0] + 1, self._matrix.shape[1] + 1))

        representation[0, 1:] = list(self.x_id.values())
        representation[1:, 0] = list(self.y_id.values())
        representation[1:, 1:] = self._matrix

        return "\n".join([
            " ".join([
                f"{value:.2f}"
                for value in row
            ])
            for row in representation
        ])
        

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

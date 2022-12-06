from typing import List, Optional, Tuple

import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, wait

from helpers.graph import create_cooccurence_graph
from helpers.utils import get_ambiguous_ids, show_graph, match_list_shape

class SimilarityScorer():
    def __init__(
        self,
        graph: nx.Graph,
        disambiguated_ids: List[int],
        similarities: Optional[np.ndarray] = None,
        **hash_kwargs
    ):
        self.graph = graph
        self.disambiguated_ids = disambiguated_ids
        self.hash_kwargs = hash_kwargs

        nodes = list(graph.nodes)

        self.id_to_index = {
            id: index
            for index, id in enumerate(nodes)
        }

        self.node_hashes = {
            id: hash_node(self.graph, id, **hash_kwargs)
            for id in nodes
        }

        if similarities is not None:
            self._similarities = similarities
        else:
            self._similarities = np.full((len(nodes), len(nodes)), np.nan, dtype=np.float16)


    def __getitem__(self, ids: Tuple[int, int]) -> float:
        # unpack tuple, convert to indices
        x_id, y_id = ids

        x_index = self.id_to_index[x_id]
        y_index = self.id_to_index[y_id]

        # check memo
        if not np.isnan(self._similarities[x_index][y_index]):
            return self._similarities[x_index][y_index]
        elif not np.isnan(self._similarities[y_index][x_index]):
            return self._similarities[y_index][x_index]

        # check id equivalence
        elif x_id == y_id:
            similarity = 1.0

        # check disambiguated inequivalence
        elif x_id in self.disambiguated_ids and y_id in self.disambiguated_ids:
            similarity = 0.0

        # check if direct neighbors
        elif y_id in (list(self.graph.successors(x_id)) + list(self.graph.predecessors(x_id))):
            similarity = 0.0

        # otherwise, do hash
        else:
            x_hash = self.node_hashes[x_id]
            y_hash = self.node_hashes[y_id]
            similarity = cosine_similarity(x_hash, y_hash)

        # memoize results
        self._similarities[x_index][y_index] = similarity
        self._similarities[y_index][x_index] = similarity

        return similarity


    def argsort_ids(self, x_ids, y_ids):
        print("argsort_ids")

        with ThreadPoolExecutor(max_workers=None) as executor:
            futures = [
                [(x_id, y_id), executor.submit(self.__getitem__, (x_id, y_id))]
                for x_id in x_ids
                for y_id in y_ids
            ]
            print("created futures")

        print("futures done")

        pair_similarities = [
            [pair, future.result()]
            for pair, future in futures
        ]
        print("got future results")

        sorted_ids_similarities = sorted(pair_similarities, key=lambda e: e[1], reverse=True)
        print(sorted_ids_similarities[:20])
        sorted_ids, _ = zip(*sorted_ids_similarities)

        return sorted_ids


    @property
    def shape(self):
        return self._similarities.shape


    def __repr__(self):
        num_memoized = np.count_nonzero(~np.isnan(self._similarities))
        total = self._similarities.size
        return f"Similiarities({num_memoized} / {total} memoized)"


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

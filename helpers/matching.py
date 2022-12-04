from typing import List, Union

import numpy as np
import networkx as nx

from data import read_isnad_data, split_data
from graph import create_cooccurence_graph
from features import get_similarity_matrix, SimilarityMatrix
from utils import get_ambiguous_ids


def merge_nodes(query_id: int, target_id: Union[int, None]):
    pass


def can_merge_neighborhoods(
    graph: nx.Graph,
    similarities: SimilarityMatrix,
    query_id: int,
    target_id: int,
    threshold: float,
):
    print(f"can_merge_neighborhoods q={query_id} t={target_id}")
    if similarities[query_id, target_id] < threshold:
        print("cannot merge, query doesn't match target")
        return False

    query_neighbor_ids = graph.neighbors(query_id)
    target_neighbor_ids = graph.neighbors(target_id)

    for query_neighbor_id in query_neighbor_ids:
        for target_neighbor_id in target_neighbor_ids:
            print(f"qn: {query_neighbor_id} tn: {target_neighbor_id}: {similarities[query_neighbor_id, target_neighbor_id]}")
            if similarities[query_neighbor_id, target_neighbor_id] > threshold:
                break
        else:
            print(f"couldn't find matching neighbor for {query_neighbor_id}")
            return False


def match_subgraphs(
    isnad_mention_ids: List[List[int]],
    disambiguated_ids: List[int],
    graph: nx.Graph,
    threshold: float
):
    while len(disambiguated_ids) > 0:
        # compute similarities
        similarity_matrix = get_similarity_matrix(graph, isnad_mention_ids, disambiguated_ids)
        print(similarity_matrix)

        # find query and target ids with the highest similarity
        for query_id, target_id in similarity_matrix.argsort():

            # if they are mergable, merge
            if can_merge_neighborhoods(
                graph,
                similarity_matrix,
                query_id,
                target_id,
                threshold
            ):
                print("CAN MERGE!")
                exit(0)
                """
                merged_data = merge_nodes(
                    query_id,
                    target_id
                )
                break
                """

        else:
            # if nothing is mergable, start uniquely disambiguating
            query_id, _ = similarity_matrix.argmax()
            merged_data = merge_nodes(
                query_id,
                target_id=None
            )

        break


if __name__ == "__main__":
    # load in data
    isnad_mention_ids, disambiguated_ids, _ = read_isnad_data(
        "nameData/names.csv",
        "communities/goldStandard_goldTags.json",
        None
    )

    # truncate for testing
    isnad_mention_ids = isnad_mention_ids[:1]
    disambiguated_ids = [
        id
        for id in disambiguated_ids
        if id in sum(isnad_mention_ids, [])
    ]
    #print(isnad_mention_ids)
    #print(disambiguated_ids)

    cooc_graph = create_cooccurence_graph(
        isnad_mention_ids,
        self_edges=False,
        max_isnads=None,
    )

    match_subgraphs(
        isnad_mention_ids,
        disambiguated_ids,
        cooc_graph,
        threshold=0.5
    )

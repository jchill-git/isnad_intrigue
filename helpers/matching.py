from typing import List, Union, Tuple

import tqdm
import numpy as np
import networkx as nx

from helpers.data import read_isnad_data, split_data
from helpers.graph import create_cooccurence_graph
from helpers.features import SimilarityMatrix
from helpers.utils import get_ambiguous_ids, match_list_shape


def merge_nodes(
    isnad_mention_ids: List[List[int]],
    disambiguated_ids: List[int],
    query_id: int,
    target_id: Union[int, None]
) -> Tuple[List[List[int]], List[int]]:
    if target_id is None:
        disambiguated_ids.append(query_id)

        print(f"{query_id} -> unique")

    else:
        isnad_mention_ids_copy = isnad_mention_ids.copy()

        mentions_flattened = sum(isnad_mention_ids, [])
        query_index = mentions_flattened.index(query_id)
        mentions_flattened[query_index] = target_id

        # consolodate disambiguated_ids
        disambiguated_ids = [id for id in disambiguated_ids if id in mentions_flattened]

        isnad_mention_ids = match_list_shape(mentions_flattened, isnad_mention_ids_copy)

        print(f"{query_id} -> {target_id}")

    return isnad_mention_ids, disambiguated_ids


def can_merge_neighborhoods(
    graph: nx.Graph,
    similarities: SimilarityMatrix,
    query_id: int,
    target_id: int,
    threshold: float,
):
    #print(f"can_merge_neighborhoods q={query_id} t={target_id}")
    if similarities[query_id, target_id] < threshold:
        #print("cannot merge, query doesn't match target")
        return False

    return True

    graph_nodes = graph.nodes
    query_neighbor_ids = list(graph.successors(query_id)) + list(graph.predecessors(query_id))
    target_neighbor_ids = list(graph.successors(target_id)) + list(graph.predecessors(target_id))

    for query_neighbor_id in query_neighbor_ids:
        for target_neighbor_id in target_neighbor_ids:
            #print(f"qn: {query_neighbor_id} tn: {target_neighbor_id}: {similarities[query_neighbor_id, target_neighbor_id]}")
            if similarities[query_neighbor_id, target_neighbor_id] > threshold:
                break
        else:
            #print(f"couldn't find matching neighbor for {query_neighbor_id}")
            return False


def match_subgraphs(
    isnad_mention_ids: List[List[int]],
    disambiguated_ids: List[int],
    isnad_mention_embeddings: List[List[List[float]]],
    threshold: float
):
    # avoid modifying inputs
    _isnad_mention_ids = isnad_mention_ids.copy()
    _disambiguated_ids = disambiguated_ids.copy()

    num_ambiguous_ids = len(sum(_isnad_mention_ids, [])) - len(_disambiguated_ids)
    progress = tqdm.tqdm(total=num_ambiguous_ids)
    while True:
        # create graph
        graph = create_cooccurence_graph(
            _isnad_mention_ids,
            isnad_mention_embeddings,
            self_edges=False
        )

        # compute similarities
        similarity_matrix = SimilarityMatrix.from_data(
            graph,
            cooc_alpha = None,
            position_alpha = None,
            nlp_alpha = 1.0,
        )

        # get ambiguous ids
        _ambiguous_ids = get_ambiguous_ids(_isnad_mention_ids, _disambiguated_ids)
        if len(_ambiguous_ids) <= 0:
            break
        print(f"_ambiguous_ids: {_ambiguous_ids}")

        query_target_similarities = similarity_matrix.take_2d(_ambiguous_ids, _disambiguated_ids)

        # find query and target ids with the highest similarity
        # TODO: come back and fix this crap
        for query_id, target_id in query_target_similarities.argsort():
            # if they are mergable, merge
            if can_merge_neighborhoods(
                graph,
                similarity_matrix,
                query_id,
                target_id,
                threshold
            ):
                _isnad_mention_ids, _disambiguated_ids = merge_nodes(
                    _isnad_mention_ids,
                    _disambiguated_ids,
                    query_id,
                    target_id
                )
                break

        else:
            # if nothing is mergable, start uniquely disambiguating
            # take query_id, target_id, which is the lowest confidence pair
            _isnad_mention_ids, _disambiguated_ids = merge_nodes(
                _isnad_mention_ids,
                _disambiguated_ids,
                query_id,
                target_id=None
            )

        num_ambiguous_ids = len(sum(_isnad_mention_ids, [])) - len(_disambiguated_ids)
        progress.update(1)


if __name__ == "__main__":
    # load in data
    isnad_mention_ids, disambiguated_ids, mention_embeddings = read_isnad_data(
        "nameData/names.csv",
        "communities/goldStandard_goldTags.json",
        None#"contrastive_embeddings.json"
    )

    # truncate for testing
    isnad_mention_ids = isnad_mention_ids[:2]
    disambiguated_ids = [
        id
        for id in disambiguated_ids
        if id in sum(isnad_mention_ids, [])
    ]

    match_subgraphs(
        isnad_mention_ids,
        disambiguated_ids,
        mention_embeddings,
        threshold=0.5
    )

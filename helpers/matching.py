from typing import List, Union, Tuple, Optional

import tqdm
import numpy as np
import networkx as nx

from helpers.data import read_isnad_data, split_data
from helpers.graph import create_cooccurence_graph
from helpers.features import SimilarityScorer
from helpers.utils import get_ambiguous_ids, match_list_shape, index_2d_list


def merge_nodes(
    isnad_mention_ids: List[List[int]],
    disambiguated_ids: List[int],
    ambiguous_ids: List[int],
    query_id: int,
    target_id: Union[int, None],
    is_labeled: Optional[List[List[bool]]] = None,
) -> Tuple[List[List[int]], List[int]]:
    if target_id is None:
        disambiguated_ids.append(query_id)

        if is_labeled is not None:
            query_x_index, query_y_index = index_2d_list(isnad_mention_ids, query_id)

            if is_labeled[query_x_index][query_y_index]:
                print(f"{query_id} (labeled) -> unique")
            else:
                print(f"{query_id} -> unique")
        else:
            print(f"{query_id} -> unique")

    else:
        query_x_index, query_y_index = index_2d_list(isnad_mention_ids, query_id)
        isnad_mention_ids[query_x_index][query_y_index] = target_id

        if is_labeled is not None and is_labeled[query_x_index][query_y_index]:
            print(f"{query_id} (labeled) -> {target_id}")
        else:
            print(f"{query_id} -> {target_id}")

    ambiguous_ids.remove(query_id)

    return isnad_mention_ids, disambiguated_ids, ambiguous_ids


def can_merge_neighborhoods(
    graph: nx.Graph,
    similarities: SimilarityScorer,
    query_id: int,
    target_id: int,
    threshold: float,
    check_neighbors: bool = True,
):
    if similarities[query_id, target_id] < threshold:
        return False

    if check_neighbors:
        query_neighbor_ids = list(graph.successors(query_id)) + list(graph.predecessors(query_id))
        target_neighbor_ids = list(graph.successors(target_id)) + list(graph.predecessors(target_id))

        if len(target_neighbor_ids) == 0 or len(query_neighbor_ids) == 0:
            return True

        # If just one query neighbor doesn't match, this algo fails
        # Could replace with jaccard index?
        for query_neighbor_id in query_neighbor_ids:
            for target_neighbor_id in target_neighbor_ids:
                if similarities[query_neighbor_id, target_neighbor_id] > threshold:
                    break
            else:
                return False

    return True


def match_subgraphs(
    isnad_mention_ids: List[List[int]],
    disambiguated_ids: List[int],
    isnad_mention_embeddings: List[List[List[float]]],
    is_labeled: List[List[bool]],
    threshold: float,
    recomputation_schedule: Optional[Tuple[float, float]] = (1, 0),
    computeless_threshold: Optional[float] = np.inf,
    check_neighbors: bool = True,
    **hash_kwargs,
):
    # avoid modifying inputs
    _isnad_mention_ids = isnad_mention_ids.copy()
    _disambiguated_ids = disambiguated_ids.copy()

    # recomputation schedule
    merges_until_recomputation, schedule_change = recomputation_schedule

    _ambiguous_ids = get_ambiguous_ids(_isnad_mention_ids, _disambiguated_ids)
    progress = tqdm.tqdm(total=len(_ambiguous_ids))
    while True:
        # get ambiguous ids, break if there are no more left
        _ambiguous_ids = get_ambiguous_ids(_isnad_mention_ids, _disambiguated_ids)
        if len(_ambiguous_ids) <= 0:
            break

        # create graph
        graph = create_cooccurence_graph(
            _isnad_mention_ids,
            isnad_mention_embeddings,
            self_edges=False
        )

        # compute similarities
        similarities = SimilarityScorer(
            graph,
            _disambiguated_ids,
            **hash_kwargs
        )

        # update schedule
        num_merged_since_recomputation = 0
        merges_until_recomputation = max(merges_until_recomputation + schedule_change, 0)

        # find query and target ids with the highest similarity
        ordered_id_pairs = similarities.argsort_ids(_ambiguous_ids, _disambiguated_ids)
        for query_id, target_id in ordered_id_pairs:
            # it's possible this node has already been merged since the last computation
            if query_id not in _ambiguous_ids:
                continue

            # if they are mergable, merge
            if can_merge_neighborhoods(
                graph,
                similarities,
                query_id,
                target_id,
                threshold,
                check_neighbors=check_neighbors,
            ):
                _isnad_mention_ids, _disambiguated_ids, _ambiguous_ids = merge_nodes(
                    _isnad_mention_ids,
                    _disambiguated_ids,
                    _ambiguous_ids,
                    query_id,
                    target_id,
                    is_labeled=is_labeled,
                )

                # check whether to recompute features
                num_merged_since_recomputation += 1
                if (
                    num_merged_since_recomputation < merges_until_recomputation or
                    similarities[query_id, target_id] >= computeless_threshold
                ):
                    continue
                else:
                    break

        # if nothing is mergable, start uniquely disambiguating
        else:
            for query_id, target_id in reversed(ordered_id_pairs):
                # it's possible this node has already been merged since the last computation
                if query_id not in _ambiguous_ids:
                    continue

                _isnad_mention_ids, _disambiguated_ids, _ambiguous_ids = merge_nodes(
                    _isnad_mention_ids,
                    _disambiguated_ids,
                    _ambiguous_ids,
                    query_id,
                    target_id=None,
                    is_labeled=is_labeled,
                )

                # check whether to recompute features
                num_merged_since_recomputation += 1
                if (
                    num_merged_since_recomputation < merges_until_recomputation or
                    similarities[query_id, target_id] >= computeless_threshold
                ):
                    continue

        progress.update(num_merged_since_recomputation)

    return _isnad_mention_ids, _disambiguated_ids


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

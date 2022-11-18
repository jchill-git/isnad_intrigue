from typing import List, Tuple

import copy
import csv
import json
import numpy as np

from helpers.utils import max_list_of_lists, match_list_shape


def read_isnad_data(
    isnad_names_path: str,
    isnad_labels_path: str,
):
    # read raw isnad data
    isnad_lengths = _read_isnad_lengths(isnad_names_path)
    isnad_labels = _read_isnad_labels(isnad_labels_path)

    # build mention_labels from raw data
    isnad_mention_ids = [
        [
            isnad_labels[isnad_index][node_index]
            if isnad_index in isnad_labels and node_index in isnad_labels[isnad_index]
            else None
            for node_index in range(isnad_length)
        ]
        for isnad_index, isnad_length in enumerate(isnad_lengths)
    ]

    # iterate through mentions and replace nones with incrementing ids
    largest_labeled_node_id = max_list_of_lists(isnad_mention_ids)
    node_id_counter = largest_labeled_node_id + 1
    for isnad_index, isnad_nodes in enumerate(isnad_mention_ids):
        for isnad_node_index, isnad_node in enumerate(isnad_nodes):
            if isnad_node is None:
                isnad_mention_ids[isnad_index][isnad_node_index] = node_id_counter
                node_id_counter += 1

    # create list of ids that are disambiguated
    max_id_value = max_list_of_lists(isnad_mention_ids)
    disambiguated_ids = [
        id
        for id in range(max_id_value + 1)
        if id <= largest_labeled_node_id
    ]

    return isnad_mention_ids, disambiguated_ids


def create_test_split(
    isnad_mention_ids: List[List[int]],
    disambiguated_ids: List[int],
    test_size: float
) -> Tuple[List[List[int]], List[int]]:
    """
    Creates a test set of mention ids by sequentally
    ambiguating previous disambiguous mentions
    """
    isnad_mentions_ids_copy = copy.deepcopy(isnad_mention_ids)

    mention_ids_flattened = sum(isnad_mentions_ids_copy, [])
    disambiguated_indices = [
        mention_index
        for mention_index, mention_id in enumerate(mention_ids_flattened)
        if mention_id in disambiguated_ids
    ]

    # choose indices to ambiguate
    num_indices_to_ambiguate = int(test_size * len(disambiguated_indices))
    indices_to_ambiguate = np.random.choice(
        disambiguated_indices,
        num_indices_to_ambiguate,
        replace=False
    )

    largest_mention_id = max(mention_ids_flattened)
    for index_to_ambiguate in indices_to_ambiguate:

        # ambiguate the mention at that index
        mention_ids_flattened[index_to_ambiguate] = largest_mention_id + 1
        largest_mention_id += 1

    # reshape back to isnad_mention_ids shape
    test_isnad_mention_ids = match_list_shape(mention_ids_flattened, isnad_mention_ids)

    # note disambiguated_ids is unchanged
    return test_isnad_mention_ids, disambiguated_ids

def _read_isnad_lengths(file_path: str):
    isnad_lengths = []

    with open(file_path) as names_csv_file:
        reader = csv.reader(names_csv_file)
        _ = next(reader)
        return [
            np.count_nonzero(row) - 1
            for row in reader
        ]

    return isnad_lengths


def _read_isnad_labels(file_path: str):
    gold_entities = [json.loads(l) for l in open(file_path, "r")]

    isnad_labels = {}
    for entity in gold_entities:
        _, _, isnad_id, isnad_name_number = entity["mentionID"].split("_")
        isnad_id = int(isnad_id)
        isnad_name_number = int(isnad_name_number)

        if isnad_id not in isnad_labels:
            isnad_labels[isnad_id] = {}

        isnad_labels[isnad_id][isnad_name_number] = entity["community"]

    return isnad_labels

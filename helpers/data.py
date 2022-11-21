from typing import List, Tuple, Optional

import csv
import json
import numpy as np
from parse import parse

from helpers.utils import max_list_of_lists, match_list_shape


def read_isnad_data(
    isnad_names_path: str,
    isnad_labels_path: str,
    isnad_embeddings_path: Optional[str] = None,
):
    # read raw isnad data
    isnad_lengths = _read_isnad_lengths(isnad_names_path)
    isnad_labels = _read_isnad_labels(isnad_labels_path)
    if isnad_embeddings_path is not None:
        isnad_embeddings = _read_isnad_embeddings(isnad_embeddings_path)
    else:
        isnad_embeddings = None

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

    return isnad_mention_ids, disambiguated_ids, isnad_embeddings


def split_data(
    isnad_mention_ids: List[List[int]],
    disambiguated_ids: List[int],
    test_mentions: Optional[List[List[bool]]] = None,
    test_size: Optional[float] = None
) -> Tuple[List[List[int]], List[int]]:
    """
    Creates a test set of mention ids by sequentally
    ambiguating previous disambiguous mentions
    """
    if test_mentions is None == test_size is None:
        raise ValueError(
            "Must include either a boolean list of lists denoting which mentions "
            "to use as test or test_size"
        )

    # randomly split if test_mentions is not given
    if test_mentions is None:
        test_mentions = random_split_test_mentions(
            isnad_mention_ids,
            disambiguated_ids,
            test_size=test_size
        )

    # flatten and prepare inputs
    isnad_mentions_ids_copy = isnad_mention_ids.copy()
    mention_ids_flattened = sum(isnad_mentions_ids_copy, [])
    test_mentions_flattened = sum(test_mentions, [])
    indices_to_ambiguate = [
        index
        for index, is_test in enumerate(test_mentions_flattened)
        if is_test
    ]
    assert len(mention_ids_flattened) == len(test_mentions_flattened)

    # ambiguate mentions
    largest_mention_id = max(mention_ids_flattened)
    for index_to_ambiguate in indices_to_ambiguate:

        # ambiguate the mention at that index
        mention_ids_flattened[index_to_ambiguate] = largest_mention_id + 1
        largest_mention_id += 1

    # reshape back to isnad_mention_ids shape
    test_isnad_mention_ids = match_list_shape(mention_ids_flattened, isnad_mention_ids)

    # note disambiguated_ids is unchanged
    return test_isnad_mention_ids, disambiguated_ids


def random_split_test_mentions(
    isnad_mention_ids: List[List[int]],
    disambiguated_ids: List[int],
    test_size: float
):
    isnad_mentions_ids_copy = isnad_mention_ids.copy()
    mention_ids_flattened = sum(isnad_mentions_ids_copy, [])
    disambiguated_indices = [
        mention_index
        for mention_index, mention_id in enumerate(mention_ids_flattened)
        if mention_id in disambiguated_ids
    ]

    num_indices_to_ambiguate = int(test_size * len(disambiguated_indices))
    indices_to_ambiguate = np.random.choice(
        disambiguated_indices,
        num_indices_to_ambiguate,
        replace=False
    )

    test_mentions_flattened = [False for _ in mention_ids_flattened]
    for index in indices_to_ambiguate:
        test_mentions_flattened[index] = True

    test_mentions = match_list_shape(test_mentions_flattened, isnad_mention_ids)

    return test_mentions


def read_isnad_names(isnad_names_path: str):
    isnad_names = []

    with open(isnad_names_path) as names_csv_file:
        reader = csv.reader(names_csv_file)
        _ = next(reader)

        for row in reader:
            names = [name for name in row[4:] if name != ""]
            isnad_names.append(names)

    return isnad_names


def _read_isnad_lengths(file_path: str):
    with open(file_path) as names_csv_file:
        reader = csv.reader(names_csv_file)
        _ = next(reader)
        return [
            np.count_nonzero(row[4:])
            for row in reader
        ]


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


def _read_isnad_embeddings(file_path: str):
    with open(file_path, "r") as embeddings_file:
        mention_embeddings = [json.loads(line) for line in embeddings_file]

    isnad_mention_embeddings = []
    for mention_embedding in mention_embeddings:
        isnad_index, mention_index = parse("JK_000916_{}_{}", mention_embedding["id"])
        isnad_index = int(isnad_index)
        mention_index = int(mention_index)

        if isnad_index > len(isnad_mention_embeddings):
            raise ValueError("isnads in embeddings file should be ordered")
        if isnad_index == len(isnad_mention_embeddings):
            isnad_mention_embeddings.append([])

        if mention_index > len(isnad_mention_embeddings[isnad_index]):
            raise ValueError("isnads in embeddings file should be ordered")
        isnad_mention_embeddings[isnad_index].append(mention_embedding["embedding"])

    return isnad_mention_embeddings

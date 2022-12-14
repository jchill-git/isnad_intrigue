from typing import List, Any, Optional

from collections import deque
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def index_2d_list(twod_list, target):
    for x_index, oned_list in enumerate(twod_list):
        for y_index, value in enumerate(oned_list):
            if value == target:
                return x_index, y_index

    raise ValueError(f"could not find {target} in list of lists")


def get_labeled_mentions(
    true_isnad_mention_ids: List[List[int]],
    true_disambiguated_ids: List[int]
):
    true_isnad_mention_ids_copy = true_isnad_mention_ids.copy()
    mention_ids_flattened = sum(true_isnad_mention_ids, [])
    labeled_mentions_flattened = [
        mention_id in true_disambiguated_ids
        for mention_id in mention_ids_flattened
    ]

    return match_list_shape(labeled_mentions_flattened, true_isnad_mention_ids_copy)



def max_list_of_lists(list_of_lists: List[List[Any]]):
    flattened = sum(list_of_lists, [])
    flattened_without_nones = [value for value in flattened if value is not None]
    return max(flattened_without_nones)


def match_list_shape(values, list_to_match):
    values.reverse() # treat as value stack
    return _match_shape(values, list_to_match)


def invert_list(values):
    values_copy = values.copy()

    values_flattened = sum(values, [])
    values_flattened_inverted = [not values for values in values_flattened]

    return match_list_shape(values_flattened_inverted, values_copy)


def get_ambiguous_ids(
    isnad_mention_ids: List[List[int]],
    disambiguated_ids: List[int],
    is_labeled: Optional[List[List[bool]]] = None
) -> List[int]:
    isnad_mention_ids_flattened = sum(isnad_mention_ids, [])
    is_labeled_flattened = (
        sum(is_labeled, [])
        if is_labeled is not None else
        [True for _ in range(len(isnad_mention_ids_flattened))]
    )
    unique_ids = np.unique(
        [
            id
            for index, id in enumerate(isnad_mention_ids_flattened)
            if is_labeled_flattened[index]
        ]
    )
    ambiguous_ids = [id for id in unique_ids if id not in disambiguated_ids]

    return ambiguous_ids


def show_graph(graph: nx.Graph, disambiguated_ids: List[int]):
    positions = nx.spring_layout(graph)

    node_color = [
        "red" if node_id in disambiguated_ids else "blue"
        for node_id in graph.nodes
    ]

    nx.draw(
        graph,
        pos=positions,
        node_color=node_color
    )

    nx.draw_networkx_labels(
        graph,
        pos=positions,
    )

    position_sum_labels = nx.get_edge_attributes(graph, "relative_position_sum")
    num_coocurrences_labels = nx.get_edge_attributes(graph, "num_coocurrences")
    relative_position_labels = {
        key: position_sum_labels[key] / num_coocurrences_labels[key]
        for key in position_sum_labels.keys()
    }
    nx.draw_networkx_edge_labels(
        graph,
        pos=positions,
        edge_labels=relative_position_labels,
        font_size=3,
    )

    plt.show()


def _match_shape(value_stack, to_match):
    """
    Recursive helper function for match_list_shape
    """
    if not isinstance(to_match, list):
        return value_stack.pop()

    return [
        _match_shape(value_stack, sub_list)
        for sub_list in to_match
    ]


# Unit test list shape matching
if __name__ == "__main__":
    a = [0, 1, 2, 3, 4]
    b = [8, [[8, [8]]], [[8], 8]]

    c = match_list_shape(a, b)
    print(c)

from typing import List, Any

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def max_list_of_lists(list_of_lists: List[List[Any]]):
    flattened = sum(list_of_lists, [])
    flattened_without_nones = [value for value in flattened if value is not None]
    return max(flattened_without_nones)


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


#def split_graph_labels(graph, )
#    pass

from typing import Optional, List

import networkx as nx


def create_cooccurence_graph(
    isnad_mention_ids: List[List[int]],
    self_edges: bool = False,
    max_isnads: Optional[int] = None,
):
    # truncate to max_isnads
    max_isnads = max_isnads or len(isnad_mention_ids)
    isnad_mention_ids = isnad_mention_ids[:max_isnads]

    # create graph
    graph = nx.DiGraph()

    # add cliques
    for mention_ids in isnad_mention_ids:
        _add_clique(graph, mention_ids, self_edges=self_edges)

    return graph


def _create_isnad_graph(isnad_data):
    """
    Only used for demonstration purposes
    """

    graph = nx.DiGraph()
    for isnad_node_ids in isnad_data["mentions_data"]:
        for node_index, node_id in enumerate(isnad_node_ids[:-1]):
            next_node_id = isnad_node_ids[node_index + 1]
            graph.add_edge(node_id, next_node_id)

    node_color = [
        "red" if node_id <= isnad_data["largest_labeled_node_id"] else "blue"
        for node_id in graph.nodes
    ]

    return graph, node_color


def _add_clique(graph, isnad_node_ids, self_edges=False):
    for node_index, node_id in enumerate(isnad_node_ids):
        for relative_position, next_node_id in enumerate(isnad_node_ids[node_index:]):
            if not self_edges and node_id == next_node_id: continue

            # if forward edge exists,
            # increment co-occurance and increase relative position
            if graph.has_edge(node_id, next_node_id):
                graph[node_id][next_node_id]["num_coocurrences"] += 1
                graph[node_id][next_node_id]["relative_position_sum"] += (
                    relative_position
                )

            # if backward edge exists,
            # increment co-occurance and decrease relative position
            elif graph.has_edge(next_node_id, node_id):
                graph[next_node_id][node_id]["num_coocurrences"] += 1
                graph[next_node_id][node_id]["relative_position_sum"] -= (
                    relative_position
                )

            # else add new forward edge
            else:
                graph.add_edge(
                    node_id,
                    next_node_id,
                    num_coocurrences=1,
                    relative_position_sum=relative_position
                )

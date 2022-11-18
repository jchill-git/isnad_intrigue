import networkx as nx
import numpy as np

from build_social_network import create_cooccurence_graph, show_graph


def hash_node(graph, node_id, alpha = 0.1):
    hash = get_social_hash(graph, node_id) + alpha * get_nlp_embeddings()


def get_social_hash(graph, node):
    num_nodes = graph.number_of_nodes()
    cooc_hash = np.zeros(num_nodes)
    position_hash = np.zeros(num_nodes)

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

    return np.concatenate((cooc_hash, position_hash))


def load_nlp_embeddings():
    pass


def score_all_vertices():
    pass


def merge_vertices(vertices, ):
    pass


if __name__ == "__main__":
    graph, node_color = create_cooccurence_graph(
        "nameData/names_disambiguated.csv",
        "communities/goldStandard_goldTags.json",
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

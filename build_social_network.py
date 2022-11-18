import csv
import json
import numpy
import matplotlib.pyplot as plt

import networkx as nx

GOLD_JSON_PATH = "communities/goldStandard_goldTags.json"
NAMES_CSV_PATH = "nameData/names_disambiguated.csv"
ADD_SELF_EDGES = True


def read_isnad_lengths():
    isnad_lengths = []

    with open(NAMES_CSV_PATH) as names_csv_file:
        reader = csv.reader(names_csv_file)
        _ = next(reader)
        return [
            numpy.count_nonzero(row) - 1
            for row in reader
        ]

    return isnad_lengths


def read_isnad_labels():
    gold_entities = [json.loads(l) for l in open(GOLD_JSON_PATH, "r")]

    isnad_labels = {}
    for entity in gold_entities:
        _, _, isnad_id, isnad_name_number = entity["mentionID"].split("_")
        isnad_id = int(isnad_id)
        isnad_name_number = int(isnad_name_number)

        if isnad_id not in isnad_labels:
            isnad_labels[isnad_id] = {}

        isnad_labels[isnad_id][isnad_name_number] = entity["community"]

    return isnad_labels


def create_isnad_data(isnad_lengths, isnad_labels, max_isnads=None):
    max_isnads = max_isnads or len(isnad_lengths)
    mentions_data = [
        [
            isnad_labels[isnad_index][node_index]
            if isnad_index in isnad_labels and node_index in isnad_labels[isnad_index]
            else None
            for node_index in range(isnad_length)
        ]
        for isnad_index, isnad_length in enumerate(isnad_lengths)
    ][:max_isnads]

    mentions_data_flatten = sum(mentions_data, [])
    mentions_data_flatten_no_nones = [value for value in mentions_data_flatten if value is not None]
    largest_labeled_node_id = numpy.max(mentions_data_flatten_no_nones)

    node_id_counter = largest_labeled_node_id + 1
    for isnad_index, isnad_nodes in enumerate(mentions_data):
        for isnad_node_index, isnad_node in enumerate(isnad_nodes):
            if isnad_node is None:
                mentions_data[isnad_index][isnad_node_index] = node_id_counter
                node_id_counter += 1

    return {
        "mentions_data": mentions_data,
        "largest_labeled_node_id": largest_labeled_node_id,
    }


def create_isnad_graph(isnad_data):
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


def _add_clique(graph, isnad_node_ids):
    for node_index, node_id in enumerate(isnad_node_ids):
        for relative_position, next_node_id in enumerate(isnad_node_ids[node_index:]):
            if not ADD_SELF_EDGES and node_id == next_node_id: continue

            # if forward edge exists, increment co-occurance and increase relative position
            if graph.has_edge(node_id, next_node_id):
                graph[node_id][next_node_id]["num_coocurrences"] += 1
                graph[node_id][next_node_id]["relative_position_sum"] += relative_position

            # if backward edge exists, increment co-occurance and decrease relative position
            elif graph.has_edge(next_node_id, node_id):
                graph[next_node_id][node_id]["num_coocurrences"] += 1
                graph[next_node_id][node_id]["relative_position_sum"] -= relative_position

            # else add new forward edge
            else:
                graph.add_edge(
                    node_id,
                    next_node_id,
                    num_coocurrences=1,
                    relative_position_sum=relative_position
                )


def create_cooccurence_graph(isnad_data):
    graph = nx.DiGraph()

    for isnad_node_ids in isnad_data["mentions_data"]:
        _add_clique(graph, isnad_node_ids)

    node_color = [
        "red" if node_id <= isnad_data["largest_labeled_node_id"] else "blue"
        for node_id in graph.nodes
    ]

    return graph, node_color

def show_graph(graph, node_color):
    positions = nx.spring_layout(graph)

    nx.draw(
        graph,
        pos=positions,
        node_color=node_color
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

if __name__ == "__main__":
    isnad_lengths = read_isnad_lengths()
    isnad_labels = read_isnad_labels()
    isnad_data = create_isnad_data(isnad_lengths, isnad_labels, max_isnads=100)
    print(isnad_data)

    graph, node_color = create_cooccurence_graph(isnad_data)
    print(graph)

    show_graph(graph, node_color)

import csv
import json
import numpy
import matplotlib.pyplot as plt

import networkx as nx

GOLD_JSON_PATH = "communities/goldStandard_goldTags.json"
NAMES_CSV_PATH = "nameData/names_disambiguated.csv"


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
    return [
        [
            isnad_labels[isnad_index][node_index]
            if isnad_index in isnad_labels and node_index in isnad_labels[isnad_index]
            else None
            for node_index in range(isnad_length)
        ]
        for isnad_index, isnad_length in enumerate(isnad_lengths)
    ][:max_isnads]


def create_graph(isnad_data):
    isnad_data_flatten = sum(isnad_data, [])
    isnad_data_flatten_no_nones = [value for value in isnad_data_flatten if value is not None]
    largest_labeled_node_id = numpy.max(isnad_data_flatten_no_nones)

    node_id_counter = largest_labeled_node_id + 1
    for isnad_index, isnad_nodes in enumerate(isnad_data):
        for isnad_node_index, isnad_node in enumerate(isnad_nodes):
            if isnad_node is None:
                isnad_data[isnad_index][isnad_node_index] = node_id_counter
                node_id_counter += 1

    graph = nx.DiGraph()
    for isnad_nodes in isnad_data:
        for node_index, community in enumerate(isnad_nodes[:-1]):
            next_community = isnad_nodes[node_index + 1]
            graph.add_edge(community, next_community)

    node_color = [
        "red" if node_id <= largest_labeled_node_id else "blue"
        for node_id in graph.nodes
    ]

    return graph, node_color

if __name__ == "__main__":
    isnad_lengths = read_isnad_lengths()
    isnad_labels = read_isnad_labels()
    isnad_data = create_isnad_data(isnad_lengths, isnad_labels, max_isnads=100)

    graph, node_color = create_graph(isnad_data)

    print(graph)

    nx.draw(graph, node_color=node_color)
    plt.show()

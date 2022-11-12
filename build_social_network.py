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

def _add_clique(graph, node_ids):
    pass


def create_cooccurence_graph(isnad_data):
    graph=nx.Graph()
    
    #create an edge between every node in an isnad
    
    #for each isnad
    for insnad_node_id in isnad_data['mentions_data']:
        
        #for each node in the isnad
        for node_index, node_id in enumerate(isnad_node_ids[:-1]):
            
            #for every subsequent node in the isnad
            for next_node_index, next_node_id in enumerate(isnad_node_ids[node_index:-1]:
          
                #if edge exists increment  weight
                if graph.has_edge(node_id, next_node_id):
                    graph[node_id][next_node_id]['weight'] += 1
                                                           
                #else add new edge
                graph.add_edge(node_id, next_node_id,weight=1)
        
    #set node color
    node_color = [
        "red" if node_id <= isnad_data["largest_labeled_node_id"] else "blue"
        for node_id in graph.nodes
    ]
                                                           
    #return graph
    return graph, node_color


if __name__ == "__main__":
    isnad_lengths = read_isnad_lengths()
    isnad_labels = read_isnad_labels()
    isnad_data = create_isnad_data(isnad_lengths, isnad_labels, max_isnads=3)
    print(isnad_data)

    graph, node_color = create_isnad_graph(isnad_data)
    #graph, node_color = create_cooccurence_graph(isnad_data)
    print(graph)

    nx.draw(graph, node_color=node_color)
    plt.show()

import csv
import json
import matplotlib.pyplot as plt

import networkx as nx

GOLD_JSON_PATH = "communities/goldStandard_goldTags.json"
NAMES_CSV_PATH = "nameData/names.csv"

if __name__ == "__main__":
    isnads_names = []
    with open(NAMES_CSV_PATH) as names_csv_file:
        reader = csv.reader(names_csv_file)
        for row_i, row in enumerate(reader):
            if row_i == 0: continue

            names = [value for value in row[4:] if value != ""]
            isnads_names.append(names)

    print(isnads_names[0])

    graph = nx.DiGraph()
    for isnad_name_i, isand_names in enumerate(isnads_names):
        graph.add_edge(isnad_name_i, isnad_name_i + 1)

    exit(0)


    gold_entities = [json.loads(l) for l in open(GOLD_JSON_PATH, "r")]

    isnad_entities = {}
    for entity in gold_entities:
        _, _, isnad_id, isnad_name_number = entity["mentionID"].split("_")
        isnad_id = int(isnad_id)
        isnad_name_number = int(isnad_name_number)

        if isnad_id not in isnad_entities:
            isnad_entities[isnad_id] = {}

        isnad_entities[isnad_id][isnad_name_number] = entity["community"]

    print(isnad_entities[0])

    for isnad_id, isnad in isnad_entities.items():
        index_communities = [[index, community] for index, community in isnad.items()]

        index_communities_sorted = sorted(index_communities, key=lambda x: x[0])
        communities_sorted = [community for index, community in index_communities_sorted]

        isnad_entities[isnad_id] = communities_sorted

    print(isnad_entities[0])

    graph = nx.DiGraph()
    for isnad in isnad_entities.values():
        for index, community in enumerate(isnad[:-1]):
            next_community = isnad[index + 1]
            graph.add_edge(community, next_community)

    print(len(graph.nodes))

    nx.draw(graph)
    plt.show()

import json

from helpers.data import read_isnad_data, split_data
from helpers.graph import create_cooccurence_graph
from helpers.utils import show_graph
from helpers.matching import match_subgraphs

if __name__ == "__main__":
    # load in true data
    true_isnad_mention_ids, true_disambiguated_ids, isnad_mention_embeddings = read_isnad_data(
        "nameData/names.csv",
        "communities/goldStandard_goldTags.json",
        "contrastive_embeddings.json"
    )

    # split into test graph
    with open("test_mentions.json", "r") as test_mentions_file:
        test_mentions = json.load(test_mentions_file)

    # truncate for testing
    true_isnad_mention_ids = true_isnad_mention_ids[:5]
    test_mentions = test_mentions[:5]

    # split
    test_mention_ids, test_disambiguated_ids = split_data(
        true_isnad_mention_ids,
        true_disambiguated_ids,
        test_mentions=test_mentions,
        #test_size=0.9
    )

    # evaluation
    match_subgraphs(
        test_mention_ids,
        test_disambiguated_ids,
        isnad_mention_embeddings,
        threshold=0.1
    )

    # pred_graph = merge_and_stuff(test_graph)
    # pred_node_ids = pred_graph.nodes()
    # true_node_ids = true_graph.nodes()
    # conll_score(true_node_ids, pred_node_ids)

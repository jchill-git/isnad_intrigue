import json

from helpers.data import read_isnad_data, split_data
from helpers.graph import create_cooccurence_graph
from helpers.utils import show_graph
from helpers.matching import match_subgraphs
from helpers.evaluation import create_communities_file, createClusters, calc_conLL

if __name__ == "__main__":
    # arguments
    gold_path = "communities/goldStandard_goldTags.json"
    embeddings_path = "contrastive_embeddings.json"
    output_file_name = "test_communities.json"

    # load in true data
    true_isnad_mention_ids, true_disambiguated_ids, isnad_mention_embeddings = read_isnad_data(
        "nameData/names.csv",
        gold_path,
        embeddings_path
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
    pred_mention_ids, pred_disambiguated_ids = match_subgraphs(
        test_mention_ids,
        test_disambiguated_ids,
        isnad_mention_embeddings,
        threshold=0.1
    )

    print()

    #create community_file
    create_communities_file(output_file_name, pred_mention_ids)

    #create model clusters
    modelClusters = createClusters(output_file_name)

    #create goldStandard clusters
    goldClusters = createClusters(gold_path, max_samples=len(sum(true_isnad_mention_ids, [])))

    #calculate ConLL score
    calc_conLL(goldClusters, modelClusters)

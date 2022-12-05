import json
import numpy as np

from helpers.data import read_isnad_data, split_data
from helpers.graph import create_cooccurence_graph
from helpers.utils import show_graph, get_ambiguous_ids, get_labeled_mentions
from helpers.matching import match_subgraphs
from helpers.evaluation import create_communities_file, createClusters, calc_conLL

if __name__ == "__main__":
    # arguments
    names_path = "nameData/names.csv"
    gold_path = "communities/goldStandard_goldTags.json"
    embeddings_path = "contrastive_embeddings.json"
    output_file_name = "test_communities.json"
    test_mentions_path = "test_mentions.json"

    # load in data
    true_isnad_mention_ids, true_disambiguated_ids, isnad_mention_embeddings = read_isnad_data(
        names_path,
        gold_path,
        embeddings_path
    )
    with open(test_mentions_path, "r") as test_mentions_file:
        test_mentions = json.load(test_mentions_file)

    # truncate for testing
    true_isnad_mention_ids = true_isnad_mention_ids[:50]
    test_mentions = test_mentions[:50]

    # split
    test_mention_ids, test_disambiguated_ids = split_data(
        true_isnad_mention_ids,
        true_disambiguated_ids,
        test_mentions=test_mentions,
        #test_size=0.0
    )

    # print some stats
    _ambiguous_ids = get_ambiguous_ids(test_mention_ids, test_disambiguated_ids)
    test_mentions_flattened = sum(test_mentions, [])
    print(
        f"num_disambiguated: {len(test_disambiguated_ids)}, "
        f"num_ambiguous: {len(_ambiguous_ids)}, "
        f"num_ambiguous_labeled: {np.count_nonzero(test_mentions_flattened)}"
    )

    # evaluation
    #"""
    pred_mention_ids, pred_disambiguated_ids = match_subgraphs(
        test_mention_ids,
        test_disambiguated_ids,
        isnad_mention_embeddings,
        is_labeled = test_mentions,  # used for printing
        check_neighbors = False,
        threshold = 0.7,
        cooc_alpha = 0.0,
        position_alpha = 0.0,
        nlp_alpha = 1.0,
    )
    #"""

    #pred_mention_ids = true_isnad_mention_ids # truth
    #pred_mention_ids = test_mention_ids # baseline

    labeled_mentions = get_labeled_mentions(true_isnad_mention_ids, true_disambiguated_ids)

    # create community_file
    create_communities_file(output_file_name, pred_mention_ids, labeled_mentions=labeled_mentions)

    # create model clusters
    modelClusters = createClusters(output_file_name)

    # create goldStandard clusters
    goldClusters = createClusters(gold_path, labeled_mentions=labeled_mentions)

    # calculate ConLL score
    calc_conLL(goldClusters, modelClusters)


# 2. not check ambiguous -> ambiguous neighbors

# 3. replace similarity with generator - Kyle
# 4. mulit-threading

# a. jaccard index
# b. deep features

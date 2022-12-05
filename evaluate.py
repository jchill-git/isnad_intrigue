import json

from helpers.data import read_isnad_data, split_data, _create_communities_file,createClusters,calc_conLL
from helpers.graph import create_cooccurence_graph
from helpers.utils import show_graph

if __name__ == "__main__":
    # load in true data
    embeddings_path="contrastive_embeddings.json"

    true_isnad_mention_ids, true_disambiguated_ids, isnad_mention_embeddings = read_isnad_data(
        "nameData/names.csv",
        "communities/goldStandard_goldTags.json",
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

    test_graph = create_cooccurence_graph(
        test_mention_ids,
        isnad_mention_embeddings,
        self_edges=False,
        max_isnads=None,
    )
    
    #show_graph(test_graph, test_disambiguated_ids)

    #create community_file
    community_file_name='test_communities.json'
    _create_communities_file(community_file_name,test_mention_ids)

    #create model clusters
    modelClusters = createClusters(community_file_name)

    #create goldStandard clusters
    gold_path='.\communities\goldStandard_goldTags.json'
    goldClusters = createClusters(gold_path)

    #calculate ConLL score
    calc_conLL(goldClusters,modelClusters)

    # pred_graph = merge_and_stuff(test_graph)
    # pred_node_ids = pred_graph.nodes()
    # true_node_ids = true_graph.nodes()
    # conll_score(true_node_ids, pred_node_ids)

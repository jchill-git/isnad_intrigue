from helpers.data import read_isnad_data, create_test_split
from helpers.graph import create_cooccurence_graph
from helpers.utils import show_graph

if __name__ == "__main__":
    # load in true data
    true_isnad_mention_ids, true_disambiguated_ids = read_isnad_data(
        "nameData/names_disambiguated.csv",
        "communities/goldStandard_goldTags.json",
    )

    # truncate for testing
    true_isnad_mention_ids = true_isnad_mention_ids[0:5]
    print(true_isnad_mention_ids)

    true_graph = create_cooccurence_graph(
        true_isnad_mention_ids,
        self_edges=False,
        max_isnads=None,
    )
    show_graph(true_graph, true_disambiguated_ids)

    # split into test graph
    test_mention_ids, test_disambiguated_ids = create_test_split(
        true_isnad_mention_ids,
        true_disambiguated_ids,
        test_size=0.9
    )
    print(test_mention_ids)

    test_graph = create_cooccurence_graph(
        test_mention_ids,
        self_edges=False,
        max_isnads=None,
    )
    show_graph(test_graph, test_disambiguated_ids)

    # evaluation

    # pred_graph = merge_and_stuff(test_graph)
    # pred_node_ids = pred_graph.nodes()
    # true_node_ids = true_graph.nodes()
    # conll_score(true_node_ids, pred_node_ids)

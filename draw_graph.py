from helpers.data import read_isnad_data
from helpers.graph import create_cooccurence_graph
from helpers.utils import show_graph


if __name__ == "__main__":
    isnad_mention_ids, disambiguated_ids, _ = read_isnad_data(
        "nameData/names.csv",
        "communities/goldStandard_goldTags.json",
        "nameData/namesWithEmbeddings_NER_strict.json"
    )

    graph = create_cooccurence_graph(
        isnad_mention_ids,
        self_edges=False,
        max_isnads=1,
    )

    show_graph(graph, disambiguated_ids)

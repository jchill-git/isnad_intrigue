import numpy
import matplotlib.pyplot as plt

from helpers.data import read_isnad_data


def get_position_distribution(isnad_mention_ids, id_i, id_j):
    position_distribution = []
    for mention_ids in isnad_mention_ids:
        try:
            index_i = mention_ids.index(id_i)
            index_j = mention_ids.index(id_j)
            position_distribution.append(index_j - index_i)
        except Exception:
            pass

    return position_distribution




if __name__ == "__main__":
    # arguments
    names_path = "nameData/names.csv"
    gold_path = "communities/goldStandard_goldTags.json"
    output_file_name = "test_communities.json"
    test_mentions_path = "test_mentions.json"

    true_isnad_mention_ids, true_disambiguated_ids, _ = read_isnad_data(
        names_path,
        gold_path,
        None
    )

    all_positions = []
    all_positions_std = []
    for index_i, id_i in enumerate(true_disambiguated_ids):
        for id_j in true_disambiguated_ids[(index_i + 1):]:
            positions = get_position_distribution(true_isnad_mention_ids, id_i, id_j)
            positions_std = numpy.std(positions) if len(positions) > 0 else None
            all_positions += positions
            if positions_std is not None:
                all_positions_std.append(positions_std)

    plt.hist(numpy.abs(all_positions))
    plt.title("Absolute value of all relative positions")
    plt.show()

    plt.hist(all_positions_std)
    plt.title("Standard deviations of relative positions")
    plt.show()

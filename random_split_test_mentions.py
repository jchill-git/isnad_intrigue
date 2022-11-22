import json
import argparse

from helpers.data import read_isnad_data, random_split_test_mentions

parser = argparse.ArgumentParser()
parser.add_argument("file_path", type=str)
parser.add_argument("--test_size", type=float, default=0.2)

if __name__ == "__main__":
    args = parser.parse_args()

    isnad_mention_ids, disambiguated_ids, _ = read_isnad_data(
        "nameData/names.csv",
        "communities/goldStandard_goldTags.json",
        "nameData/namesWithEmbeddings_NER_strict.json"
    )

    test_mentions = random_split_test_mentions(
        isnad_mention_ids,
        disambiguated_ids,
        test_size=args.test_size
    )

    with open(args.file_path, "w") as out_file:
        json.dump(test_mentions, out_file)

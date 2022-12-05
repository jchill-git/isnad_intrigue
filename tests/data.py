import pytest

from helpers.data import read_isnad_data
from helpers.features import cosine_similarity

_SIMILARITY_THRESHOLD = 0.0

def test_contrastive_embeddings():
    # load in true data
    true_isnad_mention_ids, true_disambiguated_ids, isnad_mention_embeddings = read_isnad_data(
        "nameData/names.csv",
        "communities/goldStandard_goldTags.json",
        "contrastive_embeddings.json"
    )

    # check shapes are the same
    assert len(true_isnad_mention_ids) == len(isnad_mention_embeddings)
    assert len(true_isnad_mention_ids[0]) == len(isnad_mention_embeddings[0])
    assert len(true_isnad_mention_ids[-1]) == len(isnad_mention_embeddings[-1])

    # two similar but distinct surface forms of Sulayman b. Ishaq
    similarity = cosine_similarity(
        isnad_mention_embeddings[2356][1],
        isnad_mention_embeddings[2362][1]
    )
    assert similarity == pytest.approx(0.90, abs=0.01)

    # two completely different surface forms
    similarity = cosine_similarity(
        isnad_mention_embeddings[1][1],
        isnad_mention_embeddings[-1][0]
    )
    assert similarity == pytest.approx(-0.09, abs=0.01)

    # two identical different surface forms
    similarity = cosine_similarity(
        isnad_mention_embeddings[0][5],
        isnad_mention_embeddings[3][5]
    )
    assert similarity == pytest.approx(1.0, abs=0.01)


def test_ryan_embeddings():
    # load in true data
    true_isnad_mention_ids, true_disambiguated_ids, isnad_mention_embeddings = read_isnad_data(
        "nameData/names.csv",
        "communities/goldStandard_goldTags.json",
        "ryan_embeddings.json"
    )

    # check shapes are the same
    assert len(true_isnad_mention_ids) == len(isnad_mention_embeddings)
    assert len(true_isnad_mention_ids[0]) == len(isnad_mention_embeddings[0])
    assert len(true_isnad_mention_ids[-1]) == len(isnad_mention_embeddings[-1])

    # two similar but distinct surface forms of Sulayman b. Ishaq
    similarity = cosine_similarity(
        isnad_mention_embeddings[2356][1],
        isnad_mention_embeddings[2362][1]
    )
    assert similarity == pytest.approx(0.93, abs=0.01)

    # two completely different surface forms
    similarity = cosine_similarity(
        isnad_mention_embeddings[1][1],
        isnad_mention_embeddings[-1][0]
    )
    assert similarity == pytest.approx(0.56, abs=0.01)

    # two identical different surface forms
    similarity = cosine_similarity(
        isnad_mention_embeddings[0][5],
        isnad_mention_embeddings[3][5]
    )
    assert similarity == pytest.approx(0.97, abs=0.01)

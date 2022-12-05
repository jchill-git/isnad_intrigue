import pytest

from helpers.features import cosine_similarity

def test_cosine_similarity():
    assert cosine_similarity([0, 1, 0], [0, 0, 1]) == 0
    assert cosine_similarity([0, 1, 0], [0, 1, 0]) == 1
    assert cosine_similarity([0, 1, 1], [1, 0, 0]) == 0
    assert cosine_similarity([0, 1, 1], [1, 0, 1]) == pytest.approx(0.5)

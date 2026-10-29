from collections import Counter
from fractions import Fraction

import pytest

from tanque.gen import intents
from tanque.sql import guard


def test_catalog_has_the_v1_and_v2_intents():
    ids = [i.id for i in intents.INTENTS]
    # 40 da v1 (M4) + 17 da v2 (M15, D34)
    assert len(ids) == 57
    assert len(ids) == len(set(ids))
    assert all(i.level in intents.LEVELS for i in intents.INTENTS)


def test_every_level_has_intents_and_window_only_in_hard():
    for level in intents.LEVELS:
        assert intents.by_level(level)
    assert all(i.level == "dificil" for i in intents.INTENTS if i.uses_window)
    assert any(i.uses_window for i in intents.INTENTS)


def test_target_distribution_sums_to_one():
    assert sum(intents.TARGET_DISTRIBUTION.values()) == 1
    # v2 (M15): 28/36/36 (a v1 era 35/40/25)
    assert intents.TARGET_DISTRIBUTION == {
        "facil": Fraction(28, 100),
        "medio": Fraction(36, 100),
        "dificil": Fraction(36, 100),
    }


@pytest.mark.parametrize("intent", intents.INTENTS, ids=lambda i: i.id)
def test_sql_shape_passes_guard_and_matches_window_flag(intent):
    verdict = guard.check(intent.sql_shape)
    assert verdict.ok, verdict.reason
    assert ("OVER (" in intent.sql_shape) == intent.uses_window


@pytest.mark.parametrize("intent", intents.INTENTS, ids=lambda i: i.id)
def test_instructions_mention_shape_and_variations(intent):
    text = intent.instructions()
    assert intent.id in text
    assert intent.sql_shape.strip() in text
    assert all(v in text for v in intent.variations)


def test_pilot_allocation_covers_20_distinct_intents_in_target_split():
    chosen = intents.allocate(20)
    assert len(chosen) == 20
    assert len({i.id for i in chosen}) == 20
    assert Counter(i.level for i in chosen) == {"facil": 6, "medio": 7, "dificil": 7}


@pytest.mark.parametrize("n", [1, 7, 100, 480, 500])
def test_allocation_follows_target_within_one_request(n):
    chosen = intents.allocate(n)
    assert len(chosen) == n
    counts = Counter(i.level for i in chosen)
    for level, share in intents.TARGET_DISTRIBUTION.items():
        assert abs(counts[level] - float(share) * n) <= 1


def test_allocation_keeps_window_below_20_percent_of_hard():
    chosen = intents.allocate(500)
    hard = [i for i in chosen if i.level == "dificil"]
    window = [i for i in hard if i.uses_window]
    assert 0 < len(window) / len(hard) <= float(intents.MAX_WINDOW_SHARE_OF_HARD)


def test_allocation_is_deterministic_and_prefix_balanced():
    assert [i.id for i in intents.allocate(60)] == [i.id for i in intents.allocate(60)]
    first_ten = Counter(i.level for i in intents.allocate(10))
    assert set(first_ten) == set(intents.LEVELS)


def test_allocation_rejects_negative():
    with pytest.raises(ValueError):
        intents.allocate(-1)
    assert intents.allocate(0) == []

from memory_training.scripts.classify_patch_delta_disagreements import (
    classify_facts,
    parse_memory,
)


def _facts(value: str):
    return parse_memory(value)


GOLD = '''### Ada
- [2025-01-01 10:00] carcontrol_music_set_volume.volume; value=15; condition=when calling'''


def test_evidence_ignored_and_stale_value() -> None:
    gold = _facts(GOLD)
    assert classify_facts(gold, _facts(GOLD), ())[0] == "EVIDENCE_IGNORED"
    stale = '''### Ada
- [2024-12-01 10:00] carcontrol_music_set_volume.volume; value=20; condition=when calling'''
    assert classify_facts(gold, _facts(stale), ())[0] == "STALE_VALUE"


def test_wrong_owner_and_condition_loss() -> None:
    gold = _facts(GOLD)
    wrong = GOLD.replace("### Ada", "### Bob")
    assert classify_facts(gold, _facts(wrong), ())[0] == "WRONG_OWNER"
    changed = GOLD.replace("when calling", "during meetings")
    assert classify_facts(gold, _facts(changed), ())[0] == "CONDITION_LOSS"


def test_conflicting_facts() -> None:
    predicted = GOLD + '''
- [2025-02-01 10:00] carcontrol_music_set_volume.volume; value=20; condition=when calling'''
    assert classify_facts(_facts(GOLD), _facts(predicted), ())[0] == "CONFLICTING_FACTS"

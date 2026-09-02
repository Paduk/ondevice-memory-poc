from memory_training.methods.operations import apply_operations
from memory_training.prepare_v1_patch_augmentation import reconstruct_operations


def _check(previous: str, next_memory: str) -> None:
    operations, _ = reconstruct_operations(previous, next_memory)
    if previous == next_memory:
        assert operations == []
    else:
        replayed, _ = apply_operations(previous, operations)
        assert replayed == next_memory


def test_reconstruct_operations_replays_common_changes() -> None:
    _check("", "### Gary\n- Likes green.")
    _check("### Gary\n- Likes green.", "### Gary\n- Likes green.")
    _check("### Gary\n- Likes green.", "### Gary\n- Likes blue.")
    _check(
        "### Gary\n- Likes green.",
        "### Gary\n- Likes green.\n\n### Jane\n- Uses quiet mode.",
    )
    _check(
        "### Gary\n- Likes green.\n- Uses sport mode.\n\n### Jane\n- Uses quiet mode.",
        "### Gary\n- Likes blue.\n\n### Jane\n- Uses quiet mode.\n- Seat is 2.",
    )
    _check(
        "### Gary\n- Likes green.\n\n### Jane\n- Uses quiet mode.",
        "### Gary\n- Likes green.\n- Seat is 2.\n\n### Jane\n- Uses quiet mode.",
    )
    _check("### Gary\n- Likes green.", "")

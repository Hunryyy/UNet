"""Verify Task 1 uses a strict 2D spatial block split for validation."""
from _task1_common import choose_val_block, expected_positions, expected_split_positions


def main():
    val_block = choose_val_block()
    expected_train, expected_val = expected_split_positions()
    total = len(expected_positions())

    assert val_block["rows"], "Validation block rows should not be empty"
    assert val_block["cols"], "Validation block cols should not be empty"
    assert len(expected_val) == val_block["tile_count"]
    assert len(expected_train) + len(expected_val) == total
    assert len(val_block["cols"]) < len({col for _, col, _, _ in expected_positions()}), (
        "Validation split should be a spatial sub-block, not full-width row stripes."
    )

    print("=" * 60)
    print("TEST PASSED — validation set is a strict spatial block")
    print(f"rows={val_block['row_range']} cols={val_block['col_range']} tiles={val_block['tile_count']}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()

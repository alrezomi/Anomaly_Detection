"""Shared experiment inputs for the vision and time-series pipelines."""

from __future__ import annotations

import os
from pathlib import Path


# ============================================================
# EDIT THIS SECTION TO SELECT AN EXPERIMENT
# ============================================================

# Supported values reflect the names currently present in the dataset.
TASK_NAME = "pick_place"

# The same item IDs are used for both videos and force/torque bags.
NOMINAL_ITEM_IDS = (6, 7)
TEST_ITEM_ID = 11

# A video covers an item-level execution, while the force data are split into
# numbered repetitions. Select which repetitions the time-series pipeline uses.
NOMINAL_BAG_REPETITIONS = (1, 2, 3, 4, 5)
TEST_BAG_REPETITION = 1


# ============================================================
# DATASET LAYOUT AND MODALITY-SPECIFIC NAME MAPPING
# ============================================================

SCRIPT_VS_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = SCRIPT_VS_DIRECTORY.parent
DATA_DIRECTORY = Path(
    os.environ.get("ANOMALY_DATA_ROOT", PROJECT_DIRECTORY / "data")
).resolve()

VIDEO_DIRECTORY = (
    DATA_DIRECTORY
    / "Exp_1"
    / "Exp_1"
    / "Exp_1_Videos_u_Fotos"
)
BAG_DIRECTORY = (
    DATA_DIRECTORY
    / "Exp_1"
    / "Exp_1"
    / "Exp_1_Force_log_files"
    / "Bags"
)
OUTPUT_DIRECTORY = Path(
    os.environ.get("ANOMALY_OUTPUT_ROOT", SCRIPT_VS_DIRECTORY / "outputs")
).resolve()

TASK_NAME_MAP = {
    "pick_place": {
        "video_suffix": "pick_place",
        "bag_suffix": "pp",
    },
    "open": {
        "video_suffix": "klemme_open",
        "bag_suffix": "open",
    },
}


def _task_names() -> tuple[str, str]:
    """Return the modality-specific filename suffixes for the selected task."""
    if TASK_NAME not in TASK_NAME_MAP:
        supported = ", ".join(sorted(TASK_NAME_MAP))
        raise ValueError(
            f"Unsupported TASK_NAME {TASK_NAME!r}. Supported tasks: {supported}."
        )

    mapping = TASK_NAME_MAP[TASK_NAME]
    return mapping["video_suffix"], mapping["bag_suffix"]


def video_path(item_id: int) -> Path:
    """Build the video path for one experiment item."""
    video_suffix, _ = _task_names()
    return VIDEO_DIRECTORY / f"i{item_id}_{video_suffix}.MP4"


def bag_path(item_id: int, repetition: int) -> Path:
    """Build the MCAP path for one experiment item and repetition."""
    if repetition < 1:
        raise ValueError("Bag repetition numbers must be at least 1.")

    _, bag_suffix = _task_names()
    run_name = f"exp1_i{item_id}_{bag_suffix}_{repetition}"
    return BAG_DIRECTORY / run_name / f"{run_name}_0.mcap"


def nominal_video_paths() -> list[Path]:
    return [video_path(item_id) for item_id in NOMINAL_ITEM_IDS]


def test_video_path() -> Path:
    return video_path(TEST_ITEM_ID)


def nominal_bag_paths() -> list[Path]:
    return [
        bag_path(item_id, repetition)
        for item_id in NOMINAL_ITEM_IDS
        for repetition in NOMINAL_BAG_REPETITIONS
    ]


def test_bag_path() -> Path:
    return bag_path(TEST_ITEM_ID, TEST_BAG_REPETITION)


def print_experiment_summary() -> None:
    """Print the shared selection before a pipeline starts."""
    print("Shared experiment selection:")
    print(f"  task: {TASK_NAME}")
    print(f"  nominal item IDs: {list(NOMINAL_ITEM_IDS)}")
    print(f"  test item ID: {TEST_ITEM_ID}")
    print(
        "  nominal bag repetitions: "
        f"{list(NOMINAL_BAG_REPETITIONS)}"
    )
    print(f"  test bag repetition: {TEST_BAG_REPETITION}")

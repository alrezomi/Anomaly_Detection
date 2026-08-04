"""Run the complete GMR time-series anomaly-detection pipeline."""

from pathlib import Path

import matplotlib.pyplot as plt

from experiment_config import (
    BAG_DIRECTORY,
    OUTPUT_DIRECTORY,
    PROJECT_DIRECTORY,
    nominal_bag_paths,
    print_experiment_summary,
    test_bag_path,
)
from .anomaly_detection import GMRAnomalyDetector
from .threshold import (
    apply_adaptive_gmr_threshold,
    calibrate_gmr_leave_one_run_out,
)
from .ts_data import (
    SIGNAL_COLUMNS,
    read_wrench_bag,
    sync_and_preprocess_ts_dataframe,
)
from .visualization import (
    plot_force_synchronization,
    plot_gmr_calibration,
    plot_gmr_epsilon_filtered,
    plot_gmr_mahalanobis_with_threshold,
    plot_raw_signals,
)


# ============================================================
# 1. INPUT BAGS
# Both modalities derive these paths from the shared experiment selection in
# experiment_config.py.
# ============================================================

NOMINAL_BAG_PATHS = nominal_bag_paths()
TEST_BAG_PATH = test_bag_path()


# ============================================================
# 2. BAG, SYNCHRONIZATION, AND PREPROCESSING PARAMETERS
# ============================================================

TOPIC_NAME = "/tcp_force"
RESAMPLE_LENGTH = 500

BASELINE_FRACTION = 0.10
THRESHOLD_STD_FACTOR = 4.0
MIN_FORCE_CHANGE = 0.5
PADDING_SECONDS = 0.5

INPUT_COLUMNS = ["phase"]
OUTPUT_COLUMNS = SIGNAL_COLUMNS.copy()


# ============================================================
# 3. GMR PARAMETERS
# ============================================================

N_COMPONENTS = 3
COMPONENT_THRESHOLD_MARGIN = 1.10
RANDOM_STATE = 42


# ============================================================
# 4. ADAPTIVE THRESHOLD AND FILTERING PARAMETERS
# ============================================================

THRESHOLD_BINS = 50
THRESHOLD_QUANTILE = 0.999
THRESHOLD_MARGIN = 1.60
MIN_POINTS_PER_BIN = 5
THRESHOLD_FLOOR_QUANTILE = 0.80
THRESHOLD_SMOOTH_WINDOW = 5

SCORE_SMOOTH_WINDOW = 11
MAJORITY_WINDOW = 8
MAJORITY_VOTES = 6


# ============================================================
# 5. OUTPUT PATHS
# ============================================================

THRESHOLD_CSV = OUTPUT_DIRECTORY / "gmr_phase_threshold.csv"
CALIBRATION_CSV = OUTPUT_DIRECTORY / "gmr_calibration_scores.csv"
TEST_RESULT_CSV = OUTPUT_DIRECTORY / "gmr_test_results.csv"


def check_input_paths() -> None:
    """Stop early when one of the bag paths is incorrect."""
    missing_paths = [
        path
        for path in [*NOMINAL_BAG_PATHS, TEST_BAG_PATH]
        if not Path(path).exists()
    ]

    if missing_paths:
        missing_text = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "The following bag paths were not found:\n"
            f"{missing_text}\n\n"
            f"Expected bag directory:\n  {BAG_DIRECTORY}\n\n"
            "Add the bags there or update BAG_DIRECTORY in "
            "run_time_series_test.py."
        )


def load_synchronized_execution(
    bag_path: str | Path,
) -> tuple:
    """Read, synchronize, and resample one force/torque execution."""
    raw_dataframe = read_wrench_bag(
        bag_path=bag_path,
        topic=TOPIC_NAME,
    )
    return sync_and_preprocess_ts_dataframe(
        raw_dataframe=raw_dataframe,
        resample_length=RESAMPLE_LENGTH,
        baseline_fraction=BASELINE_FRACTION,
        threshold_std_factor=THRESHOLD_STD_FACTOR,
        min_force_change=MIN_FORCE_CHANGE,
        padding_seconds=PADDING_SECONDS,
    )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    check_input_paths()

    print_experiment_summary()
    print(f"Project directory: {PROJECT_DIRECTORY}")
    print(f"Bag directory: {BAG_DIRECTORY}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")

    # --------------------------------------------------------
    # Step 1: Load and synchronize nominal executions
    # --------------------------------------------------------
    nominal_dataframes = []
    first_nominal_debug = None
    first_nominal_sync_info = None

    for run_index, bag_path in enumerate(NOMINAL_BAG_PATHS):
        print("\n" + "=" * 90)
        print(f"Loading nominal run {run_index}:")
        print(bag_path)
        print("=" * 90)

        processed, sync_info, debug_dataframe = (
            load_synchronized_execution(bag_path)
        )
        nominal_dataframes.append(processed)

        print("Processed shape:", processed.shape)
        print("Synchronization:", sync_info)

        if run_index == 0:
            first_nominal_debug = debug_dataframe
            first_nominal_sync_info = sync_info

    print("\nLoaded nominal executions:", len(nominal_dataframes))

    if first_nominal_debug is not None and first_nominal_sync_info is not None:
        plot_force_synchronization(
            debug_dataframe=first_nominal_debug,
            sync_info=first_nominal_sync_info,
            title="Force-based synchronization of the first nominal run",
        )

    plot_raw_signals(
        dataframe=nominal_dataframes[0],
        signal_columns=OUTPUT_COLUMNS,
        title="Example synchronized nominal force/torque signals",
    )

    # --------------------------------------------------------
    # Step 2: Calibrate the adaptive threshold
    # --------------------------------------------------------
    threshold_dataframe, calibration_scores = (
        calibrate_gmr_leave_one_run_out(
            nominal_dataframes=nominal_dataframes,
            input_columns=INPUT_COLUMNS,
            output_columns=OUTPUT_COLUMNS,
            n_components=N_COMPONENTS,
            threshold_margin=COMPONENT_THRESHOLD_MARGIN,
            random_state=RANDOM_STATE,
            n_bins=THRESHOLD_BINS,
            quantile=THRESHOLD_QUANTILE,
            margin=THRESHOLD_MARGIN,
            min_points_per_bin=MIN_POINTS_PER_BIN,
            threshold_floor_quantile=THRESHOLD_FLOOR_QUANTILE,
            smooth_threshold_window=THRESHOLD_SMOOTH_WINDOW,
        )
    )

    threshold_dataframe.to_csv(THRESHOLD_CSV, index=False)
    calibration_scores.to_csv(CALIBRATION_CSV, index=False)

    print("\nFirst adaptive-threshold rows:")
    print(threshold_dataframe.head())

    plot_gmr_calibration(
        threshold_dataframe=threshold_dataframe,
        calibration_scores=calibration_scores,
        title="GMR calibration from held-out nominal executions",
    )

    # --------------------------------------------------------
    # Step 3: Fit the final GMR on all synchronized nominal runs
    # --------------------------------------------------------
    detector = GMRAnomalyDetector(
        input_columns=INPUT_COLUMNS,
        output_columns=OUTPUT_COLUMNS,
        n_components=N_COMPONENTS,
        threshold_margin=COMPONENT_THRESHOLD_MARGIN,
        random_state=RANDOM_STATE,
    )
    detector.fit(nominal_dataframes)

    # --------------------------------------------------------
    # Step 4: Load, synchronize, and score the test execution
    # --------------------------------------------------------
    print("\n" + "=" * 90)
    print("Loading test bag:")
    print(TEST_BAG_PATH)
    print("=" * 90)

    test_dataframe, test_sync_info, test_debug_dataframe = (
        load_synchronized_execution(TEST_BAG_PATH)
    )

    print("Processed test shape:", test_dataframe.shape)
    print("Test synchronization:", test_sync_info)

    plot_force_synchronization(
        debug_dataframe=test_debug_dataframe,
        sync_info=test_sync_info,
        title="Force-based synchronization of the test run",
    )
    plot_raw_signals(
        dataframe=test_dataframe,
        signal_columns=OUTPUT_COLUMNS,
        title="Synchronized test force/torque signals",
    )

    raw_result = detector.score_dataframe(test_dataframe)
    test_result = apply_adaptive_gmr_threshold(
        result_dataframe=raw_result,
        threshold_dataframe=threshold_dataframe,
        score_smooth_window=SCORE_SMOOTH_WINDOW,
        majority_window=MAJORITY_WINDOW,
        majority_votes=MAJORITY_VOTES,
    )
    test_result.to_csv(TEST_RESULT_CSV, index=False)

    plot_gmr_mahalanobis_with_threshold(
        result_dataframe=test_result,
        title="GMR Mahalanobis distance and adaptive threshold",
    )
    plot_gmr_epsilon_filtered(
        result_dataframe=test_result,
        title="Filtered GMR anomaly score on the test bag",
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    print("\n" + "=" * 90)
    print("TIME-SERIES TEST FINISHED")
    print("=" * 90)
    print("Processed samples:", len(test_result))
    print("Raw alarm samples:", int(test_result["alarm_raw"].sum()))
    print(
        "Filtered alarm samples:",
        int(test_result["alarm_filtered"].sum()),
    )
    print(
        "Filtered alarm ratio:",
        float(test_result["alarm_filtered"].mean()),
    )
    print("Maximum smoothed epsilon:", float(test_result["epsilon_smooth"].max()))
    print(f"Threshold CSV: {THRESHOLD_CSV.resolve()}")
    print(f"Calibration CSV: {CALIBRATION_CSV.resolve()}")
    print(f"Test-result CSV: {TEST_RESULT_CSV.resolve()}")

    print("\nProcessing finished. Close the graph windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()

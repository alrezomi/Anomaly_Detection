"""Run the complete DINOv2 vision anomaly-detection pipeline."""

from pathlib import Path

from anomaly_detection import (
    build_nominal_video_memory,
    run_video_test,
)
from dinov2_features import DINOv2FeatureExtractor
from threshold import (
    calibrate_adaptive_threshold_leave_one_video_out,
)
from visualization import (
    create_anomaly_heatmap_video,
    plot_adaptive_calibration,
    plot_test_epsilon,
    plot_test_scores,
)


# ============================================================
# 1. INPUT VIDEOS
# Edit only these paths when you change the experiment.
# ============================================================

NOMINAL_VIDEO_PATHS = [
    r"C:/Users/ali76665/Desktop/AD/data/Exp_1/Exp_1/Exp_1_Videos_u_Fotos/i11_pick_place.MP4",
]

TEST_VIDEO_PATH = (
    r"C:/Users/ali76665/Desktop/AD/data/Exp_1/Exp_1/Exp_1_Videos_u_Fotos/i11_pick_place.MP4"
)


# ============================================================
# 2. DINOv2 AND VIDEO PARAMETERS
# These are the same main settings used in the notebook.
# ============================================================

MODEL_NAME = "facebook/dinov2-small"
DINO_INPUT_SIZE = 518

SAMPLE_FPS = 2.0
START_SEC = None
END_SEC = None
MAX_NOMINAL_FRAMES_PER_VIDEO = None
MAX_TEST_FRAMES = None

TOP_K_CLS = 5
ATTENTION_POWER = 1.5
MIN_WEIGHT = 0.15


# ============================================================
# 3. ADAPTIVE THRESHOLD PARAMETERS
# ============================================================

THRESHOLD_QUANTILE = 0.95
THRESHOLD_MARGIN = 1.11
THRESHOLD_BINS = 30
MIN_POINTS_PER_BIN = 3
THRESHOLD_SMOOTH_WINDOW = 3
THRESHOLD_FLOOR_QUANTILE = 0.20


# ============================================================
# 4. OUTPUT PATHS
# ============================================================

OUTPUT_DIRECTORY = Path("outputs")
THRESHOLD_CSV = OUTPUT_DIRECTORY / "adaptive_visual_threshold.csv"
CALIBRATION_CSV = OUTPUT_DIRECTORY / "adaptive_visual_calibration.csv"
TEST_RESULT_CSV = OUTPUT_DIRECTORY / "vision_test_results.csv"
OUTPUT_VIDEO = (
    OUTPUT_DIRECTORY / "adaptive_cls_attention_heatmap_output.mp4"
)


def check_input_paths() -> None:
    """Stop early when one of the input-video paths is incorrect."""
    missing_paths = [
        path
        for path in [*NOMINAL_VIDEO_PATHS, TEST_VIDEO_PATH]
        if not Path(path).is_file()
    ]

    if missing_paths:
        missing_text = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "The following video files were not found:\n"
            f"{missing_text}\n\n"
            "Correct the paths at the top of run_vision_test.py."
        )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    check_input_paths()

    # --------------------------------------------------------
    # Step 1: Load DINOv2 once
    # --------------------------------------------------------
    extractor = DINOv2FeatureExtractor(
        model_name=MODEL_NAME,
        input_size=DINO_INPUT_SIZE,
    )

    # --------------------------------------------------------
    # Step 2: Build nominal memory
    # --------------------------------------------------------
    nominal_memory, grid_size = build_nominal_video_memory(
        extractor=extractor,
        nominal_video_paths=NOMINAL_VIDEO_PATHS,
        sample_fps=SAMPLE_FPS,
        start_sec=START_SEC,
        end_sec=END_SEC,
        max_frames_per_video=MAX_NOMINAL_FRAMES_PER_VIDEO,
    )

    print("Nominal grid size:", grid_size)

    # --------------------------------------------------------
    # Step 3: Calibrate the adaptive threshold
    # --------------------------------------------------------
    (
        adaptive_threshold_df,
        calibration_df,
    ) = calibrate_adaptive_threshold_leave_one_video_out(
        nominal_memory=nominal_memory,
        quantile=THRESHOLD_QUANTILE,
        margin=THRESHOLD_MARGIN,
        top_k_cls=TOP_K_CLS,
        attention_power=ATTENTION_POWER,
        min_weight=MIN_WEIGHT,
        n_bins=THRESHOLD_BINS,
        min_points_per_bin=MIN_POINTS_PER_BIN,
        smooth_window=THRESHOLD_SMOOTH_WINDOW,
        threshold_floor_quantile=THRESHOLD_FLOOR_QUANTILE,
    )

    adaptive_threshold_df.to_csv(THRESHOLD_CSV, index=False)
    calibration_df.to_csv(CALIBRATION_CSV, index=False)

    print("\nFirst adaptive-threshold rows:")
    print(adaptive_threshold_df.head())

    plot_adaptive_calibration(
        adaptive_threshold_df=adaptive_threshold_df,
        calibration_df=calibration_df,
    )

    # --------------------------------------------------------
    # Step 4: Test the selected video
    # --------------------------------------------------------
    test_result_df = run_video_test(
        extractor=extractor,
        test_video_path=TEST_VIDEO_PATH,
        nominal_memory=nominal_memory,
        adaptive_threshold_df=adaptive_threshold_df,
        sample_fps=SAMPLE_FPS,
        start_sec=START_SEC,
        end_sec=END_SEC,
        max_frames=MAX_TEST_FRAMES,
        top_k_cls=TOP_K_CLS,
        attention_power=ATTENTION_POWER,
        min_weight=MIN_WEIGHT,
    )

    test_result_df.to_csv(TEST_RESULT_CSV, index=False)

    plot_test_scores(test_result_df)
    plot_test_epsilon(test_result_df)

    # --------------------------------------------------------
    # Step 5: Create the annotated heatmap video
    # --------------------------------------------------------
    create_anomaly_heatmap_video(
        extractor=extractor,
        test_video_path=TEST_VIDEO_PATH,
        nominal_memory=nominal_memory,
        output_video_path=OUTPUT_VIDEO,
        adaptive_threshold_df=adaptive_threshold_df,
        sample_fps=SAMPLE_FPS,
        start_sec=START_SEC,
        end_sec=END_SEC,
        max_frames=MAX_TEST_FRAMES,
        top_k_cls=TOP_K_CLS,
        attention_power=ATTENTION_POWER,
        min_weight=MIN_WEIGHT,
        alpha=0.45,
        visualization_mode="original",
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    alarm_count = int(test_result_df["alarm"].sum())
    frame_count = len(test_result_df)

    print("\n" + "=" * 90)
    print("VISION TEST FINISHED")
    print("=" * 90)
    print(f"Processed frames: {frame_count}")
    print(f"Alarm frames: {alarm_count}")
    print(f"Maximum epsilon: {test_result_df['epsilon'].max():.3f}")
    print(f"Threshold CSV: {THRESHOLD_CSV.resolve()}")
    print(f"Calibration CSV: {CALIBRATION_CSV.resolve()}")
    print(f"Test-result CSV: {TEST_RESULT_CSV.resolve()}")
    print(f"Heatmap video: {OUTPUT_VIDEO.resolve()}")


if __name__ == "__main__":
    main()

"""Run the complete DINOv2 vision anomaly-detection pipeline."""

from pathlib import Path
import json
from rosbag_io import RosbagImageSource

from experiment_config import (
    OUTPUT_DIRECTORY,
    PROJECT_DIRECTORY,
    VIDEO_DIRECTORY,
    nominal_video_paths,
    print_experiment_summary,
    test_video_path,
)
from .anomaly_detection import (
    build_nominal_video_memory,
    run_video_test,
)
from .dinov2_features import DINOv2FeatureExtractor
from .nominal_cache import load_nominal_cache, save_nominal_cache
from .threshold import (
    calibrate_adaptive_threshold_leave_one_video_out,
)
from .visualization import (
    create_anomaly_heatmap_video,
    plot_adaptive_calibration,
    plot_test_epsilon,
    plot_test_scores,
)


# ============================================================
# 1. INPUT VIDEOS
# Both modalities derive these paths from the shared experiment selection in
# experiment_config.py.
# ============================================================

NOMINAL_VIDEO_PATHS = nominal_video_paths()
TEST_VIDEO_PATH = test_video_path()


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

THRESHOLD_CSV = OUTPUT_DIRECTORY / "adaptive_visual_threshold.csv"
CALIBRATION_CSV = OUTPUT_DIRECTORY / "adaptive_visual_calibration.csv"
TEST_RESULT_CSV = OUTPUT_DIRECTORY / "vision_test_results.csv"
OUTPUT_VIDEO = (
    OUTPUT_DIRECTORY / "adaptive_cls_attention_heatmap_output.mp4"
)
CALIBRATION_PLOT: Path | None = None
TEST_SCORE_PLOT: Path | None = None
TEST_EPSILON_PLOT: Path | None = None
SETTINGS_JSON: Path | None = None
NOMINAL_CACHE_DIRECTORY: Path | None = None
REBUILD_NOMINAL_CACHE = False


def nominal_cache_signature() -> dict:
    """Describe every setting that changes reusable nominal results."""
    sources = []
    for source in NOMINAL_VIDEO_PATHS:
        if isinstance(source, RosbagImageSource):
            sources.append(
                {
                    "bag_path": str(source.bag_path),
                    "topic": source.topic,
                    "stage_topic": source.stage_topic,
                    "stage_count": source.stage_count,
                }
            )
        else:
            sources.append({"video_path": str(Path(source).resolve())})
    return {
        "format_version": 1,
        "sources": sources,
        "model_name": MODEL_NAME,
        "dino_input_size": DINO_INPUT_SIZE,
        "sample_fps": SAMPLE_FPS,
        "start_sec": START_SEC,
        "end_sec": END_SEC,
        "max_nominal_frames_per_video": MAX_NOMINAL_FRAMES_PER_VIDEO,
        "top_k_cls": TOP_K_CLS,
        "attention_power": ATTENTION_POWER,
        "min_weight": MIN_WEIGHT,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "threshold_margin": THRESHOLD_MARGIN,
        "threshold_bins": THRESHOLD_BINS,
        "min_points_per_bin": MIN_POINTS_PER_BIN,
        "threshold_smooth_window": THRESHOLD_SMOOTH_WINDOW,
        "threshold_floor_quantile": THRESHOLD_FLOOR_QUANTILE,
    }


def nominal_cache_runtime_signature() -> dict:
    """Settings a test must match, excluding the nominal source list."""
    signature = nominal_cache_signature()
    signature.pop("sources", None)
    return signature


def check_input_paths(include_nominal: bool = True, include_test: bool = True) -> None:
    """Stop early when one of the configured vision sources is incorrect."""
    selected_paths = []
    if include_nominal:
        selected_paths.extend(NOMINAL_VIDEO_PATHS)
    if include_test:
        selected_paths.append(TEST_VIDEO_PATH)
    missing_paths = [
        path
        for path in selected_paths
        if not (
            path.bag_path.exists()
            if isinstance(path, RosbagImageSource)
            else Path(path).is_file()
        )
    ]

    if missing_paths:
        missing_text = "\n".join(
            (
                f"  - bag={path.bag_path}, topic={path.topic}"
                if isinstance(path, RosbagImageSource)
                else f"  - video={path}"
            )
            for path in missing_paths
        )
        raise FileNotFoundError(
            "The following vision sources were not found:\n"
            f"{missing_text}\n\n"
            "For rosbag inputs, verify the directory name under /data and "
            "ensure it contains metadata.yaml. For legacy video inputs, "
            "verify experiment_config.py."
        )


def main(mode: str = "all") -> None:
    if mode not in {"all", "build-memory", "test"}:
        raise ValueError("mode must be 'all', 'build-memory', or 'test'")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    check_input_paths(include_nominal=mode != "test", include_test=mode != "build-memory")

    using_rosbag = isinstance(TEST_VIDEO_PATH, RosbagImageSource)
    if not using_rosbag:
        print_experiment_summary()
    print(f"Project directory: {PROJECT_DIRECTORY}")
    if mode == "build-memory":
        print("Vision mode: build named nominal memory only")
        print(f"Nominal sources: {len(NOMINAL_VIDEO_PATHS)}")
    elif using_rosbag:
        print("Vision input mode: ROS 2 bag image topic")
        print(f"Nominal sources: {len(NOMINAL_VIDEO_PATHS)}")
        print(f"Test source: {TEST_VIDEO_PATH.bag_path}")
        print(f"Camera topic: {TEST_VIDEO_PATH.topic}")
    else:
        print(f"Video directory: {VIDEO_DIRECTORY}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")
    if SETTINGS_JSON is not None:
        SETTINGS_JSON.write_text(
            json.dumps(
                {
                    **nominal_cache_signature(),
                    "mode": mode,
                    "test_source": str(TEST_VIDEO_PATH),
                    "nominal_cache_directory": str(NOMINAL_CACHE_DIRECTORY),
                }, indent=2, sort_keys=True
            ), encoding="utf-8"
        )

    # --------------------------------------------------------
    # Step 1: Load DINOv2 once
    # --------------------------------------------------------
    extractor = DINOv2FeatureExtractor(
        model_name=MODEL_NAME,
        input_size=DINO_INPUT_SIZE,
    )

    signature = nominal_cache_signature()
    cached = None
    if NOMINAL_CACHE_DIRECTORY is not None and mode == "test":
        cached = load_nominal_cache(
            NOMINAL_CACHE_DIRECTORY, nominal_cache_runtime_signature()
        )
        if cached is None:
            raise FileNotFoundError(
                f"No compatible named nominal memory exists at {NOMINAL_CACHE_DIRECTORY}. "
                "Run the build-memory command first. Test mode never rebuilds memory."
            )
    elif NOMINAL_CACHE_DIRECTORY is not None and not REBUILD_NOMINAL_CACHE:
        cached = load_nominal_cache(NOMINAL_CACHE_DIRECTORY, signature)

    if cached is not None:
        nominal_memory, grid_size, adaptive_threshold_df, calibration_df = cached
    else:
        # ----------------------------------------------------
        # Step 2: Build nominal memory
        # ----------------------------------------------------
        nominal_memory, grid_size = build_nominal_video_memory(
            extractor=extractor,
            nominal_video_paths=NOMINAL_VIDEO_PATHS,
            sample_fps=SAMPLE_FPS,
            start_sec=START_SEC,
            end_sec=END_SEC,
            max_frames_per_video=MAX_NOMINAL_FRAMES_PER_VIDEO,
        )

        # ----------------------------------------------------
        # Step 3: Calibrate the adaptive threshold
        # ----------------------------------------------------
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
        if NOMINAL_CACHE_DIRECTORY is not None:
            save_nominal_cache(
                NOMINAL_CACHE_DIRECTORY,
                nominal_memory,
                grid_size,
                adaptive_threshold_df,
                calibration_df,
                signature,
            )
            print(f"Saved reusable nominal cache: {NOMINAL_CACHE_DIRECTORY}")

    print("Nominal grid size:", grid_size)

    if mode == "build-memory":
        plot_adaptive_calibration(
            adaptive_threshold_df, calibration_df,
            output_path=CALIBRATION_PLOT,
        )
        print("\nDINO NOMINAL MEMORY BUILD FINISHED")
        print(f"Memory directory: {NOMINAL_CACHE_DIRECTORY}")
        return

    adaptive_threshold_df.to_csv(THRESHOLD_CSV, index=False)
    calibration_df.to_csv(CALIBRATION_CSV, index=False)

    print("\nFirst adaptive-threshold rows:")
    print(adaptive_threshold_df.head())

    plot_adaptive_calibration(
        adaptive_threshold_df=adaptive_threshold_df,
        calibration_df=calibration_df,
        output_path=CALIBRATION_PLOT,
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

    plot_test_scores(test_result_df, output_path=TEST_SCORE_PLOT)
    plot_test_epsilon(test_result_df, output_path=TEST_EPSILON_PLOT)

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

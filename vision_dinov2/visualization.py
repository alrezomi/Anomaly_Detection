"""Plots and heatmap-video generation for DINOv2 anomaly detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from anomaly_detection import (
    NominalItem,
    add_progress_to_nominal_memory,
    compare_frame_to_nominal_memory,
    threshold_for_progress,
)
from dinov2_features import DINOv2FeatureExtractor, iter_video_frames


def show_attention_map(
    image: Image.Image,
    attention_scores: np.ndarray,
    grid_size: int,
    title: str = "DINOv2 attention map",
    alpha: float = 0.45,
) -> None:
    """Display CLS-to-patch attention over one image."""
    heatmap = attention_scores.reshape(grid_size, grid_size)

    plt.figure(figsize=(7, 5))
    plt.imshow(image)
    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=alpha,
        extent=(0, image.width, image.height, 0),
    )
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def show_anomaly_heatmap(
    image: Image.Image,
    weighted_patch_scores: np.ndarray,
    grid_size: int,
    title: str = "DINOv2 anomaly heatmap",
    alpha: float = 0.45,
) -> None:
    """Display weighted local anomaly scores over one image."""
    heatmap = weighted_patch_scores.reshape(grid_size, grid_size)

    plt.figure(figsize=(7, 5))
    plt.imshow(image)
    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=alpha,
        extent=(0, image.width, image.height, 0),
    )
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def make_heatmap_overlay(
    frame: Image.Image,
    weighted_patch_scores: np.ndarray,
    grid_size: int,
    alpha: float = 0.45,
) -> np.ndarray:
    """Create an RGB heatmap overlay for one frame."""
    frame_rgb = np.asarray(frame.convert("RGB"))
    height, width = frame_rgb.shape[:2]

    heatmap = weighted_patch_scores.reshape(grid_size, grid_size)
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)

    heatmap_resized = cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )
    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return cv2.addWeighted(
        frame_rgb,
        1.0 - alpha,
        heatmap_rgb,
        alpha,
        0.0,
    )


def plot_adaptive_calibration(
    adaptive_threshold_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    title: str = "CLS-stage adaptive visual threshold calibration",
) -> None:
    """Plot held-out nominal scores and their adaptive threshold."""
    plt.figure(figsize=(12, 4.5))

    plt.scatter(
        calibration_df["matched_progress"],
        calibration_df["score"],
        s=12,
        alpha=0.35,
        label="Held-out nominal scores",
    )
    plt.plot(
        adaptive_threshold_df["stage_center"],
        adaptive_threshold_df["adaptive_threshold"],
        linewidth=2.5,
        label="Adaptive threshold",
    )

    plt.xlabel("Matched nominal progress / visual stage")
    plt.ylabel("Attention-weighted patch anomaly score")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def plot_test_scores(
    result_df: pd.DataFrame,
    title: str = "CLS-assisted visual anomaly score",
) -> None:
    """Plot raw anomaly score and threshold over video time."""
    plt.figure(figsize=(12, 4.5))

    plt.plot(
        result_df["timestamp_sec"],
        result_df["score"],
        marker="o",
        linewidth=2,
        label="Visual anomaly score",
    )

    if result_df["threshold"].notna().any():
        plt.plot(
            result_df["timestamp_sec"],
            result_df["threshold"],
            linewidth=2.5,
            label="Threshold",
        )

    alarm_rows = result_df[result_df["alarm"]]

    if not alarm_rows.empty:
        plt.scatter(
            alarm_rows["timestamp_sec"],
            alarm_rows["score"],
            marker="x",
            s=90,
            label="Alarm",
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def plot_test_epsilon(
    result_df: pd.DataFrame,
    title: str = "Normalized adaptive visual anomaly score",
) -> None:
    """Plot score divided by threshold; values above one are alarms."""
    if result_df["epsilon"].isna().all():
        print("No epsilon plot was created because no threshold was supplied.")
        return

    plt.figure(figsize=(12, 4.5))

    plt.plot(
        result_df["timestamp_sec"],
        result_df["epsilon"],
        marker="o",
        linewidth=2,
        label="Normalized visual anomaly score",
    )
    plt.axhline(
        1.0,
        linewidth=2.5,
        label="Threshold = 1",
    )

    alarm_rows = result_df[result_df["alarm"]]

    if not alarm_rows.empty:
        plt.scatter(
            alarm_rows["timestamp_sec"],
            alarm_rows["epsilon"],
            marker="x",
            s=90,
            label="Alarm",
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Normalized score")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def _draw_text_panel(
    overlay_bgr: np.ndarray,
    timestamp_sec: float,
    score: float,
    threshold: float | None,
    epsilon: float,
    alarm: bool,
    cls_distance: float,
    matched_progress: float,
    matched_video_id: int,
    matched_frame_id: int,
    threshold_mode: str,
) -> np.ndarray:
    """Draw monitoring information on one BGR frame."""
    height, width = overlay_bgr.shape[:2]
    panel_height = min(165, height)

    cv2.rectangle(
        overlay_bgr,
        (0, 0),
        (width, panel_height),
        (0, 0, 0),
        -1,
    )

    threshold_text = (
        "None" if threshold is None else f"{threshold:.5f}"
    )
    epsilon_text = (
        "None" if np.isnan(epsilon) else f"{epsilon:.2f}"
    )

    lines = [
        f"t={timestamp_sec:.2f}s | score={score:.5f}",
        f"{threshold_mode} threshold={threshold_text} | eps={epsilon_text}",
        (
            f"CLS dist={cls_distance:.5f} | "
            f"matched progress={matched_progress:.2f}"
        ),
        (
            f"match=nominal video {matched_video_id}, "
            f"frame {matched_frame_id}"
        ),
    ]

    for line_index, text in enumerate(lines):
        cv2.putText(
            overlay_bgr,
            text,
            (20, 30 + line_index * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    status_text = "ALARM" if alarm else "normal"
    status_color = (0, 0, 255) if alarm else (0, 255, 0)

    cv2.putText(
        overlay_bgr,
        status_text,
        (max(20, width - 180), 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2,
        cv2.LINE_AA,
    )

    return overlay_bgr


def create_anomaly_heatmap_video(
    extractor: DINOv2FeatureExtractor,
    test_video_path: str | Path,
    nominal_memory: list[NominalItem],
    output_video_path: str | Path,
    adaptive_threshold_df: pd.DataFrame | None = None,
    fixed_threshold: float | None = None,
    sample_fps: float = 2.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
    max_frames: int | None = None,
    top_k_cls: int = 5,
    attention_power: float = 1.5,
    min_weight: float = 0.15,
    alpha: float = 0.45,
    visualization_mode: str = "original",
) -> pd.DataFrame:
    """
    Create an MP4 containing the anomaly heatmap and detector information.

    visualization_mode:
    - "original": overlay on the original video frame.
    - "model": overlay on the square model input.
    """
    if visualization_mode not in {"original", "model"}:
        raise ValueError(
            "visualization_mode must be either 'original' or 'model'."
        )

    add_progress_to_nominal_memory(nominal_memory)

    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    writer: cv2.VideoWriter | None = None
    rows: list[dict[str, float | int | bool | str | None]] = []

    try:
        for frame_id, timestamp_sec, frame in iter_video_frames(
            video_path=test_video_path,
            sample_fps=sample_fps,
            start_sec=start_sec,
            end_sec=end_sec,
            max_frames=max_frames,
        ):
            (
                test_features,
                test_attention,
                test_cls,
                grid_size,
            ) = extractor.extract(frame)

            (
                score,
                _patch_scores,
                weighted_patch_scores,
                _weights,
                best_match,
                cls_distance,
            ) = compare_frame_to_nominal_memory(
                test_features=test_features,
                test_attention=test_attention,
                test_cls=test_cls,
                nominal_memory=nominal_memory,
                top_k_cls=top_k_cls,
                attention_power=attention_power,
                min_weight=min_weight,
            )

            matched_progress = float(best_match["progress"])

            if adaptive_threshold_df is not None:
                current_threshold = threshold_for_progress(
                    matched_progress,
                    adaptive_threshold_df,
                )
                threshold_mode = "adaptive"
            else:
                current_threshold = fixed_threshold
                threshold_mode = "fixed"

            if current_threshold is None:
                epsilon = np.nan
                alarm = False
            else:
                epsilon = score / max(float(current_threshold), 1e-8)
                alarm = bool(epsilon > 1.0)

            if visualization_mode == "model":
                visualization_frame = extractor.resize_frame(frame)
            else:
                visualization_frame = frame.convert("RGB")

            overlay_rgb = make_heatmap_overlay(
                frame=visualization_frame,
                weighted_patch_scores=weighted_patch_scores,
                grid_size=grid_size,
                alpha=alpha,
            )
            overlay_bgr = cv2.cvtColor(
                overlay_rgb,
                cv2.COLOR_RGB2BGR,
            )

            overlay_bgr = _draw_text_panel(
                overlay_bgr=overlay_bgr,
                timestamp_sec=timestamp_sec,
                score=score,
                threshold=current_threshold,
                epsilon=epsilon,
                alarm=alarm,
                cls_distance=cls_distance,
                matched_progress=matched_progress,
                matched_video_id=int(best_match["video_id"]),
                matched_frame_id=int(best_match["frame_id"]),
                threshold_mode=threshold_mode,
            )

            if writer is None:
                height, width = overlay_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_video_path),
                    fourcc,
                    sample_fps,
                    (width, height),
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not create output video: {output_video_path}"
                    )

            writer.write(overlay_bgr)

            rows.append(
                {
                    "frame_id": frame_id,
                    "timestamp_sec": timestamp_sec,
                    "score": score,
                    "threshold": current_threshold,
                    "epsilon": epsilon,
                    "alarm": alarm,
                    "threshold_mode": threshold_mode,
                    "cls_distance": cls_distance,
                    "matched_video_id": int(best_match["video_id"]),
                    "matched_frame_id": int(best_match["frame_id"]),
                    "matched_progress": matched_progress,
                }
            )

            print(
                f"Video frame={frame_id:03d}, "
                f"time={timestamp_sec:.2f}s, "
                f"score={score:.5f}, "
                f"alarm={alarm}"
            )
    finally:
        if writer is not None:
            writer.release()

    if not rows:
        raise ValueError(
            "No frames were written. Check the video path and time limits."
        )

    print("\nSaved heatmap video to:")
    print(output_video_path)

    return pd.DataFrame(rows)

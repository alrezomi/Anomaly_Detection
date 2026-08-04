"""Leave-one-run-out calibration and adaptive thresholding for GMR scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .anomaly_detection import GMRAnomalyDetector


def build_phase_threshold_from_scores(
    calibration_scores: pd.DataFrame,
    n_bins: int = 50,
    quantile: float = 0.999,
    margin: float = 1.60,
    min_points_per_bin: int = 5,
    threshold_floor_quantile: float = 0.80,
    smooth_threshold_window: int | None = 5,
) -> pd.DataFrame:
    """Build a phase-dependent Mahalanobis threshold from nominal scores."""
    required_columns = {"phase", "mahalanobis"}
    if not required_columns.issubset(calibration_scores.columns):
        raise ValueError(
            "calibration_scores must contain 'phase' and 'mahalanobis'."
        )
    if calibration_scores.empty:
        raise ValueError("calibration_scores is empty.")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    all_scores = calibration_scores["mahalanobis"].to_numpy(dtype=float)

    global_threshold = float(np.quantile(all_scores, quantile) * margin)
    threshold_floor = float(
        np.quantile(all_scores, threshold_floor_quantile)
    )

    thresholds: list[float] = []
    counts: list[int] = []

    for bin_index in range(n_bins):
        left = bins[bin_index]
        right = bins[bin_index + 1]

        if bin_index == n_bins - 1:
            mask = (
                (calibration_scores["phase"] >= left)
                & (calibration_scores["phase"] <= right)
            )
        else:
            mask = (
                (calibration_scores["phase"] >= left)
                & (calibration_scores["phase"] < right)
            )

        bin_scores = calibration_scores.loc[
            mask,
            "mahalanobis",
        ].to_numpy(dtype=float)
        counts.append(len(bin_scores))

        if len(bin_scores) >= min_points_per_bin:
            threshold = float(np.quantile(bin_scores, quantile) * margin)
        else:
            threshold = np.nan

        thresholds.append(threshold)

    threshold_dataframe = pd.DataFrame(
        {
            "phase_center": centers,
            "threshold_mahalanobis": thresholds,
            "count": counts,
        }
    )

    threshold_dataframe["threshold_mahalanobis"] = (
        threshold_dataframe["threshold_mahalanobis"]
        .interpolate()
        .bfill()
        .ffill()
        .fillna(global_threshold)
    )

    threshold_dataframe["threshold_mahalanobis"] = np.maximum(
        threshold_dataframe["threshold_mahalanobis"].to_numpy(),
        threshold_floor,
    )

    if smooth_threshold_window is not None and smooth_threshold_window > 1:
        threshold_dataframe["threshold_mahalanobis"] = (
            threshold_dataframe["threshold_mahalanobis"]
            .rolling(
                window=smooth_threshold_window,
                center=True,
                min_periods=1,
            )
            .median()
        )

    return threshold_dataframe


def calibrate_gmr_leave_one_run_out(
    nominal_dataframes: list[pd.DataFrame],
    input_columns: list[str],
    output_columns: list[str],
    n_components: int = 3,
    threshold_margin: float = 1.10,
    random_state: int = 42,
    n_bins: int = 50,
    quantile: float = 0.999,
    margin: float = 1.60,
    min_points_per_bin: int = 5,
    threshold_floor_quantile: float = 0.80,
    smooth_threshold_window: int | None = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate the adaptive threshold using held-out nominal executions."""
    if len(nominal_dataframes) < 2:
        raise ValueError(
            "At least two nominal executions are required for calibration."
        )

    held_out_results: list[pd.DataFrame] = []

    for held_out_run in range(len(nominal_dataframes)):
        print(
            "Leave-one-run-out calibration: "
            f"holding out nominal run {held_out_run}"
        )

        training_dataframes = [
            dataframe
            for run_index, dataframe in enumerate(nominal_dataframes)
            if run_index != held_out_run
        ]
        validation_dataframe = nominal_dataframes[held_out_run]

        temporary_detector = GMRAnomalyDetector(
            input_columns=input_columns,
            output_columns=output_columns,
            n_components=n_components,
            threshold_margin=threshold_margin,
            random_state=random_state,
        )
        temporary_detector.fit(training_dataframes)

        validation_scores = temporary_detector.score_dataframe(
            validation_dataframe
        )
        validation_scores["held_out_run"] = held_out_run
        held_out_results.append(validation_scores)

    calibration_scores = pd.concat(
        held_out_results,
        ignore_index=True,
    )

    threshold_dataframe = build_phase_threshold_from_scores(
        calibration_scores=calibration_scores,
        n_bins=n_bins,
        quantile=quantile,
        margin=margin,
        min_points_per_bin=min_points_per_bin,
        threshold_floor_quantile=threshold_floor_quantile,
        smooth_threshold_window=smooth_threshold_window,
    )

    scores = calibration_scores["mahalanobis"].to_numpy(dtype=float)
    print("\nAdaptive-threshold calibration summary:")
    print("Minimum held-out score:", float(scores.min()))
    print("Mean held-out score:", float(scores.mean()))
    print("Maximum held-out score:", float(scores.max()))
    print(f"{quantile:.3f} quantile:", float(np.quantile(scores, quantile)))
    print(
        "Equivalent global threshold:",
        float(np.quantile(scores, quantile) * margin),
    )

    return threshold_dataframe, calibration_scores


def apply_adaptive_gmr_threshold(
    result_dataframe: pd.DataFrame,
    threshold_dataframe: pd.DataFrame,
    score_smooth_window: int = 11,
    majority_window: int = 8,
    majority_votes: int = 6,
) -> pd.DataFrame:
    """Apply the phase threshold, score smoothing, and majority filtering."""
    required_result_columns = {"phase", "mahalanobis"}
    if not required_result_columns.issubset(result_dataframe.columns):
        raise ValueError(
            "result_dataframe must contain 'phase' and 'mahalanobis'."
        )

    required_threshold_columns = {
        "phase_center",
        "threshold_mahalanobis",
    }
    if not required_threshold_columns.issubset(threshold_dataframe.columns):
        raise ValueError(
            "threshold_dataframe must contain 'phase_center' and "
            "'threshold_mahalanobis'."
        )

    if score_smooth_window < 1:
        raise ValueError("score_smooth_window must be at least 1.")
    if majority_window < 1:
        raise ValueError("majority_window must be at least 1.")
    if not 1 <= majority_votes <= majority_window:
        raise ValueError(
            "majority_votes must be between 1 and majority_window."
        )

    output = result_dataframe.copy()
    thresholds = np.interp(
        output["phase"].to_numpy(dtype=float),
        threshold_dataframe["phase_center"].to_numpy(dtype=float),
        threshold_dataframe["threshold_mahalanobis"].to_numpy(dtype=float),
    )

    output["adaptive_threshold_mahalanobis"] = thresholds
    output["epsilon_adaptive"] = (
        output["mahalanobis"] / np.maximum(thresholds, 1e-8)
    )

    output["epsilon_smooth"] = (
        output["epsilon_adaptive"]
        .rolling(
            window=score_smooth_window,
            center=True,
            min_periods=1,
        )
        .median()
    )
    output["alarm_raw"] = output["epsilon_smooth"] > 1.0

    rolling_votes = (
        output["alarm_raw"]
        .astype(int)
        .rolling(
            window=majority_window,
            center=False,
            min_periods=1,
        )
        .sum()
    )
    output["alarm_filtered"] = rolling_votes >= majority_votes

    return output

"""Plots for time-series synchronization, calibration, and anomaly results."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_raw_signals(
    dataframe: pd.DataFrame,
    signal_columns: list[str],
    title: str = "Force/torque signals",
) -> None:
    """Plot the selected force and torque channels against time."""
    plt.figure(figsize=(12, 5))

    for column in signal_columns:
        plt.plot(dataframe["time"], dataframe[column], label=column)

    plt.xlabel("Time [s]")
    plt.ylabel("Signal value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def plot_force_synchronization(
    debug_dataframe: pd.DataFrame,
    sync_info: dict[str, Any],
    title: str = "Force-based synchronization",
) -> None:
    """Show the force magnitude and the selected active task segment."""
    plt.figure(figsize=(12, 4))
    plt.plot(
        debug_dataframe["time"],
        debug_dataframe["force_mag"],
        linewidth=2,
        label="Force magnitude",
    )
    plt.plot(
        debug_dataframe["time"],
        np.full(len(debug_dataframe), sync_info["threshold"]),
        linewidth=2,
        label="Activity threshold",
    )
    plt.axvspan(
        sync_info["start_time"],
        sync_info["end_time"],
        alpha=0.20,
        label="Selected active segment",
    )

    plt.xlabel("Original bag time [s]")
    plt.ylabel("Force magnitude")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def plot_gmr_calibration(
    threshold_dataframe: pd.DataFrame,
    calibration_scores: pd.DataFrame,
    title: str = "Leave-one-run-out nominal calibration",
) -> None:
    """Plot held-out nominal scores and the phase-adaptive threshold."""
    plt.figure(figsize=(13, 4.5))
    plt.scatter(
        calibration_scores["phase"],
        calibration_scores["mahalanobis"],
        s=8,
        alpha=0.25,
        label="Held-out nominal scores",
    )
    plt.plot(
        threshold_dataframe["phase_center"],
        threshold_dataframe["threshold_mahalanobis"],
        linewidth=2.5,
        label="Adaptive threshold",
    )

    plt.xlabel("Task phase")
    plt.ylabel("Mahalanobis distance")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def plot_gmr_mahalanobis_with_threshold(
    result_dataframe: pd.DataFrame,
    title: str = "GMR Mahalanobis distance with adaptive threshold",
) -> None:
    """Plot Mahalanobis distance, threshold, and filtered alarms."""
    plt.figure(figsize=(13, 4.5))
    plt.plot(
        result_dataframe["time"],
        result_dataframe["mahalanobis"],
        linewidth=2.2,
        label="Mahalanobis distance",
    )
    plt.plot(
        result_dataframe["time"],
        result_dataframe["adaptive_threshold_mahalanobis"],
        linewidth=2.2,
        label="Adaptive threshold",
    )

    alarm_points = result_dataframe[result_dataframe["alarm_filtered"]]
    if not alarm_points.empty:
        plt.scatter(
            alarm_points["time"],
            alarm_points["mahalanobis"],
            marker="x",
            s=80,
            label="Filtered alarm",
        )

    plt.xlabel("Synchronized time [s]")
    plt.ylabel("Mahalanobis distance")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def plot_gmr_epsilon_filtered(
    result_dataframe: pd.DataFrame,
    title: str = "Filtered normalized GMR anomaly score",
) -> None:
    """Plot the smoothed normalized score and the alarm threshold."""
    plt.figure(figsize=(13, 4.5))
    plt.plot(
        result_dataframe["time"],
        result_dataframe["epsilon_smooth"],
        linewidth=2.2,
        label="Smoothed normalized score",
    )
    plt.plot(
        result_dataframe["time"],
        np.ones(len(result_dataframe)),
        linewidth=2.2,
        label="Normalized threshold = 1",
    )

    alarm_points = result_dataframe[result_dataframe["alarm_filtered"]]
    if not alarm_points.empty:
        plt.scatter(
            alarm_points["time"],
            alarm_points["epsilon_smooth"],
            marker="x",
            s=80,
            label="Filtered alarm",
        )

    plt.xlabel("Synchronized time [s]")
    plt.ylabel("Normalized anomaly score")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

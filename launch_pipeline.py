"""Launch the vision pipeline, time-series pipeline, or both."""

from __future__ import annotations

import argparse

from experiment_config import print_experiment_summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run anomaly detection using shared experiment inputs."
    )
    parser.add_argument(
        "section",
        nargs="?",
        choices=("both", "vision", "time-series"),
        default="both",
        help="Pipeline section to run (default: both).",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    print_experiment_summary()

    if arguments.section in {"both", "vision"}:
        from vision_dinov2.run_vision_test import main as run_vision

        print("\nStarting vision pipeline...")
        run_vision()

    if arguments.section in {"both", "time-series"}:
        from time_series_gmr_scripts.run_time_series_test import (
            main as run_time_series,
        )

        print("\nStarting time-series pipeline...")
        run_time_series()

    # The individual plotting calls are non-blocking so processing can
    # continue. Keep any remaining figures open after the selected work ends.
    import matplotlib.pyplot as plt

    if plt.get_fignums():
        print("\nProcessing finished. Close the graph windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()

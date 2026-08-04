"""Gaussian-mixture-regression anomaly detection for force/torque signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class GMRAnomalyDetector:
    """
    Model nominal force/torque signals as a function of normalized task phase.

    The model fits a joint Gaussian mixture over input and output variables.
    For each sample, it selects the most likely mixture component from the
    input variables, computes the conditional output distribution, and uses
    Mahalanobis distance as the anomaly score.
    """

    def __init__(
        self,
        input_columns: list[str],
        output_columns: list[str],
        n_components: int = 3,
        covariance_type: str = "full",
        reg_covar: float = 1e-5,
        threshold_margin: float = 1.10,
        random_state: int = 42,
    ) -> None:
        if not input_columns:
            raise ValueError("At least one input column is required.")
        if not output_columns:
            raise ValueError("At least one output column is required.")
        if covariance_type != "full":
            raise ValueError(
                "This GMR implementation currently requires covariance_type='full'."
            )

        self.input_columns = list(input_columns)
        self.output_columns = list(output_columns)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.threshold_margin = threshold_margin
        self.random_state = random_state

        self.joint_columns = list(
            dict.fromkeys([*self.input_columns, *self.output_columns])
        )
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=random_state,
        )

        self.component_thresholds: dict[int, float] = {}
        self.training_distances: dict[int, list[float]] = {}
        self.fitted = False

    def _index_sets(self) -> tuple[list[int], list[int]]:
        input_indices = [
            self.joint_columns.index(column)
            for column in self.input_columns
        ]
        output_indices = [
            self.joint_columns.index(column)
            for column in self.output_columns
        ]
        return input_indices, output_indices

    def _safe_inverse(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        regularized = matrix + np.eye(matrix.shape[0]) * self.reg_covar
        return np.linalg.pinv(regularized)

    def _gaussian_logpdf(
        self,
        sample: np.ndarray,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        sample = np.asarray(sample, dtype=float)
        mean = np.asarray(mean, dtype=float)
        covariance = np.asarray(covariance, dtype=float)

        dimension = len(sample)
        covariance = (
            covariance + np.eye(dimension, dtype=float) * self.reg_covar
        )
        inverse_covariance = np.linalg.pinv(covariance)
        difference = sample - mean

        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            stable_determinant = np.linalg.det(
                covariance + np.eye(dimension) * 1e-4
            )
            log_determinant = np.log(max(stable_determinant, 1e-12))

        return float(
            -0.5
            * (
                dimension * np.log(2.0 * np.pi)
                + log_determinant
                + difference.T @ inverse_covariance @ difference
            )
        )

    def _select_component_from_input(
        self,
        scaled_input: np.ndarray,
    ) -> int:
        input_indices, _ = self._index_sets()
        log_probabilities: list[float] = []

        for component in range(self.n_components):
            component_weight = self.gmm.weights_[component]
            component_mean = self.gmm.means_[component]
            component_covariance = self.gmm.covariances_[component]

            input_mean = component_mean[input_indices]
            input_covariance = component_covariance[
                np.ix_(input_indices, input_indices)
            ]

            log_probability = np.log(component_weight + 1e-12)
            log_probability += self._gaussian_logpdf(
                sample=scaled_input,
                mean=input_mean,
                covariance=input_covariance,
            )
            log_probabilities.append(float(log_probability))

        return int(np.argmax(log_probabilities))

    def _conditional_output_distribution(
        self,
        scaled_input: np.ndarray,
        component: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        input_indices, output_indices = self._index_sets()

        mean = self.gmm.means_[component]
        covariance = self.gmm.covariances_[component]

        input_mean = mean[input_indices]
        output_mean = mean[output_indices]

        covariance_ii = covariance[np.ix_(input_indices, input_indices)]
        covariance_oi = covariance[np.ix_(output_indices, input_indices)]
        covariance_io = covariance[np.ix_(input_indices, output_indices)]
        covariance_oo = covariance[np.ix_(output_indices, output_indices)]

        inverse_covariance_ii = self._safe_inverse(covariance_ii)

        conditional_mean = (
            output_mean
            + covariance_oi
            @ inverse_covariance_ii
            @ (scaled_input - input_mean)
        )
        conditional_covariance = (
            covariance_oo
            - covariance_oi @ inverse_covariance_ii @ covariance_io
        )
        conditional_covariance = 0.5 * (
            conditional_covariance + conditional_covariance.T
        )
        conditional_covariance += (
            np.eye(conditional_covariance.shape[0]) * self.reg_covar
        )

        return conditional_mean, conditional_covariance

    def _score_scaled_point(
        self,
        scaled_joint_sample: np.ndarray,
    ) -> tuple[float, int]:
        input_indices, output_indices = self._index_sets()
        scaled_input = scaled_joint_sample[input_indices]
        scaled_output = scaled_joint_sample[output_indices]

        component = self._select_component_from_input(scaled_input)
        predicted_mean, predicted_covariance = (
            self._conditional_output_distribution(
                scaled_input=scaled_input,
                component=component,
            )
        )

        difference = scaled_output - predicted_mean
        inverse_covariance = self._safe_inverse(predicted_covariance)
        squared_distance = float(
            difference.T @ inverse_covariance @ difference
        )
        mahalanobis_distance = float(np.sqrt(max(squared_distance, 0.0)))

        return mahalanobis_distance, component

    def _validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        missing_columns = [
            column
            for column in self.joint_columns
            if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing model columns: {missing_columns}")
        if dataframe.empty:
            raise ValueError("The input DataFrame is empty.")

    def fit(self, nominal_dataframes: list[pd.DataFrame]) -> None:
        """Fit the scaler, joint GMM, and component-level reference limits."""
        if not nominal_dataframes:
            raise ValueError("At least one nominal execution is required.")

        training_arrays: list[np.ndarray] = []
        for dataframe in nominal_dataframes:
            self._validate_dataframe(dataframe)
            training_arrays.append(
                dataframe[self.joint_columns].to_numpy(dtype=float)
            )

        training_data = np.vstack(training_arrays)
        scaled_training_data = self.scaler.fit_transform(training_data)
        self.gmm.fit(scaled_training_data)

        component_distances: dict[int, list[float]] = {
            component: [] for component in range(self.n_components)
        }

        for scaled_sample in scaled_training_data:
            distance, component = self._score_scaled_point(scaled_sample)
            component_distances[component].append(distance)

        all_distances = [
            distance
            for distances in component_distances.values()
            for distance in distances
        ]
        global_maximum = max(all_distances) if all_distances else 1.0
        global_maximum = max(global_maximum, 1e-6)

        self.component_thresholds = {}
        for component in range(self.n_components):
            distances = component_distances[component]
            reference_maximum = max(distances) if distances else global_maximum
            self.component_thresholds[component] = (
                max(reference_maximum, 1e-6) * self.threshold_margin
            )

        self.training_distances = component_distances
        self.fitted = True

        print("GMR detector fitted.")
        print("Input columns:", self.input_columns)
        print("Output columns:", self.output_columns)
        print("Number of components:", self.n_components)
        print("Component reference thresholds:")
        for component, threshold in self.component_thresholds.items():
            print(f"  component {component}: {threshold:.4f}")

    def score_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate Mahalanobis and component-normalized scores."""
        if not self.fitted:
            raise RuntimeError("The GMR detector must be fitted first.")

        self._validate_dataframe(dataframe)
        joint_data = dataframe[self.joint_columns].to_numpy(dtype=float)
        scaled_joint_data = self.scaler.transform(joint_data)

        rows: list[dict[str, float | int | bool]] = []

        for index, scaled_sample in enumerate(scaled_joint_data):
            distance, component = self._score_scaled_point(scaled_sample)
            component_threshold = self.component_thresholds[component]
            epsilon = distance / max(component_threshold, 1e-8)

            rows.append(
                {
                    "index": index,
                    "time": float(
                        dataframe["time"].iloc[index]
                        if "time" in dataframe.columns
                        else index
                    ),
                    "phase": float(
                        dataframe["phase"].iloc[index]
                        if "phase" in dataframe.columns
                        else np.nan
                    ),
                    "mahalanobis": distance,
                    "component": component,
                    "component_threshold": component_threshold,
                    "epsilon": epsilon,
                    "anomaly": bool(epsilon > 1.0),
                }
            )

        return pd.DataFrame(rows)

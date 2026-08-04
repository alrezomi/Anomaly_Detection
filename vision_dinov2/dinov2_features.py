"""DINOv2 model loading, video reading, and feature extraction."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from PIL import Image

from rosbag_io import RosbagImageSource, iter_rosbag_image_frames


FrameSource = str | Path | RosbagImageSource
from transformers import AutoImageProcessor, AutoModel


class DINOv2FeatureExtractor:
    """
    Load DINOv2 once and extract:
    - local patch features,
    - CLS-to-patch attention,
    - the global CLS token.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        input_size: int = 518,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.input_size = input_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        print(f"Loading {self.model_name} on {self.device}...")

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)

        try:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                attn_implementation="eager",
            ).to(self.device)
        except TypeError:
            self.model = AutoModel.from_pretrained(
                self.model_name,
            ).to(self.device)

        self.model.eval()
        self.patch_size = int(getattr(self.model.config, "patch_size", 14))

        print("DINOv2 loaded.")
        print("Patch size:", self.patch_size)
        print("Input size:", self.input_size)

    def resize_frame(self, image: Image.Image) -> Image.Image:
        """Convert a frame to RGB and resize it to a fixed square input."""
        image = image.convert("RGB")
        return image.resize(
            (self.input_size, self.input_size),
            Image.Resampling.BICUBIC,
        )

    def extract(
        self,
        image: Image.Image,
        normalize_features: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Extract DINOv2 patch features, CLS attention, and CLS feature.

        Returns
        -------
        patch_features:
            Array with shape [number_of_patches, feature_dimension].
        attention_scores:
            Normalized CLS-to-patch attention with shape [number_of_patches].
        cls_feature:
            Normalized CLS feature with shape [feature_dimension].
        grid_size:
            Side length of the square patch grid.
        """
        image = self.resize_frame(image)

        try:
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
            )
        except TypeError:
            inputs = self.processor(
                images=image,
                return_tensors="pt",
            )

        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        with torch.inference_mode():
            try:
                outputs = self.model(
                    **inputs,
                    output_attentions=True,
                    interpolate_pos_encoding=True,
                )
            except TypeError:
                outputs = self.model(
                    **inputs,
                    output_attentions=True,
                )

        if outputs.attentions is None:
            raise RuntimeError(
                "DINOv2 did not return attention matrices. "
                "The model must be loaded with attn_implementation='eager'."
            )

        hidden = outputs.last_hidden_state[0]
        last_attention = outputs.attentions[-1][0]
        mean_attention = last_attention.mean(dim=0)

        pixel_values = inputs["pixel_values"]
        input_height = int(pixel_values.shape[-2])
        input_width = int(pixel_values.shape[-1])

        grid_height = input_height // self.patch_size
        grid_width = input_width // self.patch_size
        expected_patch_count = grid_height * grid_width

        total_tokens = int(hidden.shape[0])
        special_token_count = total_tokens - expected_patch_count

        if special_token_count < 1:
            raise ValueError(
                "Unexpected DINOv2 token structure: "
                f"total_tokens={total_tokens}, "
                f"expected_patches={expected_patch_count}."
            )

        cls_feature = hidden[0]
        patch_features = hidden[special_token_count:]
        attention_scores = mean_attention[0, special_token_count:]

        if patch_features.shape[0] != expected_patch_count:
            raise ValueError(
                "The extracted patch count does not match the expected patch grid."
            )

        if attention_scores.shape[0] != expected_patch_count:
            raise ValueError(
                "The extracted attention count does not match the expected patch grid."
            )

        if normalize_features:
            patch_features = torch.nn.functional.normalize(
                patch_features,
                dim=-1,
            )
            cls_feature = torch.nn.functional.normalize(
                cls_feature,
                dim=-1,
            )

        patch_features_np = (
            patch_features.detach().cpu().float().numpy()
        )
        cls_feature_np = (
            cls_feature.detach().cpu().float().numpy()
        )
        attention_scores_np = (
            attention_scores.detach().cpu().float().numpy()
        )

        attention_scores_np = attention_scores_np - attention_scores_np.min()
        attention_scores_np = attention_scores_np / (
            attention_scores_np.max() + 1e-8
        )

        if grid_height != grid_width:
            raise ValueError(
                f"Expected a square patch grid, got "
                f"{grid_height} x {grid_width}."
            )

        grid_size = grid_height

        if not hasattr(self, "_printed_extraction_info"):
            print("DINOv2 extraction check:")
            print(f"  input: {input_height} x {input_width}")
            print(f"  total tokens: {total_tokens}")
            print(f"  special tokens removed: {special_token_count}")
            print(f"  patch tokens: {patch_features_np.shape[0]}")
            print(f"  patch grid: {grid_size} x {grid_size}")
            print(f"  feature dimension: {patch_features_np.shape[1]}")
            print(f"  CLS shape: {cls_feature_np.shape}")
            self._printed_extraction_info = True

        return (
            patch_features_np,
            attention_scores_np,
            cls_feature_np,
            grid_size,
        )


def iter_video_frames(
    video_path: str | Path,
    sample_fps: float = 2.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
    max_frames: int | None = None,
) -> Iterator[tuple[int, float, Image.Image]]:
    """
    Read a video and yield sampled frames in chronological order.

    Yields
    ------
    processed_frame_id, timestamp_seconds, PIL_RGB_frame
    """
    video_path = str(video_path)

    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero.")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0:
        source_fps = 30.0

    start_frame = 0 if start_sec is None else int(start_sec * source_fps)
    end_frame = (
        total_frames - 1
        if end_sec is None
        else int(end_sec * source_fps)
    )

    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)

    if start_frame > end_frame:
        cap.release()
        raise ValueError(
            "The selected start time is after the selected end time."
        )

    sample_step = max(1, int(round(source_fps / sample_fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    processed_count = 0
    current_frame_index = start_frame

    try:
        while current_frame_index <= end_frame:
            success, frame_bgr = cap.read()

            if not success:
                break

            relative_index = current_frame_index - start_frame

            if relative_index % sample_step == 0:
                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )
                frame_pil = Image.fromarray(frame_rgb).convert("RGB")
                timestamp_sec = current_frame_index / source_fps

                yield processed_count, timestamp_sec, frame_pil

                processed_count += 1

                if max_frames is not None and processed_count >= max_frames:
                    break

            current_frame_index += 1
    finally:
        cap.release()


def iter_source_frames(
    source: FrameSource,
    sample_fps: float = 2.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
    max_frames: int | None = None,
) -> Iterator[tuple[int, float, Image.Image]]:
    """Yield sampled frames from either a normal video or a ROS bag camera."""
    if isinstance(source, RosbagImageSource):
        yield from iter_rosbag_image_frames(
            source=source,
            sample_fps=sample_fps,
            start_sec=start_sec,
            end_sec=end_sec,
            max_frames=max_frames,
        )
        return

    yield from iter_video_frames(
        video_path=source,
        sample_fps=sample_fps,
        start_sec=start_sec,
        end_sec=end_sec,
        max_frames=max_frames,
    )

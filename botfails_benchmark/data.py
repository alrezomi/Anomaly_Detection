"""BotFails episodes: extracted 2 fps camera frames, robot data and per-frame anomaly labels.

Layout (Hugging Face kantine/BotFails):
  BotFails/normal_train/<task>_expert/{data,meta,videos}   normal demonstrations (memory)
  BotFails/test/<task>_anomaly/{data,meta,videos}          test episodes
  BotFails/labels/<task>_anomaly/episode_XXXXXX_labels.csv one 0/1 per frame (1 = anomaly)
Frames come from extract_frames.py: <frames>/<split>/<folder>/<episode>/<camera>/frames.csv.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS = 30.
TASKS = ('domotic_dishTidyUp', 'domotic_groceriesSorting', 'domotic_makingCoffee', 'domotic_pouringCoffee',
         'domotic_setTheTable', 'domotic_vegetablesAndFruitsSorting', 'industrial_robothon_buttons',
         'industrial_robothon_hatchAndProbe', 'industrial_screws_sorting', 'industrial_soldering')
SUFFIX = {'normal_train': 'expert', 'test': 'anomaly'}


@dataclass(frozen=True)
class Episode:
    dataset: Path  # .../BotFails
    frames: Path   # extracted frame root
    split: str     # normal_train or test
    task: str      # e.g. domotic_dishTidyUp
    index: int

    @property
    def name(self) -> str:
        return f'episode_{self.index:06d}'

    @property
    def folder(self) -> Path:
        return self.dataset / self.split / f'{self.task}_{SUFFIX[self.split]}'

    @property
    def key(self) -> str:
        return f'{self.split}/{self.task}/{self.name}'

    def frame_dir(self, camera: str) -> Path:
        return self.frames / self.split / self.folder.name / self.name / camera

    def frame_table(self, camera: str) -> list[tuple[int, float, Path]]:
        """(frame index, time in s, JPEG path) of the extracted frames, in time order."""
        directory = self.frame_dir(camera)
        with (directory / 'frames.csv').open() as stream:
            rows = [(int(r['frame_index']), float(r['timestamp_sec']), directory / r['file'])
                    for r in csv.DictReader(stream)]
        if not rows:
            raise ValueError(f'No extracted frames for {self.key} {camera}.')
        return rows

    def robot(self) -> dict[str, np.ndarray]:
        """30 Hz joint positions (observation.state) and leader-arm commands (action); index 5 is the gripper."""
        import pyarrow.parquet as pq
        table = pq.read_table(self.folder / 'data' / 'chunk-000' / f'{self.name}.parquet',
                              columns=['observation.state', 'action', 'frame_index']).to_pydict()
        order = np.argsort(table['frame_index'])
        return dict(state=np.asarray(table['observation.state'], float)[order],
                    action=np.asarray(table['action'], float)[order])

    @property
    def labels_path(self) -> Path:
        return self.dataset / 'labels' / f'{self.task}_anomaly' / f'{self.name}_labels.csv'

    def labels(self) -> np.ndarray:
        """One 0/1 anomaly label per 30 fps frame."""
        values = np.loadtxt(self.labels_path, dtype=float, ndmin=1).astype(int)
        if not set(np.unique(values)) <= {0, 1}:
            raise ValueError(f'Unexpected label values in {self.labels_path}.')
        return values


def episodes(dataset: Path, frames: Path, split: str, task: str, labelled_only: bool = True) -> list[Episode]:
    """Episodes of one split and task. Test episodes without a label file are skipped
    (setTheTable episode 7 has none in the dataset or in FIDeL's repository)."""
    folder = Path(dataset) / split / f'{task}_{SUFFIX[split]}' / 'data' / 'chunk-000'
    result = []
    for path in sorted(folder.glob('episode_*.parquet')):
        episode = Episode(Path(dataset), Path(frames), split, task, int(path.stem.split('_')[1]))
        if split == 'test' and labelled_only and not episode.labels_path.is_file():
            continue
        result.append(episode)
    if not result:
        raise ValueError(f'No {split} episodes for {task} in {folder}.')
    return result


def frame_labels(episode: Episode, frame_indices) -> np.ndarray:
    labels = episode.labels()
    indices = np.asarray(frame_indices, int)
    if indices.max() >= len(labels):
        raise ValueError(f'{episode.key}: frame {indices.max()} beyond {len(labels)} labels.')
    return labels[indices]


def load_image(path: Path, crop=None):
    """RGB image; crop = (x0, y0, x1, y1) in fractions of the image, upscaled so the longer side is 640 px."""
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert('RGB')
    if crop is None:
        return image
    width, height = image.size
    x0, y0, x1, y1 = crop
    image = image.crop((int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)))
    scale = 640 / max(image.size)
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.BICUBIC)

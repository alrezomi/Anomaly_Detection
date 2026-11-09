"""Decode BotFails AV1 videos once and keep the frames our DINO samples (2 fps) as JPEG.

The project image's OpenCV cannot decode AV1, so this runs where PyAV (dav1d) is
installed. The benchmark then reads JPEGs, like the compressed ROS images that
our system reads from bags.

    python -m botfails_benchmark.extract_frames --dataset <BotFails> --output <frames> --camera logitech_1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def episode_videos(dataset: Path, split: str, camera: str, tasks=None):
    for task_dir in sorted((dataset / split).iterdir()):
        if tasks and task_dir.name.rsplit('_', 1)[0] not in tasks:
            continue
        for video in sorted((task_dir / 'videos' / 'chunk-000' / f'observation.images.{camera}').glob('episode_*.mp4')):
            yield task_dir.name, video


def extract(video: Path, output: Path, step: int, quality: int, threads: int) -> dict:
    import av
    output.mkdir(parents=True, exist_ok=True)
    index = output / 'frames.csv'
    if index.is_file():  # Already complete (the index is written last).
        with index.open() as stream:
            rows = list(csv.DictReader(stream))
        return dict(frames=int(rows[-1]['decoded_frames']) if rows else 0, kept=len(rows), reused=True)
    rows, decoded = [], 0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = threads
        fps = float(stream.average_rate)
        for frame_index, frame in enumerate(container.decode(stream)):
            decoded += 1
            if frame_index % step:
                continue
            name = f'{frame_index:06d}.jpg'
            frame.to_image().save(output / name, quality=quality)
            rows.append(dict(frame_index=frame_index, timestamp_sec=round(frame_index / fps, 6), file=name))
    for row in rows:
        row['decoded_frames'] = decoded
    temporary = index.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['frame_index', 'timestamp_sec', 'file', 'decoded_frames'])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(index)
    return dict(frames=decoded, kept=len(rows), fps=fps, reused=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--camera', default='logitech_1')
    parser.add_argument('--splits', nargs='+', default=['normal_train', 'test'])
    parser.add_argument('--step', type=int, default=15, help='Keep every step-th frame (15 at 30 fps = 2 fps).')
    parser.add_argument('--quality', type=int, default=95)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--task', action='append', help='Only these tasks (e.g. domotic_pouringCoffee); repeat.')
    args = parser.parse_args()
    summary = []
    for split in args.splits:
        for task, video in episode_videos(args.dataset, split, args.camera, args.task):
            target = args.output / split / task / video.stem / args.camera
            result = extract(video, target, args.step, args.quality, args.threads)
            summary.append(dict(split=split, task=task, episode=video.stem, camera=args.camera, **result))
            print(json.dumps(summary[-1]), flush=True)
    if not args.task:
        (args.output / f'extraction_{args.camera}.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

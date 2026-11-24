# BotFails benchmark (comparison with FIDeL, ICRA 2026)

Runs our DINO -> VLM cascade on [BotFails](https://huggingface.co/datasets/kantine/BotFails)
(LeRobot v2.0, SO-100 arm, 30 fps, two third-person Logitech cameras, AV1 video),
the dataset of *Failure Identification in Imitation Learning via Statistical and
Semantic Filtering* (Rolland et al., ICRA 2026, arXiv 2604.13788).
Results and the full write-up: `/home/steribot/projects/AD/evaluation/FIDeL/README.md`.

## What is the same as on our robot

| Part | Setting (pipeline_config.json / code constants) |
|---|---|
| DINO | DINOv3 ViT-S/16+, 832 px, 2 fps, CLS top-5 matching, attention weights, pooled 3x3 score, leave-one-episode-out progress calibration (95 % quantiles, 30 bins), Euclidean metric |
| Alarm | pooled score > 1.1 x threshold in 3 frames in a row (`AlarmEpisodes`), re-armed after the score drops or the phase changes |
| VLM | RynnBrain-CoP-8B base model (no LoRA), first-token probability of the answer options (`answer_distribution`), frames at the alarm and 0.5 s / 1 s before it, mean over frames < 0.5 = failure, which stops the robot |

## What BotFails forces us to change

| Our robot | BotFails | Here |
|---|---|---|
| Stages 1/2/3 published by the robot | no stage messages | `phases.py`: hold/release from the gripper command and finger position (causal) |
| Stage-constrained DINO matching | no stages | matching over all nominal frames (our `ignore_recorded_stages` mode); progress = frame position |
| Gripper camera for the VLM | only third-person cameras | `logitech_1`, the camera FIDeL's VLM also uses (DINO uses it too) |
| Stage 2 "held?" / stage 3 "on the destination?" | ten different tasks | one template (`prompts.py`) + per-task objects (`tasks.json`) |
| ROS bags (JPEG images) | AV1 MP4 | `extract_frames.py` decodes once (PyAV) and keeps the 2 fps frames as JPEG |

The hold question was dropped: the gripper already reports that it holds an
object, and every hold wording failed on 1-12 of 12 normal frames from the far
camera. Checks are accepted only if they falsely fail at most 1 of 12 random
normal training frames per phase (no test data used). Buttons (top of the box
not visible) and screws (screws too small) originally had no accepted check.
Empty task tables now contain an explicit task-context question, and uncovered phases use
the same instruction-based template. A task with neither a question nor an instruction raises
an error. There is no DINO-only fallback. New context questions use the existing 0.5 default,
with distinct names so old calibrated cutoffs are not reused for different questions.
Their limited normal-alarm check and the legacy 0.9527 soldering cutoff audit are documented
in `evaluation/FIDeL/improvement/threshold_review/REVIEW.md`.

## Improved configuration

`tasks_final.json` and `thresholds_final.json` (chosen on development episodes; see
`evaluation/FIDeL/improvement/README.md`) add a camera per task, region crops (`crop`), per-question thresholds
from normal frames (`--vlm-thresholds`) and a re-check while a three-frame DINO alarm persists (`--recheck-sec`).
Every stop requires VLM confirmation. Task-specific DINO decisions and checks triggered only by release
were removed on 2026-11-12 at the user's request. Old release-only commands are rejected.
`--split-file` / `--subset` restrict DINO, the cascade and the report to development or held-out
episodes, and `dino --cache` keeps features for re-scoring.

Historical hybrid results remain in `evaluation/FIDeL/improvement/final/cascade_test`;
they do not measure the strict cascade. The saved strict candidate is `final/cv_vlm`.
`experiments/final.sh` now runs the strict cascade and writes separate `strict_*` outputs.
The archived rule-selection scripts are not part of the current evaluation.

## Run

Reports now default to derived **failure** labels for threshold and full-cascade evaluation.
Build them once with `python -m botfails_benchmark.failure_labels --dataset <BotFails>`.
They live separately in `<evaluation>/failure_labels`; original anomaly labels and raw detector
diagnostics remain unchanged. See `failure_labels/README.md` for uncertainty and timing limitations.
Unknown episodes are excluded explicitly in `label_coverage.csv`; there is no silent label fallback.
Use `report --label-type anomaly` only to reproduce historical anomaly-based evaluations.
Frame FPR and episode false-stop rate are reported separately.

```sh
# 1. frames (host, needs PyAV): 2 fps JPEGs of logitech_1
python -m botfails_benchmark.extract_frames --dataset <BotFails> --output <frames> --camera logitech_1
# 2-4. in the GPU image, evaluation folder mounted at /evaluation
python run_botfails_benchmark.py dino    --dataset /evaluation/dataset/BotFails --frames /evaluation/frames --output /evaluation/results/dino_euclidean
python run_botfails_benchmark.py cascade --dataset ... --frames ... --dino-run /evaluation/results/dino_euclidean --output /evaluation/results/cascade
python run_botfails_benchmark.py report  --dataset ... --dino-run ... --cascade-run ... --output /evaluation/results/report
```
`cascade --split normal_train` replays the leave-one-episode-out DINO scores of the
normal training episodes (false stops in normal runs, no test data).

# Multimodal anomaly detection

This project contains two anomaly-detection sections that use one shared
experiment selection:

- DINOv2 vision anomaly detection
- GMR force/torque time-series anomaly detection

## Select the inputs once

Edit the first section of `experiment_config.py`:

```python
TASK_NAME = "pick_place"
NOMINAL_ITEM_IDS = (6, 7)
TEST_ITEM_ID = 11

NOMINAL_BAG_REPETITIONS = (1, 2, 3, 4, 5)
TEST_BAG_REPETITION = 1
```

The item IDs and task are shared by both modalities. For the example above,
the configuration produces:

```text
Vision nominal:
  i6_pick_place.MP4
  i7_pick_place.MP4

Vision test:
  i11_pick_place.MP4

Time-series nominal:
  exp1_i6_pp_1 ... exp1_i6_pp_5
  exp1_i7_pp_1 ... exp1_i7_pp_5

Time-series test:
  exp1_i11_pp_1
```

The video is stored at item level, while the force/torque data are split into
repetitions. That is why the time-series section has additional repetition
settings.

## Required layout

The paths are derived from the source files and do not depend on the Windows
username, drive letter, clone location, or terminal working directory.

```text
AD/
|-- data/
|   `-- Exp_1/
|       `-- Exp_1/
|           |-- Exp_1_Videos_u_Fotos/
|           |   |-- i6_pick_place.MP4
|           |   |-- i7_pick_place.MP4
|           |   `-- i11_pick_place.MP4
|           `-- Exp_1_Force_log_files/
|               `-- Bags/
|                   |-- exp1_i6_pp_1/
|                   |   `-- exp1_i6_pp_1_0.mcap
|                   `-- ...
`-- Script_VS/
    |-- experiment_config.py
    |-- launch_pipeline.py
    |-- time_series_gmr_scripts/
    `-- vision_dinov2/
```

## Run on Windows

Run both sections sequentially:

```powershell
.\.venv\Scripts\python.exe .\launch_pipeline.py
```

Run only vision:

```powershell
.\.venv\Scripts\python.exe .\launch_pipeline.py vision
```

Run only time series:

```powershell
.\.venv\Scripts\python.exe .\launch_pipeline.py time-series
```

The launcher runs sections sequentially so they do not compete for CPU, GPU,
or memory. Each section still owns its modality-specific processing settings.
For example, the force topic remains configured in
`time_series_gmr_scripts/run_time_series_test.py`.

Generated CSV files and the annotated video are written to
`Script_VS/outputs`.

## Read the new ROS 2 bag format

The bag is the source of truth; you do not need to convert camera topics to
MP4 before inference. First inspect any new recording:

```powershell
.\.venv\Scripts\python.exe .\inspect_rosbag.py topics ..\data\bag_20260730_094942
```

The example bag contains three RGB viewpoints, joint states, TCP pose, and
stage markers. It does not currently contain a force/wrench topic. Topic names
are recording-dependent, so inspect each dataset instead of hard-coding an
assumption.

To preview exactly what DINOv2 receives (sampled at 2 FPS and resized to
518 x 518), run:

```powershell
.\.venv\Scripts\python.exe .\run_rosbag_vision.py `
  --test-bag ..\data\bag_20260730_094942 `
  --camera-topic /flange_camera/cam33/color/image_raw `
  --preview-only
```

The result is `outputs/flange_camera_cam33_color_image_raw_model_input.mp4`.
This is a visualization copy; inference reads the image messages directly
from the bag.

### Train/test one or several camera viewpoints

Adaptive threshold calibration requires at least two separate nominal
recordings. Supply each normal bag once, then repeat `--camera-topic` for any
number of viewpoints:

```powershell
.\.venv\Scripts\python.exe .\run_rosbag_vision.py `
  --nominal-bag ..\data\normal_run_01 `
  --nominal-bag ..\data\normal_run_02 `
  --test-bag ..\data\test_run_01 `
  --camera-topic /flange_camera/cam01/color/image_raw `
  --camera-topic /flange_camera/cam33/color/image_raw `
  --camera-topic /flange_camera/cam45/color/image_raw
```

Each viewpoint deliberately gets its own nominal memory, thresholds, CSVs,
and heatmap video. Do not pool different viewpoints into one DINO memory:
camera-position differences would then be confused with process anomalies.

The rosbag input selection is made directly in this command:

- Repeat `--nominal-bag PATH` for every successful training recording.
- Set `--test-bag PATH` to the recording being evaluated.
- Repeat `--camera-topic TOPIC` for every selected camera viewpoint.
- Set `--output-dir PATH` when outputs should go somewhere other than the
  default `outputs` directory.

For example, inside Docker the mounted data and output paths are:

```bash
docker compose -p anomaly-detection run --rm \
  --entrypoint python pipeline run_rosbag_vision.py \
  --nominal-bag /data/normal_run_01 \
  --nominal-bag /data/normal_run_02 \
  --test-bag /data/test_run_01 \
  --camera-topic /flange_camera/cam33/color/image_raw \
  --output-dir /outputs/experiment_01
```

For the older MP4/MCAP dataset, edit nominal/test IDs and paths in
`experiment_config.py`; its shared `OUTPUT_DIRECTORY` controls those legacy
pipeline outputs.

### Select time-series topics

Any supported pose, joint-state, or wrench topic can be exported independently:

```powershell
.\.venv\Scripts\python.exe .\inspect_rosbag.py export-topic `
  ..\data\bag_20260730_094942 `
  --topic /tcp_pose_broadcaster/pose `
  --output outputs\tcp_pose.csv

.\.venv\Scripts\python.exe .\inspect_rosbag.py export-topic `
  ..\data\bag_20260730_094942 `
  --topic /joint_states `
  --output outputs\joint_states.csv

.\.venv\Scripts\python.exe .\inspect_rosbag.py export-stages `
  ..\data\bag_20260730_094942 `
  --output outputs\recording_stages.csv
```

When a future bag contains `geometry_msgs/msg/Wrench` or `WrenchStamped`, use
the same `export-topic` command with that topic name. The readers retain bag
timestamps and keep topics as separate tables because cameras, joints, poses,
and force sensors may publish at different rates. Synchronization/resampling
should happen explicitly when those signals are selected for a model.

### Recorded stages and automatic fallback

For rosbag camera inputs, the vision pipeline reads `/recording_stage` by
default. A recorded sequence is used only when it contains the complete,
ordered numeric sequence `1, 2, 3`. Non-numeric messages such as `Error` are
kept in the raw stage export but ignored as stage transitions.

Some recorders publish a rapid `1, 2, 3` initialization sequence. The resolver
works backwards from the final stage and selects the latest valid preceding
markers, which prevents this startup sequence from defining the operation.
For `bag_20260730_094942`, the selected starts are approximately 0.64, 31.62,
and 55.22 seconds.

If the stage topic is absent, a marker is missing, or the sequence is out of
order, that recording automatically uses equal-duration intervals. This is
equivalent to the previous frame-progress behavior. The console reports either
`recorded` or `equal_intervals` for every nominal bag so the decision is
visible during a run.

The vision result CSV also includes `execution_stage`, `execution_progress`,
`stage_source`, and `latest_recorded_marker`. Therefore an `Error` marker is
preserved for later evaluation even though it does not change the numeric
stage sequence. Because it was entered manually after the physical anomaly,
it should be treated as an approximate annotation rather than the exact
anomaly onset time.

If a recording uses another topic or number of stages, pass for example
`--stage-topic /my_stage --stage-count 4`. To force the former behavior for
an experiment, add `--ignore-recorded-stages`.

### Labels from recorded stage messages

Generate or refresh one label for every bag with:

```powershell
.\.venv\Scripts\python.exe .\build_dataset_manifest.py `
  ..\data --output dataset_labels.csv --startup-ignore-sec 0.1
```

This scans every immediate ROS bag directory automatically. Add `--recursive`
if bags are nested more deeply. After the startup-ignore interval, the label is
derived only from the final three recorded stage messages:

- If any of those final three messages contains `Error`, `Anomaly`, `Fail`, or
  a related spelling, the whole bag is labeled `fail`.
- If up to three final messages exist and none is a failure marker, the bag is
  labeled `normal`.
- A missing stage topic, no stage messages, or an unreadable bag gives it
  `unknown`.

The generated CSV contains no nominal/test split and no manual label override.
You continue selecting nominal and test bag paths yourself when launching the
pipeline.

## Run with Docker

Docker provides the same Linux, Python, PyTorch, and package environment on
Windows and Ubuntu. The large data directory is mounted read-only and is not
copied into the image. Generated files are written to the host output directory
selected in `.env`.

The Docker container is headless. Native Matplotlib windows are therefore not
displayed, but calculations, CSV output, and annotated videos still work.

### Build once

Run from the cloned `anomaly_detection` repository directory:

Create the local host-path file first. It is ignored by Git, so each computer
can use its own paths:

```bash
cp .env.example .env
```

Edit `.env` to configure host filesystem paths only:

```dotenv
ANOMALY_DATA_DIR=../data
ANOMALY_OUTPUT_DIR=./outputs
PIPELINE_CONFIG_FILE=./pipeline_config.json
COMPOSE_PROJECT_NAME=anomaly-detection
STAGE_STARTUP_IGNORE_SEC=0.1
```

Relative paths are resolved from the folder containing `compose.yaml`.
Absolute paths are also supported. Only the left/host paths change between
computers; the container always sees them as `/data` and `/outputs`.

Before selecting nominal and test bags, build the image and scan all recordings:

```bash
docker compose build
docker compose run --rm bag-labels
```

This recursively discovers every ROS 2 bag underneath `ANOMALY_DATA_DIR`;
timestamp names such as `bag_20260730_094554` require no code changes. Review
`dataset_labels.csv` inside `ANOMALY_OUTPUT_DIR` to see each bag's `normal`,
`fail`, or `unknown` label.

Only after reviewing those labels, create the experiment configuration:

```bash
cp pipeline_config.example.json pipeline_config.json
```

Edit `pipeline_config.json` to select the experiment:

```json
{
  "nominal_bags": [
    "/data/bag_20260730_094554",
    "/data/bag_20260730_094800"
  ],
  "test_bag": "/data/bag_20260730_110856",
  "camera_topics": [
    "/flange_camera/cam33/color/image_raw"
  ],
  "output_dir": "/outputs/experiment_01",
  "nominal_cache_dir": "/outputs/nominal_cache",
  "rebuild_nominal_cache": false,
  "stage_topic": "/recording_stage",
  "stage_count": 3,
  "ignore_recorded_stages": false,
  "preview_only": false,
  "max_preview_frames": null
}
```

### Named DINO memories and high-resolution heatmaps

Set a unique memory name and a DINO input size that is divisible by patch size
14:

```json
"vision_memory_name": "pick_place_dino728_v1",
"dino_input_size": 728,
"nominal_cache_dir": "/outputs/nominal_cache",
"output_dir": "/outputs/experiments/pick_place_dino728_v1_test_01",
"rebuild_nominal_cache": true
```

This produces a 52 x 52 patch grid and stores each camera memory under
`/outputs/nominal_cache/pick_place_dino728_v1/<camera>/`. Choose a different
`output_dir` for every experiment so videos and CSV results are not overwritten.
After the first successful build, set `rebuild_nominal_cache` to `false`.
To select an existing memory later, use its `vision_memory_name`, matching
`dino_input_size`, and keep rebuilding disabled. The saved signature is checked
before loading, so a memory made with incompatible bags, topics, or DINO settings
is rejected rather than silently reused.

To prevent a late test frame from matching an early nominal pose, enable strict
same-stage matching:

```json
"ignore_recorded_stages": false,
"stage_constrained_matching": true
```

CLS top-k selection then searches only nominal frames assigned to the test
frame's current stage. Calibration uses the identical restriction. This changes
the detector and threshold calibration, so build it under a new
`vision_memory_name`; an older unconstrained memory is intentionally rejected.

Memory construction and testing are deliberately separate:

```bash
# Uses nominal_bags and optional dino_calibration_bags; no test is run.
docker compose run --rm vision-memory

# Uses test_bag plus an existing named memory; nominal_bags are not read.
# This command never builds or rebuilds memory.
docker compose run --rm vision-test
```

After changing Python source code pulled from Git, rebuild the shared pipeline
image once with `docker compose build vision-memory`. Both services then use the
same updated image. Keep only one top-level `output_dir` entry in the JSON.

### DINO calibration on separate successful recordings

The optional top-level `dino_calibration_bags` list enables a sequence-aware
threshold inspired by [FIDeL, section III-E](https://arxiv.org/html/2604.13788v1#S3.SS5).
Keep `nominal_bags` as retrieval memory. Calibration bags must be separate
successful executions, and evaluation bags must be separate from both lists.
Replay disables memory and calibration bags in the evaluation selector.
This affects DINO only; LoRA, classifiers and GMR do not need retraining.

The existing CLS shortlist and attention-weighted patch distance are retained.
Calibration uses that exact same matcher, camera, sampling and stage restriction.
It does **not** implement the paper's optimal transport or patch-level statistical
test. It adapts the paper's independent nominal split and maximum-deviation idea
to the existing scalar frame score, keeping inference cost unchanged.

Whole calibration recordings are partitioned deterministically (seed 42): half
estimate a mean/standard-deviation curve against matched nominal progress; half
set the bound from each recording's **largest standardized deviation**. The
upper order statistic uses `ceil((N + 1) * 0.90)` for N limit recordings, with
the existing 1.10 margin on its bandwidth. At least 18 calibration bags are
required. The mean and scale are smoothed before maxima are calculated; the
final calibrated bound is not smoothed down afterwards. Equal recording weights
within each progress bin prevent long recordings from dominating the shape.

The old 90th percentile was over frames, which does not control alarms over an
entire execution. The new sequence quantile is still **not a promised real-world
false-alarm rate**: shifted conditions, correlated recordings, longer executions
or changed sampling can invalidate exchangeability. Evaluate both normal and
failure bags afterwards, including missed failures and detection delay.

For the lab dataset, this one-time command edits your existing config in place
(no second config). It reserves setting 1 runs 16–30 and setting 2 runs 1–11 for
calibration, keeping setting 1 runs 31–32 and the additional evaluation runs out.
Check that both settings are mounted under `/data/Nominal` in Docker.

```bash
python3 - <<'PY'
import json
from pathlib import Path
path = Path('pipeline_config.json')
config = json.loads(path.read_text())
config['dino_calibration_bags'] = (
    [f'/data/Nominal/setting 1/nominal_1.{i}' for i in range(16, 31)]
    + [f'/data/Nominal/setting 2/nominal_2.{i}' for i in range(1, 12)]
)
config['dino_threshold_scale'] = 1.0
path.write_text(json.dumps(config, indent=2) + '\n')
PY
docker compose run --rm --build vision-memory
docker compose up --build online-replay
```

Stop the running replay server before rebuilding calibration. With
`rebuild_nominal_cache: false`, compatible memory features are reused; only the
new calibration bags are processed. Each query bag is released before the next
one, and calibration never adds queries to retrieval memory. The existing
`calibration.csv` records shape/limit roles, standardized deviations and bound
crossings; `adaptive_threshold.csv` records the mean, scale, bandwidth and run
counts. These are calibration diagnostics, not independent evaluation metrics.
Rerunning an unchanged build reuses the complete cache.

`dino_threshold_scale` is the single sensitivity adjustment: `1.0` uses the
calibrated curve, `1.1` raises it by 10%. Higher values reduce alarms but can miss
or delay failures; lower values do the opposite. It multiplies only the decision
limit, not the distance or heatmap. Restart replay after changing it; no memory
rebuild is needed. Batch tests, heatmap-video alarms and GUI replay apply the same
multiplier. Leave it at 1.0 for the first independent evaluation. An empty or
absent calibration list preserves compatibility with existing caches/legacy
calibration; the GUI note reports which calibration was actually loaded.

### DINO local-region scoring for small object mismatches

Set the optional top-level `"dino_score_mode": "local"` to enable local scoring.
The default `"mean"` retains the previous whole-image attention-weighted score.
Local mode uses the same DINO model, normalized patch features, CLS shortlist,
stage restriction and reference-frame selection. No new network is loaded.

Why this change: at input size 728, a frame has 52 x 52 = 2,704 patch tokens.
A missing or misplaced small object can be diluted by averaging all patch
differences. DINOv2 supplies spatial patch features, but CLS attention is not a
task-specific object mask. Its authors document attention/feature artifacts in
models without registers; increasing attention power does not guarantee emphasis
on the manipulated object. See the official
[DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
and [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588).

After matching a nominal frame, local mode:

1. Computes unweighted patch distances at the same image locations. Attention
   still participates in the existing frame matcher, but cannot suppress a
   region's final mismatch score.
2. Averages each overlapping 3 x 3 patch neighborhood. This limits whole-image
   dilution and reduces isolated patch artifacts; it is not object segmentation.
3. Learns a separate nominal mean/scale for every neighborhood versus matched
   nominal progress. Normal robot motion therefore does not directly set the
   variance of a quiet object location.
4. Uses separate limit recordings to calibrate the maximum standardized
   deviation across **all regions and all sampled times** in each recording.
   The final region bound is `mean + margin * bandwidth * scale`. This remains
   a conservative sequence-level limit; missed failures must still be measured.
5. Selects the region with the largest distance / bound ratio. The graph shows
   that region's raw distance and its bound, both in the same units. The region
   can change between frames; the saved row includes its coordinates, the old
   global score, and the threshold multiplier.

The GUI heatmap uses a fixed ratio scale: 0 is blue, 1 (the decision limit) is
approximately green, and 2 or more is red. A white rectangle marks the selected
region. Colors are no longer stretched independently for every frame. A red
area in the old heatmap only indicated a relatively large difference in that
frame, not a calibrated alarm.

Keep the current separate `nominal_bags` and `dino_calibration_bags` lists. Using
the same recording as query and memory permits exact self-matches, leading to
artificially small distances/thresholds. Using all recordings is possible with
whole-recording cross-fitting, but that calibrates against a different reference
set than the final full memory and is not automatically more accurate. Given
the available independent data, we keep the separate split. More *diverse*
successful memory runs may help, but they require fresh calibration and must
not also serve as calibration/evaluation runs. No bag-list changes are made by
this update.

To use local mode, stop replay, pull the update, and manually set:

```json
"dino_score_mode": "local",
"dino_threshold_scale": 1.0
```

Then run:

```bash
docker compose run --rm --build vision-memory
docker compose up --build online-replay
```

With `rebuild_nominal_cache: false`, compatible memory features are reused.
Calibration bags must be processed again because older CSV files contain only
whole-image scores, not spatial patch differences. The compact regional profile
is saved automatically as `local_threshold.npz` beside the existing memory;
there is no extra config path. Cache signatures prevent mixing mean/local
thresholds. Returning to mean mode also requires recalibration of that cache.
`adaptive_threshold.csv` contains the mean regional bound as a **summary only**;
per-frame CSV/JSON results contain the actual selected-region bound used for
the alarm. `calibration.csv` retains per-run maxima and selected-region scores
for checking whether a particular successful recording inflated the limit.

This change does not recognize the label "scissors", infer grasp intent, repair
incorrect stage alignment, or guarantee detection under occlusion. It makes
localized visual differences available to the decision. Validate it on separate
successful and failed recordings before drawing accuracy conclusions. Synthetic
tests check small-region sensitivity, location dependence, normal-motion
variation, calibration separation, multiplier parity and replay/batch agreement.

Every rosbag vision run writes two raw videos per camera:

- `<camera>_model_input.mp4`: the exact square input processed by DINO.
- `<camera>_raw_original.mp4`: recorded resolution/aspect ratio for human review
  and RynnBrain. RynnBrain prefers this file automatically.

The bag paths above are container paths underneath the host data folder from
`.env`. The output subfolder is created underneath the host output folder.

### Reuse nominal DINOv2 memory

The first full run saves one reusable cache per camera underneath
`nominal_cache_dir`. Later runs with a different `test_bag` and `output_dir`
load that cache and skip nominal feature extraction and adaptive-threshold
calibration. Keep `rebuild_nominal_cache` set to `false` for normal reuse.

The cache signature includes nominal bag paths, camera topic, model and input
size, sampling, stage configuration, and threshold parameters. If any of those
change, the runner automatically rejects the old cache and rebuilds it. Set
`rebuild_nominal_cache` to `true` to force a rebuild manually, then return it
to `false` afterward. Changing only the test bag or test output directory does
not invalidate the nominal cache.

The default image installs CPU PyTorch and works without an NVIDIA GPU. Later
builds reuse Docker's cache when dependencies have not changed.

### Run configured ROS bag processing

No paths are required in the run command:

```bash
docker compose run --rm rosbag-vision
```

Generate or refresh normal/fail labels from every bag under the configured
data directory:

```bash
docker compose run --rm bag-labels
```

The clean label file is written to `dataset_labels.csv` in the configured host
output directory. `dataset_labels_details.csv` shows timestamped startup,
earlier, and considered markers separately for debugging. By default, stage
messages earlier than 0.1 seconds are treated as cached startup history. Of the
remaining messages, only the final three affect the label.

### Run without GPU

Both sections:

```powershell
docker compose -p anomaly-detection run --rm pipeline
```

Only vision or time series:

```powershell
docker compose -p anomaly-detection run --rm pipeline vision
docker compose -p anomaly-detection run --rm pipeline time-series
```

The Hugging Face model cache is stored in a Docker volume, so DINOv2 is not
downloaded again for every run.

## RynnBrain VLM experiments

RynnBrain is an optional third pipeline with its own GPU image and dependencies;
it does not modify the DINO/time-series container. It reuses the existing
`pipeline_config.json`; copy the `rynnbrain` section from
`pipeline_config.example.json` into your local configuration.

Evaluation is multi-turn and always uses visual memory, not a saved text
description: turn 1 shows the model the nominal demonstration frames sampled
directly from `rynnbrain.reference_bags` (via `rynnbrain.memory_camera_topics`),
and turn 2 shows the test bag's frames and asks for a decision against what it
just saw. There is no separate "build memory" step to run first.

By default, `source` is `generated_videos`, so the turn-2 test frames come from
the existing `<camera>_raw_original.mp4` / `<camera>_heatmap.mp4` files in the
configured vision `output_dir`. It does not rerun DINO. Set `source` to
`rosbag` only when direct bag sampling is specifically wanted for the test side.

The configuration selects the checkpoint, number of uniformly sampled time
steps, and input modes.
Supported modes are `raw`, `heatmap`, and paired `raw_heatmap`.
`rynnbrain.memory_camera_topics` selects nominal-demonstration viewpoints, while
`rynnbrain.camera_topics` independently selects test viewpoints.
The total visual load is approximately `num_frames x number_of_cameras`, or
twice that for paired raw/heatmap input.

Build the shared GPU image once:

```bash
docker compose build rynnbrain-test-multiturn
```

Run it against the configured `test_bag`:

```bash
docker compose run --rm rynnbrain-test-multiturn
```

It prints each prompt and unmodified response in the terminal, including the
intermediate response after the nominal turn. `rynnbrain_responses_multiturn.json`
stores the task description, both responses, and an exact per-turn manifest of
the image labels and sizes sent to the model.
Selected model inputs, parsed decisions, confidence, full responses, and frame
timestamps are saved under `output_dir`. Each input mode also contains
`vlm_input_storyboard.jpg`, showing the exact images and order sent to the VLM,
plus a `nominal/` folder with the reference-bag frames shown in turn 1.
`selected_vlm_frames.csv` records the source video, exact frame index, timestamp,
FPS, and camera topic. Use `rynnbrain.sampling_start_sec` and
`rynnbrain.sampling_end_sec` to exclude stale frames before or after the actual
task while keeping raw and heatmap selection aligned.

### CoP last-layer vectors and PCA

The pipeline exports one fixed-size vector for every tested bag and input mode
by default; set `rynnbrain.cop_vectors.enabled` to `false` to turn it off.
RynnBrain-CoP uses Chain-of-Point training, but the checkpoint does not expose a
separately named "CoP embedding." This pipeline therefore uses a precise,
repeatable representation: the final
normalized language-decoder state at the last non-padding token of the complete
turn-2 prompt, captured immediately before the test answer is generated. It
contains the nominal-reference and test context, but not generated decision
words that could leak the predicted class into the PCA.

Each evaluation writes:

- `cop_vectors/<input_mode>.npy`: the raw one-dimensional float32 vector.
- `cop_vectors/<input_mode>.json`: its bag label, model/configuration signature,
  vector definition, dimension, decision, and paths.

At the end of a benchmark, all compatible saved vectors below the benchmark
directory are L2-normalized, centered, and fitted jointly with PCA. A separate
PCA is fitted for each input mode so differences between `raw` and
`raw_heatmap` do not masquerade as failure separation. Ground-truth `normal`
points appear as **Nominal**, `fail` points as **Failure**, and `unknown` points
are excluded. Nominal bags use circles; each failure category, read from its
existing `Failure_<number>_<description>` parent folder, has its own marker
shape and color. Categories share the same styling across input modes within
an analysis. The legend sits outside the plot and includes category counts.
Individual bag-name annotations are omitted to keep dense clusters readable;
bag names and categories remain available in the coordinate CSV. Failure bags
without a category folder are grouped as `uncategorized_failure`.
The `cop_pca/` directory contains, per input mode:

- `cop_vectors_<mode>.npz`: aligned raw vectors, labels, names, and paths.
- `cop_pca_<mode>.csv`: PC1/PC2 coordinates, category, and name for every plotted bag.
- `cop_pca_model_<mode>.npz`: fitted PCA axes, feature mean, explained
  variance, and preprocessing identifier.
- `cop_pca_<mode>.png`: scatter plot with distinct category markers and a legend.
- `cop_pca_summary.json`: counts, explained variance, output paths, or the
  reason a plot was skipped.

Prefer held-out `normal` bags for the nominal PCA class. By default, bags in
`nominal_bags` or `rynnbrain.reference_bags` remain excluded because evaluating
reference-memory data can make separation look artificially easy. To run only a
small explicit set, repeat `--bag` with eligible bag directory names and include
at least one held-out normal and one failure bag (several of each is better):

```bash
docker compose run --build --rm benchmark \
  --config /config/pipeline_config.json \
  --bag held_out_normal_bag \
  --bag failure_bag
```

When recorded stage labels are unreliable, select and label each bag explicitly
with repeatable `--normal-bag` and `--failure-bag` options. These manual labels
override the recorded label and are used for accuracy, vector metadata, and PCA:

```bash
docker compose run --build --rm benchmark \
  --config /config/pipeline_config.json \
  --normal-bag normal_bag_01 \
  --failure-bag failure_bag_01 \
  --failure-bag failure_bag_02
```

For an explicitly exploratory plot, `--include-nominal-bags` allows bags from
the DINO `nominal_bags` list to be selected with `--bag`. This is not a held-out
evaluation, especially for heatmap-based modes. A bag in
`rynnbrain.reference_bags` is always excluded because it is already shown in
turn 1.

The benchmark creates the PCA automatically. To recreate plots later from the
already-saved vectors without loading RynnBrain or processing bags again:

```bash
docker compose run --rm --entrypoint python benchmark \
  -m rynnbrain_vlm.cop_analysis \
  --input-dir /outputs/experiments/example/rynnbrain_benchmark
```

### LoRA fine-tuning for generated success/failure decisions

The optional LoRA trainer adapts the VLM's **generated decision**, using labeled
nominal and failure executions. It freezes the original RynnBrain weights and
vision encoder and trains small adapters in the language decoder's attention
projections (`q_proj` and `v_proj` by default). It uses the existing multi-turn
prompts: nominal reference images and a frozen-base nominal response, followed
by the execution images. Only the target `Decision: success` or
`Decision: failure` and the end-of-answer token contribute to cross-entropy
loss. Prompts, images and the nominal response are masked from the loss. No
visual-evidence or reasoning annotations are invented from binary labels.
This follows the standard [PEFT LoRA API](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/lora).

**Choose the bags yourself in `pipeline_config.json`, under
`rynnbrain.lora.training`:**

```json
"normal_bags": [
  "/data/Nominal/setting 1/nominal_1.2",
  "/data/Nominal/setting 1/nominal_1.3"
],
"failure_bags": [
  "/data/Failure_1_grasp_miss/failure_1.1",
  "/data/Failure_2_slip_at_start/failure_2.1"
],
"validation_normal_bags": [],
"validation_failure_bags": []
```

These are examples only; the committed lists are empty. List membership is the
explicit ground truth: `normal_bags` means success and `failure_bags` means
failure, regardless of recorded-stage markers or previous VLM predictions.
Full container paths, data-root-relative paths, and unique bag names are
accepted. Training requires both classes. For validation, either supply both
classes in the validation lists or leave both lists empty. Duplicate bags,
overlap between training and validation, and selecting a nominal reference
demonstration as a supervised sample are rejected. Keep a further test set out
of both lists for the final benchmark.

Validate your selection without loading the model:

```bash
docker compose run --build --rm rynnbrain-lora-train \
  --config /config/pipeline_config.json --validate-only
```

Then train explicitly:

```bash
docker compose run --rm rynnbrain-lora-train
```

With the current `source: "rosbag"` and `input_mode: "raw"`, images come directly
from the selected ROS bags using the same sampling and camera ordering as
inference; a benchmark run is not required first. For `generated_videos` or
heatmap input modes, generate the selected bags' DINO outputs first. Training
reads `<training.benchmark_dir>/<bag_name>/<camera>_raw_original.mp4` and/or
`<camera>_heatmap.mp4`; `benchmark_dir` defaults to the existing
`<rynnbrain.output_dir>_benchmark`. Global per-video overrides are rejected for
multi-bag training. Nominal images always come from `reference_bags`.

The trainer uses one GPU, one bag per forward pass, and gradient accumulation
(8 bags per optimizer update by default; the final smaller group is handled
separately). Each successful optimizer step updates only the LoRA weights.
`class_weight: "balanced"` gives nominal and failure training samples equal
total weight; `"none"` uses ordinary per-bag loss. Validation loss is unweighted.
Training defaults are rank 8, alpha 16, dropout 0.05, 3 epochs and learning rate
0.0001. The rank and projection settings live directly under `rynnbrain.lora`;
the other settings live under its `training` section.

Training uses bfloat16 and gradient checkpointing by default. Set
`training.precision` to `"float16"` for a GPU without bfloat16 support; float16
uses gradient scaling. Training loads the full base model on the GPU, so it
needs more memory than inference with CPU offloading. It does not use
quantization or the inference `max_memory` setting. Reduce `num_frames` or
`model.max_image_size` if needed. Overlong examples are rejected with their
token count; images and answers are never silently truncated. The configured
`generation.do_sample` must be false for repeatable nominal context.

With `training.output_dir: null`, training saves beneath
`<rynnbrain.output_dir>/lora_adapter/`:

- `adapter_model.safetensors` and `adapter_config.json`: only the learned adapter.
- `training_manifest.json`: exact bag labels/splits, settings and nominal response.
- `loss_history.csv`: training loss and whether each optimizer update succeeded.
- `training_summary.json`: epoch training/validation losses and saved epoch.
- `training_frames/`: resized, lossless execution frames reused across epochs.

With validation, the adapter with the lowest validation loss is kept; otherwise
the final epoch is kept. A new run requires an empty output directory, so set
a different `training.output_dir` for another experiment. Training starts from
the base checkpoint; optimizer-state resumption is not implemented.

To evaluate the trained adapter, set **`rynnbrain.model.lora_adapter_path`** to
the printed adapter directory, then use the existing evaluation or benchmark
commands. For the current configuration, the default directory is
`/outputs/experiments/pick_place_dino728_v102/rynnbrain/lora_adapter`.
Keep this setting `null` for the original model and before training a new
adapter. Training is never triggered by evaluation. The classifier can remain
disabled throughout this workflow.

During adapter evaluation, the nominal turn uses the frozen base model; the
adapter is enabled for the execution decision turn. The benchmark automatically
excludes the adapter's training bags, even with `--include-nominal-bags`, and
single-bag evaluation also rejects training bags. Adapter weights are
fingerprinted in the CoP comparison signature, so PCA and saved classifiers
cannot silently combine base-model features with adapted-model features.
Existing paths and bag names are unchanged. Use a separate benchmark output
directory to retain both base and adapted results for comparison.

The adapter is trained on `Decision: success` or `Decision: failure`, followed
by the end-of-answer token. A decision without an explanation is therefore
expected; increasing `max_new_tokens` does not force it to continue.
For an optional description of the visible evidence, add `"explain_decision": true`
inside the existing `rynnbrain.generation` object, then run:

```bash
docker compose run --rm --build rynnbrain-test-multiturn
```

This performs one extra generation on the same reference and observation images
using the original model with the adapter temporarily disabled. It is an
independent visual description, **not the adapted model's own explanation or
confidence**, and may disagree with its decision. The adapted decision is not
shown to this pass. The decision response and CoP vector remain unchanged;
`visual_evidence`, its source, and any generation error are saved separately in
the existing result files. The option defaults to false and needs no retraining.

Adapter loading preserves the base model's device map instead of automatically
recalculating it after the base weights have already occupied GPU memory.
Startup prints the preserved placement and input execution device. CPU offload
may still be necessary when the base model does not fit in available VRAM.
Inference follows the input embedding's actual execution device after adapter
loading, including Accelerate offload hooks. A stale `input_device: "cuda"`
setting no longer sends tokens to CUDA when the model has been placed on CPU.
The resolved device is printed at startup. If it reports CPU unexpectedly,
check `nvidia-smi` on the lab PC for available GPU memory; CPU inference is slow.

### Optional LoRA supervision for failure onset and both camera views

This is a new, opt-in training mode. Existing decision-only adapters and commands
still work. No configuration file, bag list or experiment path is changed for you.

Make these edits manually in your current `pipeline_config.json`:

1. Set **both** `rynnbrain.camera_topics` and `rynnbrain.memory_camera_topics`
   to this same ordered list (the top-level DINO `camera_topics` can stay as it is):

   ```json
   [
     "/top_camera/cam_135222070433/color/image_raw/compressed",
     "/flange_camera/cam_230422272745/color/image_raw/compressed"
   ]
   ```

2. Under `rynnbrain.lora.training`, add `"temporal_supervision": true`.
   Populate its `normal_bags` and `failure_bags` lists; optional validation lists
   must also contain both classes. Keep reference demonstrations and final test
   bags separate from training and validation.
3. Keep `rynnbrain.source: "rosbag"`, `rynnbrain.input_modes: ["raw"]`,
   `lora.training.input_mode: "raw"`, and your existing `num_frames: 8`.
   Both cameras must exist in all selected bags, including reference bags.
4. Set `rynnbrain.model.lora_adapter_path` to `null` for training a **new** adapter
   from the base checkpoint. Your existing v102 adapter files are retained.
   With `lora.training.output_dir: null`, this mode saves to
   `<rynnbrain.output_dir>/lora_temporal_adapter`. An explicit training output
   directory overrides that default and must be empty.

Then run these commands in order:

```bash
docker compose run --rm --build rynnbrain-lora-train \
  --config /config/pipeline_config.json --validate-only
docker compose run --rm rynnbrain-lora-train
```

In this mode, validation also reads the camera streams and recorded error-button
annotations without loading the VLM. A failure bag must have an eligible marker
on `stage_topic` (normally `/recording_stage`); a nominal bag must not have one.
The existing startup-marker exclusion applies. Missing/unreadable or conflicting
annotations, missing camera frames, and failure markers outside the sampled
camera window stop training with an error rather than silently changing labels.

Each image is labeled with its camera topic and actual timestamp in seconds from
that bag's start, using the same clock as the error-button annotation. The
existing per-camera uniform sampling is retained; cameras are not assumed to
be synchronized. Eight timesteps produce 16 execution images across two cameras,
plus 16 reference images for one reference bag. This mode does not add more
sampled timesteps.

Each training bag supplies eight examples: its first camera pair, then its first
two pairs, and so on. An example contains only images available by its cutoff.
Before the recorded failure, its answer is:

```text
Decision: success
Failure onset (s): none
```

At and after the recorded failure, its answer is, for example:

```text
Decision: failure
Failure onset (s): 12.350
```

Here success means **no failure observed yet**, even if the task is unfinished.
The failure label stays positive after a later recovery. The recorded onset
appears only in the supervised answer, never in the input prompt. All prefixes
of a bag stay in its original training or validation split. Balanced weighting
uses the resulting prefix labels; gradient accumulation counts prefix examples.

Training applies causal cross-entropy to the decision, onset text and end token;
input/context tokens are masked from the loss. Only the configured LoRA language
attention weights change; base and vision weights remain frozen. The numeric
time is learned as answer tokens, not with a separate numerical regression loss.
The manifest records the camera/time protocol, annotations and prefix targets.
Two views and multiple prefixes require more GPU memory and training time than
decision-only training. Overlong inputs are rejected without truncation.

After training, set `rynnbrain.model.lora_adapter_path` to the **new directory
printed by the trainer**. Retain the same two camera lists and frame count.
Loading the adapter automatically selects the timestamp-aware prompts and checks
those input settings against its manifest. Changing only the training flag does
not convert an existing adapter.

The changed adapter and camera inputs also change CoP features. For a single test
with the classifier enabled, retrain the selected classifier first:

```bash
docker compose run --rm cop-classifier-train
docker compose run --rm rynnbrain-test-multiturn
```

Alternatively, `docker compose run --rm --build benchmark` performs its existing
automatic classifier preparation. Use your chosen new output paths to retain old
results. kNN, logistic regression, their timelines and their alert-time metrics
remain available; old classifier features cannot silently match the new adapter.

The temporal VLM's response is retained in the existing per-bag files. Per-bag
results and benchmark summaries additionally contain `vlm_failure_onset_sec`,
`vlm_failure_onset_status`, `vlm_failure_time_error_sec` (prediction minus marker)
and `vlm_failure_time_absolute_error_sec`. Missing/invalid predictions leave
errors blank. These are separate from classifier threshold-crossing times and
their existing timing graphs. A time estimated from the full bag is retrospective;
it is not the time at which an online system raised an alert. This supervision
can help localization, but sparse frames and button timing still limit accuracy;
evaluate improvement on bags held out from both training and validation.

### One experiment output directory and shared cameras

The RynnBrain training, test, classifier-training and benchmark commands now
resolve missing paths from the top-level `output_dir`, without rewriting your
JSON. Keep that experiment directory and one top-level `camera_topics` list.
Remove these redundant keys from your personal config to use inheritance:

- `rynnbrain.output_dir`
- `rynnbrain.camera_topics` and `rynnbrain.memory_camera_topics`
- `rynnbrain.lora.training.output_dir`
- `rynnbrain.cop_classifier.training.input_dir`
- `rynnbrain.cop_classifier.model_paths`

For an experiment root such as `/outputs/experiments/pick_place_dino728_v130`,
the resulting defaults are:

| Setting | Relative to the experiment root |
| --- | --- |
| Single-test outputs | `rynnbrain/` |
| Benchmark and classifier-training vectors | `rynnbrain_benchmark/` |
| Temporal LoRA training output | `rynnbrain/lora_temporal_adapter/` |
| Ordinary LoRA training output | `rynnbrain/lora_adapter/` |
| Logistic classifier, raw mode | `cop_classifier/raw_logistic.npz` |
| kNN classifier, raw mode | `cop_classifier/raw_knn.npz` |

Explicit overrides still take precedence. An omitted or null individual output
path selects its default; an omitted camera list inherits. Reference cameras
inherit the VLM camera list, which inherits the top-level list. If you retain an
old explicit path, changing the experiment root will **not** override it. LoRA
and classifier training bag lists remain separate and are never inferred from
one another. The shared DINO nominal cache stays at its configured location.

`rynnbrain.model.lora_adapter_path` remains explicit: set it to null while
training a new adapter, then set it to the path printed by training before
evaluation. It is never selected automatically from the experiment folder.
With `temporal_supervision: true`, a single top-camera topic still trains
timestamped execution prefixes and failure-onset answer targets from the
recorded error marker. `inference.feature_only` does not disable this training.
Keep `cop_classifier.plot_timeline: true` to evaluate classifier onset times.

To change the output folder for each evaluation **without retraining the
classifier**, keep its artifact path fixed and disable benchmark training:

```json
"cop_classifier": {
  "enabled": true,
  "method": "logistic",
  "train_on_benchmark": false,
  "model_paths": {
    "raw": "/outputs/experiments/pick_place_dino728_v130/cop_classifier/raw_logistic.npz"
  }
}
```

Merge these keys into your existing classifier object, retaining its other
settings. Now the top-level `output_dir` can be `v131`, `v132`, etc., while the
LoRA path and classifier path continue pointing to the trained v130 artifacts.
The benchmark loads the classifier without extracting training vectors, fitting
weights, or writing its model files. It reuses the exact nominal-reference text
saved with that classifier and verifies model/adapter and representation
settings before evaluating bags. Missing or incompatible classifiers cause an
error, never an automatic replacement. The `.json` sidecar must remain beside
the `.npz`; any trained timeline/onset models are also loaded from that original
classifier directory. New test vectors, reports and timelines go to the new
experiment folder. This applies to kNN too: select `method: "knn"` and its saved
`raw_knn.npz` file. Saved thresholds and weights remain in effect.

`train_on_benchmark` defaults to true for compatibility with the automatic
first-run training workflow. The explicit `cop-classifier-train` command still
trains when this flag is false; `benchmark --prepare-classifier` also explicitly
overrides it. Both can replace the configured classifier artifact, so choose a
new artifact path when intentionally training a separate version. Setting
`enabled: false` disables scoring entirely; it is not the reuse option.

### Online replay interface (eight observations)

The separate replay application keeps RynnBrain, the selected LoRA, the saved
classifier, and a nominal-reference KV cache alive between executions. It reads
one camera from a recorded bag on its original timestamp clock. A separate
inference worker appends selected frames to the execution cache, extracts the
existing final-normalized prompt-end vector, and updates a browser video/score
view. It stops at `rynnbrain.num_frames` (at most eight), or at the end of the
bag. This first version is recorded-bag replay, not a live ROS subscriber.

No config changes are required. Use your existing trained LoRA and classifier
paths. This command **never trains** either model, regardless of
`train_on_benchmark`. It does not run DINO. Start the service on the lab PC:

```bash
docker compose up --build online-replay
```

On the lab PC open `http://localhost:8765`. From a different PC, forward port
8765 using VS Code's Ports panel, or establish an SSH tunnel:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18765:127.0.0.1:8765 steribot@172.17.21.63
```

For this SSH command, open **http://127.0.0.1:18765** on Windows. The server
still uses **8765** on the lab PC; **18765** is only the local tunnel entrance.
If using VS Code forwarding instead, open the actual forwarded address shown
in its Ports panel. Windows localhost is not the lab PC
unless the tunnel is running. Wait for **ready**, select a bag, set the interval
and processing mode in the browser, and click **Start execution**:

- **Cached only (fast)**: default; no uncached pass, checks shown as not checked.
- **Diagnostics (compare all observations)**: checks every observation and uses
  the uncached score if a check fails, retaining both results for investigation.
- **Strict verification (stop on mismatch)**: stops at the first failed check.

Settings apply to the next execution and stay selected for this server session;
they do not edit the config or retrain/reload the model. After restarting the
service, the defaults are cached-only replay and a 10-second interval.
The selector automatically lists the configured `test_bag` first, followed by
ROS bag directories containing `metadata.yaml` under the mounted data folder
(`ANOMALY_DATA_ROOT`, default `/data`). **Keep `test_bag` as a single string**;
no new config list is needed. Training/reference bags are disabled after the
model loads, using the saved adapter/classifier training manifests; the backend
also rejects them. Bags used only for DINO memory are not excluded for that reason.
The scan reads directory names, not image streams, and does not run a benchmark.
Restart the service to discover newly added bags. Advanced CLI usage supports
repeated `--bag /data/path/to/bag` arguments to restrict the selector explicitly.
**Stop** takes effect between model updates; it cannot cancel a GPU kernel.
Leave the terminal open to retain the model. Ctrl+C shuts down after the current
update and saves its results. For a Compose-managed service, allow an in-flight
model call time to finish when stopping: `docker compose stop -t 120 online-replay`.

If the page is blank, test the server in a second **lab PC** terminal:

```bash
curl --max-time 5 http://127.0.0.1:8765/state
```

If this returns `"status": "ready"`, the model is waiting for the browser's
Start button; check SSH/VS Code port forwarding. If it returns `loading`, the
JSON and browser show the initialization phase. If it fails, inspect
`docker compose logs --tail 80 online-replay`. Startup failures also print their
traceback. The HTTP page is served independently of model initialization.

The browser's interval (default 10 seconds) selects observations at least that far apart, starting
at `sampling_start_sec`. Selection uses only frames that have already arrived;
it does not look at the bag end or error-button annotation. If inference takes
longer, the worker uses the latest available eligible frame. The producer uses
up to 15 camera frames/s for preview, with a one-frame mailbox, so inference
never accumulates a queue of old frames. The eight selected observations may
cover only part of a long bag. Change the interval deliberately for your task;
it is not automatically chosen from the bag duration. There is no rolling
window or cache eviction in this version.

With strict verification (also available as `--verify-cache`), every selected prefix is also evaluated by the original
uncached feature extractor. The application saves both vectors, relative L2
error, cosine similarity, score difference, and separate feature timings. It
stops on relative L2 error above `1e-3`, score difference above `1e-3`, or a
changed threshold decision. Verification increases observed latency; after it
passes on your real adapter/hardware, omit `--verify-cache` for the normal fast
path. Local regression coverage uses a small real Qwen3-VL with active LoRA,
multi-token images, varying image grids, and DeepStack features; it is not a
speed or accuracy measurement on the lab GPU.

If strict verification stops, select **Diagnostics** in the browser (CLI:
`--diagnose-cache`) to investigate all available observations (up to eight). This mode retains the
same limits and checks every prefix. On a mismatch it displays/saves the
**uncached** score and vector, while retaining the cached candidate separately
as `frame_*_cached.npy`. Failed checks remain failed: the JSON has
`cache_verification_status: "failed"` even if replay completes. Each row records
`score_computation`, the separate feature/score/decision checks, and both scores.
The browser explicitly labels fallback rows and reports the failed limits.
This mode collects diagnostics; it neither fixes nor approves a mismatching
cache, and is not a measurement of cached-only end-to-end latency. Missing,
invalid, or non-finite baseline outputs still stop the run. Config, classifier,
LoRA and tolerance values are not changed.

The reference text is reused from the saved classifier. Its KV cache is built
with the active inference LoRA, matching the existing second-turn computation;
the base-model reference-generation cache is never substituted. Each new image
alone goes through the vision encoder. The execution-frame KV history is
retained, while temporary evaluation-instruction states are cropped after each
score. Image coordinates and masks are derived from the same full prompt.
Current prompt tokenization/image preprocessing still prepares the selected
history; the expensive old-image vision/decoder computations are reused. A new
execution gets a fresh copy of the fixed reference cache. Weights remain fixed.

Logistic scoring uses the saved onset classifier when `logistic_onset.enabled`
is true, otherwise the saved complete-execution classifier. kNN uses the saved
prefix memories/thresholds by observation index; its timeline companion must
exist. **These scores are diagnostic:** elapsed-time live sampling differs from
the completed-bag sampling used for existing training. A kNN observation index
is not validated task-progress alignment, and a logistic score is not a
guaranteed calibrated onset probability. Cache equivalence checks verify the
computation for identical inputs, not online detection accuracy. No automatic
classifier retraining or threshold adjustment is performed.

Both logistic regression and kNN work with incremental cached observations.
The selected classifier must match the feature extractor that produced its
training vectors: LoRA identity, reference, camera inputs, prompts and other
recorded feature settings. A mismatch now lists each differing field and its
saved/current values. Experiment folder numbers need not match, but switching
to an unrelated saved classifier does not make its feature space compatible.
The application keeps this check strict for both methods.

kNN additionally needs the matching `raw_knn_timeline.npz` and JSON companion
beside `raw_knn.npz`. It compares each observed prefix with nominal memories
for the same observation count; it does not wait for all eight test images.
To prepare a new compatible kNN memory, choose a **new** `model_paths.raw`
destination, keep `method: "knn"` and `plot_timeline: true`, then run
`docker compose run --rm --build cop-classifier-train`. This uses the current
LoRA and nominal training list and prepares the prefix memories too. Replay
never invokes training, even if `train_on_benchmark` is true. Stop the replay
service before training, then restart it to load the new artifacts.

The current kNN threshold is a training diagnostic: at each prefix length,
leave out one nominal bag, calculate its mean distance to its k nearest other
nominal vectors, and take the configured quantile of these distances (default
95%, using the higher observed quantile and a minimal upward floating-point
adjustment for ties). Each prefix has its own threshold. Replay loads those
saved thresholds; changing `threshold_quantile` in the config does not recalibrate
an existing artifact. This does not guarantee 5% false alarms on unseen bags or
across a complete sequence. Offline prefixes were sampled over complete bag
durations; replay prefixes use elapsed-time intervals, so observation count
does not ensure the same task progress. Calibration for replay needs separate
nominal executions processed with the intended replay sampling, and evaluation
on other held-out executions. Do not choose a threshold from a single failure
test graph.

`benchmark` and `online-replay` are independent entry points. The benchmark
processes its selected bag set and exports combined tables/plots. The GUI
processes only the bag you start, saving each execution under `online_replay/`.
It does not launch or monitor a benchmark, nor automatically combine its results.

The GUI's **Find similar nominal frame** switch displays the exact scored observation beside
its closest nominal image in the recorded stage. This is an experimental
retrieval preview; it does **not** replace the CoP classifier memory, change the
anomaly score, or recalibrate thresholds. A history-dependent CoP vector cannot
simply be replaced by an independent image descriptor.

The descriptor is the float32 mean of one image's Qwen3-VL visual-merger tokens,
L2-normalized for cosine comparison. It is not a learned CLS token. Replay
captures these tokens from the existing new-image forward pass, so it adds no
second vision/VLM pass per observation. Retrieval quality still needs checking
on your images. The preview reports the selected stage, nominal bag/time,
cosine distance and candidate count. Lower distance means greater visual
similarity, not a calibrated failure probability.

On the first execution with this preview enabled, the worker prepares up to
eight nominal images **per stage per nominal bag**, using the existing
`cop_classifier.training.normal_bags` list, or `reference_bags` if that list is
empty. It encodes these images once without language decoding. The page remains
responsive and shows preparation progress; Stop also cancels preparation.
Memory is reused in the session and cached under
`<output_dir>/online_replay/visual_memory/`. Recording identity, stage annotations
and feature settings invalidate stale cache entries. This cache is separate
from the trained kNN/logistic artifacts. No new config paths are required.

The preview uses corrected stage annotations from the recorded bag, so it is
retrospective alignment, not a live stage estimator. Unknown stages or missing
nominal candidates are reported instead of searching a different stage. Replay
continues with the original classifier if preview preparation is unavailable.
Specifically, it takes the final three **numbered** messages and requires
`1, 2, 3` in increasing timestamp order. Sequences such as `1, Error, 2, 3` and
`1, 2, Error, 3` have the same stage-boundary rule: Error/Fail does not advance
the stage. Earlier numbered markers are ignored. Frames before the corrected
stage-1 start have unknown stage; missing or invalid boundaries disable the
preview instead of borrowing an old marker or dividing the bag into thirds.
The first Error/Fail at or after that corrected stage-1 start is saved separately
as the failure-time annotation; it is never used to select a match or score.
Saved results include the matching metadata, top three candidates, visual
descriptor and matched thumbnail for each available observation, plus separate
preparation/search timings. Visual pooling time is included in cached feature
time. The benchmark is unchanged.

Use the **Detector** selector before starting an execution:

- **VLM** uses the configured CoP kNN/logistic classifier. Turn **Find similar
  nominal frame** off for the previous classifier-only behavior. Turning it on
  adds the visual matching preview; it does not replace the classifier's memory.
  Cached processing remains the default. Cache comparison controls are under
  **Advanced: VLM cache diagnostics**.
- **DINO vision** loads the existing named memory from
  `<nominal_cache_dir>/<vision_memory_name>/<camera_slug>/`. Its native CLS
  shortlist, attention-weighted patch scoring and saved adaptive thresholds
  remain in use. It shows the scored image's heatmap and the actual selected
  nominal bag, frame number and timestamp. Similar-frame search is intrinsic
  to DINO scoring, so the VLM preview switch does not disable it. If memory is
  missing or incompatible, prepare it with
  `docker compose run --rm --build vision-memory`. Replay never rebuilds DINO
  memory automatically. It uses the first configured RynnBrain camera; that
  camera must have compatible DINO memory. Stage-constrained replay requires
  the final numbered `1, 2, 3` boundaries instead of guessing missing stages.
  Replay processes **every sampled image through the end of the recording**,
  at the existing DINO `SAMPLE_FPS` rate (currently 2 fps), rather than eight
  VLM observations. This is every detector input, not every raw camera message.
  Each image supplies its native score, adaptive threshold and heatmap. If
  inference cannot keep up, replay falls behind wall time rather than dropping
  inputs; Observation → result includes this delay.
- **GMR online TS** has a **TS signal** selector: TCP pose (default), joint
  positions, gripper position, TCP pose + gripper position, or force/torque.
  It fits and calibrates a separate model for the selected signals using the
  existing top-level `nominal_bags`. No new config fields or paths are needed.
  The first run prepares the nominal model; subsequent runs reuse it. Test
  scores are computed as replay advances and appear on the GUI graph immediately.
  The original offline GMR scripts and their synchronization are unchanged.

  **Stage policy:** ignore stage 2/3 until the first stage 1; stage 1 starts
  detection, stage 2 advances progress, and the first stage 3 after that start
  ends detection. Repeated/backward markers do not rewind the run. Error/Fail
  markers are recorded for evaluation only and never advance progress or affect
  scores. Since stage 3 means *end*, there are two active intervals: 1 to 2 and
  2 to 3. Nominal runs need both intervals; a test can end early at stage 3.
  Missing start/end markers produce an explicit error rather than an invented
  window. If stage 2 is absent in a test, progress remains in interval 1 until
  stage 3; no boundary is guessed. This policy applies to GMR replay only.

  The current stage's end time is unknown online. Phase is estimated from elapsed
  time within the stage divided by that stage's median nominal duration, clamped
  to its half of the phase range. Stage 2 resets progress to the second half.
  A slower/faster execution can still affect scores; this is an estimate, not
  exact motion alignment. Leave-one-run-out calibration excludes the held-out
  nominal from both model fitting and duration estimation.

  Signals are sampled on a **10 Hz clock** using only the latest received values,
  including held gripper feedback; there is no interpolation from future data.
  All selected signals must have arrived after stage 1 before scoring starts.
  Pose, joints and force older than 1 second stop the run with a stale-data error;
  gripper feedback is held because it can be sparse while stationary. Camera
  playback proceeds independently (up to 15 fps); TS scoring works without video.
  There is no eight-observation limit. If processing falls behind, replay keeps
  the samples and reports the delay rather than silently skipping them.

  GMR scaling, conditional Mahalanobis distance and nominal adaptive threshold
  calibration retain their existing methods. Score smoothing is now a trailing
  11-sample median (past/current data only), followed by the existing 6-of-8
  majority vote. These filters introduce detection delay. The main graph shows
  **raw joint Mahalanobis distance and its adaptive nominal threshold**, in
  distance units. Red points mark instantaneous threshold crossings; the
  separate filtered-alarm status still uses the trailing median of epsilon
  (distance / threshold) and majority voting. This display change does not
  change alarm decisions or require recalibrating an existing online model.
  CSV/JSON save both forms: `anomaly_score`/`raw_distance`,
  `threshold`/`adaptive_threshold`, `epsilon_raw`, `epsilon_smooth`,
  `alarm_filtered`, current stage and estimated phase. **Scoring time**
  includes scoring, filtering and channel diagnostics; **TS sample to result** includes replay
  delay. Nominal model preparation time is reported separately.

  **Individual signal plots** show every channel used by the selected TS mode,
  measured values, conditional nominal means, and projected joint-limit envelopes
  in original units. Hover over a plot to inspect a timestamp. Orientation plots
  are the six rotation-matrix features used by GMR, not Euler angles. The
  envelopes are the projection of the joint Mahalanobis boundary onto each axis:
  `mean_j +/- distance_threshold * sqrt(covariance_jj)`, transformed back through
  the nominal scaler. The covariance includes the same regularization used by
  scoring. They are diagnostic envelopes, not separately calibrated channel
  thresholds, confidence intervals, or physical limits. A joint alarm can occur
  with all channels inside these envelopes because GMR also models correlations.
  The displayed scaled deviation is the absolute marginal residual divided by
  its modeled standard deviation; it is not an additive contribution or gain.
  Channel amplitudes alone do not determine influence: the nominal scaling and
  covariance matter. No channel weights are changed by these plots. Each saved
  row's `channels` dictionary contains `observed`, `expected`, `lower`, `upper`
  and `standardized_deviation`; no additional config fields are needed.

The signal readers prefer `/tcp_pose_broadcaster/pose`, `/joint_states`, and
`/Robotiq2FGripperRobotInputMsg`. The same typed topics must exist in all nominal
and test bags. TCP position uses metres; orientation uses six rotation-matrix
components so quaternion sign changes do not create artificial anomalies.
Joint positions retain their names and native units; joint effort is not
treated as measured TCP force. Joint angles that wrap in the recordings may
still require separate preprocessing. `/tf` is not used as a force source.
The existing GMR StandardScaler handles different channel scales using nominal
training data.

Gripper mode uses **actual position** (`gPO`, or its snake-case equivalent),
normalized from 0–255 encoder counts to 0–1, not requested position (`gPR`) and
not a claimed opening distance in millimetres. It holds the latest feedback
between messages rather than interpolating an artificial closing motion.
See the [Robotiq Hand-E manual](https://assets.robotiq.com/website-assets/support_archives/document_en/Hand-E_Manual_UniversalRobots_PDF_20240402.pdf)
for the register meanings. The reader includes the exact interface supplied
from the recording workspace: eight `uint8` fields in order `g_act`, `g_gto`,
`g_sta`, `g_obj`, `g_flt`, `g_pr`, `g_po`, `g_cu`. This supports the older SQLite
bags without installing the custom ROS package in Docker. Embedded bag
definitions take precedence over this fallback. Pose and joint modes work
independently of the gripper definition.

The causal GMR model, duration estimates, thresholds and nominal source identity
are cached under `<output_dir>/online_replay/gmr_online/`. Only nominal artifacts
are reused; test scores are always computed during replay. A changed nominal
source, signal selection or training setting produces a different cache key.
Old offline score files are not used by the new GUI GMR mode.

The observation interval and eight-frame limit apply only to VLM. DINO and GMR
ignore the VLM sampling window and continue through their own recording/signal
window. Graphs and CSV/JSON include every scored sample; the visible table shows
the latest 200 rows to keep the interface responsive. CSV is flushed per sample;
GMR JSON snapshots are updated about once per second and on completion/error.

Only one detector is active. Switching modes releases the previous model to
avoid competing for GPU memory; the selected detector stays loaded across
executions in that mode. The initial server startup still loads the configured
VLM. A VLM startup error does not prevent selecting DINO or GMR. Preparation
errors remain visible, and a failed mode can be retried with Start execution.
The output JSON records the detector type, score source
and preparation metadata. DINO adds `frame_*_heatmap.jpg`; non-VLM vector files
are labeled by `vector_kind` (DINO CLS or GMR distance/phase), not CoP features.

Every execution has a unique directory under
`<output_dir>/online_replay/<timestamp_uuid>/<bag>/` containing:

- `online_results.json`: incrementally saved status, source/model identity,
  scores, verification results and elapsed times, including partial/error runs.
- `online_scores.csv`: per-observation score, threshold, cached feature time,
  classifier time, observation-to-result delay and peak allocated GPU bytes.
- `frame_*.jpg` and `frame_*.npy`: selected observations and scored vectors;
  `frame_*_uncached.npy` is added during verification. Diagnostic fallback rows
  use the uncached vector in the primary file and retain `frame_*_cached.npy`.
- `online_timeline.png`: score/threshold plot after completion or stopping.

The graph uses observation timestamps. The interface separately reports when
the result became available and how far its observation lags the camera view.
These files do not overwrite benchmark reports or model artifacts. Frame
preprocessing and inference errors remain visible in the interface.

Raw rosbag benchmarks (`source: "rosbag"`, `input_modes: ["raw"]`) now skip
DINO automatically. The ordinary command is sufficient:

```bash
docker compose run --rm --build benchmark
```

Use `--with-dino` if you also want independent DINO results. Heatmap/video
input modes continue running DINO by default, and `--skip-dino` still allows
reuse of existing videos. `--skip-vlm` runs DINO only. If explicitly requested
DINO processing fails, raw-rosbag VLM evaluation still proceeds; video/heatmap
evaluation remains blocked by that failure. `benchmark_summary.csv` records
`dino_status` separately from VLM results. Bag selection and training exclusions
are unchanged; `--include-nominal-bags` remains optional and explicit.

### Feature-only classifier inference and runtime measurements

To score CoP features without generating the execution answer, manually add this
object inside `rynnbrain` in your existing configuration:

```json
"inference": {
  "feature_only": true
}
```

Keep `cop_vectors.enabled` and `cop_classifier.enabled` true. The configured
`cop_classifier.method` still selects logistic regression or nominal kNN; no
adapter, bag list, classifier method or path is changed automatically. Setting
`feature_only` to false (or omitting it) retains generated VLM decisions.

Feature-only mode performs one execution-prompt forward pass, retaining the same
final normalized, last-prompt-token vector used by the existing classifiers. It
does not enter the token-generation loop or generate an execution explanation.
The LoRA adapter, camera inputs, timestamps and prompts still apply. The nominal
reference description is needed to preserve the trained context: it is generated
once during benchmark preparation and reused for evaluation; a standalone test
generates it once. Image decoding, tokenization, visual encoding and prompt
processing are still required. The speed improvement depends on those costs,
answer length and CPU offloading; there is no promised speedup factor.

Logistic regression continues to report `failure_probability`. kNN continues to
report `anomaly_score` (a distance, not a probability). The VLM decision is saved
as `not_generated`, its answer is empty and its accuracy is unscored. The clean
table includes `classifier_decision` and `classifier_correct` separately. A
temporal adapter does not generate an onset timestamp in this mode; classifier
timeline alerts and their existing timing evaluation remain available.

Keep `rynnbrain.cop_classifier.plot_timeline` true when evaluating failure onset.
Setting it to false disables **prefix scoring as well as the timeline plot**:
there is then no first threshold-crossing time and no classifier timing-error
comparison. This faster setting is appropriate only for full-execution scoring.
Feature-only mode itself does not disable timelines. Optional logistic
onset-classifier preparation is also separate startup
work. Use raw rosbag input and skip DINO for a VLM/classifier benchmark:

```bash
docker compose run --rm --build benchmark \
  --config /config/pipeline_config.json --skip-dino --include-nominal-bags
```

This still excludes reference, LoRA-training and classifier-training bags.
The ordinary `rynnbrain-test-multiturn` and `cop-classifier-train` commands also
honor feature-only mode. Switching this mode alone does not change the feature
comparison signature or require classifier retraining. Changing the adapter,
camera inputs or prompts does. Missing preparation vectors are extracted without
execution-answer generation when feature-only mode is enabled.

If results differ after switching modes, check a single held-out `test_bag`
before rerunning the benchmark. This command uses the **already trained**
classifier and compares normal generation with direct extraction before and
after generation, using identical images, prompts, LoRA weights and reference
text. It does not retrain anything or overwrite existing evaluation outputs:

```bash
docker compose run --rm --build rynnbrain-test-multiturn \
  --config /config/pipeline_config.json --verify-feature-only \
  --baseline-dir /outputs/experiments/pick_place_dino728_v116/rynnbrain_benchmark
```

`--baseline-dir` is optional; set it to your earlier benchmark directory. The
configured `test_bag` should be a held-out bag present in both experiments.
Results go into a new timestamped directory under
`rynnbrain.output_dir/feature_verification/<bag>/`, with `comparison.json`, three
vectors and a persistent log. The report includes vector cosine similarity and
relative error, both paths' classifier scores/decisions, generated text and
timings. A numerical match requires relative vector L2 error at most 0.001,
absolute classifier-score difference at most 0.001, and identical classifier
decisions. These are diagnostic tolerances, not an accuracy guarantee. The
timings include first-call effects and are not a warmed-up throughput benchmark.

When a baseline is supplied, the report also lists differences in saved feature
settings (including adapter hash, cameras and prompts) and added/removed bags
from the two saved `benchmark_summary.csv` files. A baseline with different
settings or classifiers is not a controlled speed-only comparison. Missing
baseline artifacts are reported explicitly. No images need to be uploaded to
inspect these reports. Do not discard the first-turn reference text as a speed
optimization: it changes the trained context, whereas caching the same text
preserves it.

Timing columns are saved in each bag's result CSV/JSON and the benchmark tables:

| Column | Measured wall-clock seconds |
| --- | --- |
| `runtime_input_sec` | Input setup, classifier loading, reference/test image loading and reference-image saving |
| `runtime_reference_sec` | Separate nominal-response generation in feature-only single tests; zero when reused |
| `runtime_model_sec` | Tokenization, transfers and execution feature extraction or response generation |
| `runtime_classifier_sec` | Applying the loaded full-execution classifier to the vector |
| `runtime_diagnostics_sec` | Optional explanations, neighbour plots and timeline processing |
| `runtime_bag_sec` | Whole VLM evaluation call, including image/result preparation and diagnostics |
| `runtime_total_bag_sec` | Benchmark only: DINO (if enabled), VLM evaluation and per-bag output writing |

CUDA work is synchronized at measurement boundaries. `runtime_model_sec` in
normal generation mode also includes nominal generation if it was not reused.
Shared input/reference costs and `runtime_bag_sec` are repeated across a bag's
input-mode rows; do not sum those duplicates. These offline measurements include
bag decoding and diagnostics as specified, not camera-to-alert live latency.
The current commands remain offline bag evaluators, not a ROS streaming node.

`benchmark_runtime.json` records model loading, classifier preparation (including
model loading), mean per-row VLM timing fields, total elapsed time and progress.
`benchmark_run.log` retains printed output and Python tracebacks across runs.
The summary and clean tables are checkpointed after each completed bag; final
plots are generated after the bag loop. A failed/interrupted run's tables are
partial. Check the runtime JSON's status before treating reports as complete.
A hard process kill cannot write a final status, so its last phase remains marked
running; the log and completed rows are still retained. These files aid diagnosis
but do not automatically resume a stopped run. Older plots in a reused output
directory can still belong to its previous run.

Single tests similarly write `inference_runtime.json` and `inference_run.log`
beneath `rynnbrain.output_dir`, including model-loading time separately.

### Nominal-only kNN detector

The checked-in config selects `rynnbrain.cop_classifier.method: "knn"`.
This detector uses only the existing `cop_classifier.training.normal_bags`
list. It ignores `training.failure_bags` and `training.bags`; those lists remain
available if you switch back to logistic regression. No failure examples are
needed to fit the kNN detector or its decision threshold.

Set these fields **inside your existing `cop_classifier` object**, keeping
your current paths and bag lists:

```json
"enabled": true,
"method": "knn",
"plot_timeline": true,
"knn": {
  "n_neighbors": 3,
  "threshold_quantile": 0.95
}
```

The detector stores the full L2-normalized nominal CoP vectors. For a new bag,
its anomaly score is the mean Euclidean distance to the `n_neighbors` nearest
nominal vectors. Larger distance means less similar to the nominal examples.
PCA is only for visualization; its two coordinates are not classifier inputs.
This is nearest-neighbour novelty detection, not k-means clustering.

The cutoff comes from nominal training distances calculated with each bag
left out of its own neighbours. A new score above their configured quantile
is classified as failure. With the default `n_neighbors: 3`, at least four
nominal bags are required. Use diverse nominal executions and assess false
alarms on separate, unseen nominal bags: a 95th-percentile training cutoff
does not guarantee a 5% false-alarm rate on new data. The saved threshold
implements this strict cutoff with the same `score >= threshold` rule used
by the reports. No test labels select the threshold.

Run the same benchmark command:

```bash
docker compose run --rm --build benchmark
```

It prepares only the selected nominal training vectors, fits kNN, then
generates each eligible test bag's full response once. Compatible saved training vectors and
responses are reused. Missing or stale ones are extracted with the configured
VLM and LoRA adapter. `--include-nominal-bags --skip-dino` still supports a
VLM-only benchmark including DINO-memory bags; reference, LoRA-training, and
kNN-training bags remain excluded. An old logistic failure-training list
alone does not exclude bags while kNN is active.

The existing `model_paths.raw` may stay `.../raw_logistic.npz`. kNN derives
`.../raw_knn.npz` in that same directory, preserving the logistic model.
Other configured filenames similarly receive `_knn`; a name already ending
in `_knn` is used as-is. Training and inference resolve this path identically.
The sibling JSON records the training bag names, feature signature, and cutoff
settings; `raw_knn_training_predictions.csv` contains leave-one-out training
distances, not held-out accuracy estimates. Single-bag inference uses the
saved detector via the usual `rynnbrain-test-multiturn` service. Standalone
`cop-classifier-train` also supports kNN and only prepares/fits the selected
nominal bags; it does not score the entire benchmark.

For kNN, `benchmark_clean.csv` contains `anomaly_score`, `anomaly_threshold`,
and `classifier_decision` alongside the original VLM decision and correctness.
`failure_probability` stays empty: distance is not a calibrated probability.
The detailed outputs use `classifier_anomaly_score` and
`classifier_score_kind: "knn_distance"`. Reports include:

- `benchmark_anomaly_score_raw.png`: distance boxplots with category markers.
- `benchmark_anomaly_score_summary.csv`: distance statistics by actual outcome.
- `benchmark_anomaly_roc/`: ROC/AUROC per failure category and overall, using
  continuous distances, with the saved cutoff's recall and false-alarm rate.

PCA, VLM decision ROC, response JSON/CSV files, selected frames, and classifier
decision statistics continue to be produced. Failure labels are still used
for **evaluation**, although kNN training needs only nominal data.

To return to the supervised classifier, set `method: "logistic"`; it uses
the original model path, nominal/failure lists, probability columns, and
probability plots. Old configs without `method` retain logistic behavior.
`training.threshold`, `regularization_c`, and `class_weight` apply only to
logistic regression; kNN uses its own settings above.

The VLM/LoRA weights are not changed by classifier training. If the loaded
LoRA adapter was trained with failures, the **classifier** is nominal-only,
but the complete pipeline has still seen failure supervision. For an
experiment without that task-specific supervision, evaluate the base model
with `lora_adapter_path: null`; changed feature signatures trigger preparation
of matching training vectors.

#### Anomaly score and threshold over time

Every evaluated kNN test bag also gets `knn_neighbors_raw.png` and
`knn_neighbors_raw.csv` in the same output directory. The figure shows the
test execution beside the nominal training examples in a PCA projection fitted
only to those nominal vectors, plus a ranked chart of actual full-feature
distances. Blue bars mark the k neighbours used for the score; their mean is
the anomaly score. The threshold applies to that mean, not each individual
distance. PCA is only a visual aid and can hide separation. The CSV lists every
nominal bag, its distance, rank, and whether it contributed to the score, together
with the evaluation ID and LoRA identity. This adds no VLM inference passes and
works independently of `plot_timeline`. `knn_neighbors_status` and
`knn_neighbors_error` in the response files report any plotting problem.

With kNN, `cop_classifier.plot_timeline` defaults to `true`. The normal
benchmark automatically prepares **progress-matched nominal-prefix detectors**
and saves two files per evaluated bag and input mode in its existing
`rynnbrain_multiturn/` directory:

- `knn_timeline_raw.png`: anomaly distance and the matched nominal threshold
  against actual seconds from the start of the bag/video. Red dots exceed the
  threshold; thresholds can change as the execution progresses.
- `knn_timeline_raw.csv`: timestamps, scores, thresholds, sampled progress,
  image indices, errors, evaluation ID, and adapter identity for each snapshot.

For `num_frames: 8`, snapshot 1 uses the first selected execution timestep,
snapshot 2 uses the first two, and so on. Every snapshot uses the same nominal
reference and configured base model/LoRA. It receives no execution images from
later timesteps. For multiple cameras, a timestep includes that sample from
each camera; its plotted time is the latest included timestamp.

Training uses the existing `training.normal_bags` only. It creates a separate
nominal vector memory for each prefix length and estimates that prefix's
threshold from leave-one-bag-out nominal distances, using the same k and
quantile as the complete-execution detector. No failure or test labels choose
the thresholds. A healthy unfinished task is thus compared with nominal
executions observed up to the same sampled progress, rather than only with
finished executions.

The last point reuses the existing full-execution vector, and its score and
threshold match the bag's summary exactly. VLM answers, overall benchmark
statistics, ROC/AUROC, and PCA still use the full-execution results. Prefix
scores are kept separate from those results.

The prefix detector is saved beside the complete detector as
`raw_knn_timeline.npz` plus a JSON sidecar. Nominal prefix vectors are cached
under each training bag's `rynnbrain_multiturn/cop_timeline/` directory, outside
the full-vector/PCA input files. They are reused only when their full-vector
evaluation ID, model/adapter/prompt signature, and sampling protocol match.
Missing or stale prefix caches are refreshed without regenerating compatible
full responses. `--refresh-vectors` also refreshes prefix vectors.

This adds inference work: eight timesteps require seven extra short vector
extractions per training bag on the first run, and per evaluated bag. They
reuse the reference response and do not generate full text answers. Set
`plot_timeline: false` to keep only the original whole-execution evaluation.
No new benchmark command or separate LoRA training is required. For a single
bag, first run the usual benchmark or `cop-classifier-train` once to prepare
the prefix detector, then use `rynnbrain-test-multiturn` normally.

Keep `rynnbrain.num_frames` at its existing value (e.g. 8) and omit
`cop_classifier.timeline_num_frames` to use the original sampled snapshots.
The optional denser kNN sampling setting remains supported, but it changes
sampling resolution, not the model's ability to recognize early failure evidence.
If returning from a denser kNN detector, rerun `cop-classifier-train` once after
removing that setting to rebuild matching nominal-prefix memories; compatible
full-execution vectors are reused. This is offline analysis, not streaming.

The graph shades the interval from the last below-threshold snapshot to the
first above-threshold snapshot. This is a bracket for the **first observed score
crossing**, not a physical failure-start annotation or an interpolated estimate.
If the first valid sample is already anomalous, no lower bound is invented;
missing snapshots widen the bracket and are counted in the response JSON's
`knn_timeline.first_alert`. The result CSV also includes
`knn_last_below_before_alert_sec` and `knn_first_above_threshold_sec`.
Speed/stage misalignment, the retained image history, and the detector itself
can delay an alert even when the failure is visible in an earlier sampled frame.

These are **offline sampled-progress timelines**: the nominal and test bags
are aligned by sample index across their respective recordings, not by exact
robot stage or a shared duration. Different execution speeds can affect this
alignment. The lines connect evaluated snapshots; they do not establish an
exact failure-onset time between samples. A recording with too few sampled
timesteps cannot produce a matching timeline. Snapshot failures appear as
gaps with errors in the CSV. A timeline failure is recorded in
`knn_timeline_error` and does not discard the VLM response or final score.

#### Logistic failure score over time

With `cop_classifier.method: "logistic"`, `enabled: true`, and
`plot_timeline: true` (the default), single tests and benchmarks also save
`logistic_timeline_raw.png` and `logistic_timeline_raw.csv` in each evaluated
bag's existing output directory. The graph shows the failure probability from
0 to 100%, the saved classifier's fixed decision threshold (usually 50%), and
above-threshold snapshots. Logistic failure probability is the anomaly score
here; it is not a kNN distance or the probability that the VLM's answer is correct.
The CSV stores `anomaly_score` and `failure_probability` as the same 0–1 value,
plus `failure_percent`, `threshold`, `threshold_percent`, timestamps, errors,
evaluation ID, and adapter identity. Aggregate benchmark columns keep their
existing meanings: logistic scores remain in `failure_probability`.

By default this reuses the **existing complete-execution logistic classifier** and the
original sampled frames. For eight timesteps it makes seven extra short prefix
vector extractions, with the same LoRA/reference/prompt, without generating
additional full answers. The final point reuses the original full-execution
vector, exactly matching the probability and threshold in the normal results.
No prefix classifiers, new training labels, or failure-start annotations are
required. `timeline_num_frames` is not used for logistic timelines.

Prefix scores are a **diagnostic**: the classifier was trained on completed
executions, so these scores are not calibrated probabilities that a failure
has already happened. A failed bag can have normal early frames; assigning its
failure label to every prefix would not teach accurate failure-start timing.
The optional timestamp-supervised model below addresses this training mismatch.

Use the normal `rynnbrain-test-multiturn` or `benchmark` command. Run
`cop-classifier-train` first only if a compatible logistic classifier is not
already available. `plot_timeline: false` skips the extra inference. The result
CSV and response JSON report `logistic_timeline_status`, `logistic_timeline_plot`,
`logistic_timeline_csv`, and `logistic_timeline_error`; the JSON's
`logistic_timeline.first_alert` records the first observed threshold-crossing
interval. Snapshot errors leave gaps; a timeline failure retains the VLM answer
and final classifier result. Switching methods clears stale timeline graphs
from the previously used method in that same output folder.

#### Train logistic timelines with recorded failure times

To teach the timeline classifier whether a failure has **already occurred**, add
this field yourself inside `rynnbrain.cop_classifier` (keep your current paths
and bag lists):

```json
"logistic_onset": {
  "enabled": true
}
```

Use `method: "logistic"`, `enabled: true`, `plot_timeline: true`,
`rynnbrain.source: "rosbag"`, and input mode `raw`. The feature is off when the
new field is absent or false, and does not change kNN behavior.

Then train and run a held-out test with the existing commands:

```bash
docker compose run --rm --build cop-classifier-train
docker compose run --rm --build rynnbrain-test-multiturn
```

Or run `docker compose run --rm --build benchmark`: its normal classifier
preparation trains both logistic models before evaluating held-out bags.
No LoRA retraining is needed. This code does not edit your configuration.

Training uses only `cop_classifier.training.normal_bags` and `failure_bags`
(or explicitly selected training bags). It reuses compatible saved vectors and
extracts missing prefix vectors from the same frozen VLM/LoRA. With eight frames,
each training bag supplies eight visual prefixes: first frame, first two frames,
and so on. This does **not** increase the frame count. The first run needs extra
prefix inference; later runs reuse matching timestamped caches. Older caches
without complete timestamps are regenerated. `--saved-vectors-only` requires
these prefix caches and access to the bags' stage annotations.

Labels come from the first eligible failure marker on `stage_topic`, using the
existing startup filter (0.1 seconds by default):

- Nominal bag: all prefixes have target 0.
- Failure bag: prefixes strictly before the marker have target 0; prefixes at
  or after it have target 1, even if the robot subsequently recovers.
- Missing/unreadable failure annotations, nominal bags with failure markers,
  and failures recorded after the final sampled image are excluded from onset
  fitting, with reasons reported. At least two usable nominal and two annotated
  failure bags are required. Whole-execution training keeps its existing labels.

One separate logistic classifier is fitted across the labeled prefixes, using
the existing regularization, class-weight setting, and threshold. Time and stage
messages are **not prediction inputs**. All prefixes from a bag stay together
in cross-validation; preprocessing is fitted within each training fold. Its
out-of-fold diagnostics evaluate this classifier only: the upstream frozen
VLM/LoRA is not retrained per fold. Use entirely held-out bags for final evaluation.

The new model is saved beside the whole-execution model as
`raw_logistic_onset.npz` and `.json` when the original is `raw_logistic.npz`.
The metadata includes annotation provenance, excluded bags, grouped validation,
and the complete-model fingerprint. `raw_logistic_onset_training_predictions.csv`
lists every training prefix, timestamp, target, fold, and out-of-fold score.

With this option enabled, **all timeline points**, including the last one, use
the onset classifier. `logistic_timeline_raw.csv` identifies it through
`threshold_source: timestamp_supervised_prefix_classifier` and its model path.
The timeline's final probability may differ from `benchmark_clean.csv`'s
`failure_probability`, which retains the separate **whole-execution outcome**
meaning. VLM responses, whole-execution ROC/boxplots, and LoRA weights are unchanged.
Failure-timing errors and their aggregate graph use the new timeline's first
threshold crossing. Missing/stale onset models produce a clear timeline error;
they never silently fall back to the old diagnostic classifier.

This supervision can help, but improvement is not guaranteed. It assumes the
execution was normal before the button press; human reaction delay can make
that label imperfect. Eight snapshots cannot pinpoint events between snapshots,
and the frozen visual features must contain evidence of the failure. Scores are
not guaranteed calibrated probabilities, especially with balanced class weights.
Compare timing error **and missed detections/false alerts** on the same held-out
bags before concluding it is better. Set `logistic_onset.enabled` back to false
to use the original whole-execution classifier for diagnostic timelines.

### Failure-time evaluation against the error button

For raw rosbag inputs, classifier timelines are now compared automatically with
the recorded error-button event on `stage_topic` (normally `/recording_stage`).
This works for both kNN and logistic regression, with `plot_timeline: true`,
and adds no model inference or training. No config paths or bag lists need to
change. Keep the existing eight-frame sampling if that is your current setup.

The reference time is the **first failure marker after the startup filter**.
The existing failure-marker vocabulary is reused (`Error`, `Anomaly`, `fail`,
`failure`, `failed`, `NOK`, `not OK`, `Fehler`, case-insensitive). All stage
messages are examined for onset, not just the final three messages used by the
existing bag-label heuristic. Benchmark `--startup-ignore-sec` is respected;
otherwise `$STAGE_STARTUP_IGNORE_SEC` or its default 0.1 seconds is used.
The complete marker identity, timestamp, count, and filter are recorded in
`failure_annotation` in each response JSON. Existing bag labels are unchanged;
a nominal label that conflicts with an error marker is explicitly flagged.

The predicted time is the **first sampled classifier score at or above its
threshold**, including early alerts. Recorded and predicted times both use
seconds since ROS bag start. The timing error is:

```text
failure_time_error_sec = predicted_failure_time_sec - recorded_failure_time_sec
```

Positive means late; negative means an early alert. For example, a button press
at 20 s and first classifier alert at 26 s gives +6 s error and 6 s absolute
error. No interpolation between snapshots is used. This evaluates the
classifier's alert against the human annotation; it is not the VLM's textual
decision timing, inference wall-clock latency, or an exact physical-onset claim.
The evaluated bag's annotation is used only after prediction, never to choose
thresholds, train models, or add information to its prompt. The optional onset
classifier above uses annotations from its separate training bags.

Each case's existing `knn_timeline_raw.png` or `logistic_timeline_raw.png` gets
a green vertical line for the recorded error-button time and a timing-error
caption. The per-case result CSV/JSON and both `benchmark_summary.csv` and
`benchmark_clean.csv` gain:

- `recorded_failure_time_sec`
- `predicted_failure_time_sec`
- `failure_time_error_sec` (signed)
- `failure_time_absolute_error_sec`
- `failure_timing_status`, `failure_timing_error`, `failure_timing_method`

The benchmark also writes `benchmark_failure_timing/failure_timing_raw_knn.png`
(or `failure_timing_raw_logistic.png`), its per-bag CSV, and
`failure_timing_summary.json`. The figure is one horizontal bar plot of signed
timing differences for **failure-labeled bags with valid comparisons**. Nominal
cases are hidden. Blue bars indicate early alerts and orange bars late alerts;
a prominent purple dashed line spans every row at the mean signed difference.
The header also reports the **mean absolute error** so early and late errors
cannot cancel. Long bag labels wrap, and the layout scales to the displayed cases.
The same statistics appear under `failure_timing` in `benchmark_statistics.json`.
Each input mode is summarized separately, without pooling multiple predictions
for the same bag into a single mean.

Means include only failure-labeled bags with complete, aligned timelines, a failure
annotation inside the sampled time window, and an observed alert. All evaluated bags
stay in the CSV table; missed detections (`not_detected`), missing markers,
nominal false alerts, partial timelines, annotation/label conflicts, and failed
evaluations have explicit statuses and counts. They are **not assigned zero
error**. When no case can be compared, means are unavailable (`null`), not zero.
The graph counts failure cases without valid timing differences in its footer;
they are excluded from the bars and mean, rather than appearing as zero-error bars.

Video playback timestamps are not assumed to equal bag time: generated-video,
heatmap, and mixed raw/heatmap inputs currently report `unaligned_timebase`
rather than subtracting unrelated clocks. Your `source: "rosbag"` with
`input_modes: ["raw"]` uses the supported common clock. Missing stage topics
or optional plot errors preserve the existing VLM/classifier results.

Use the normal `rynnbrain-test-multiturn` or `benchmark` commands. A single test
gets its annotation overlay and timing fields; the benchmark additionally
aggregates the bags executed in that run. Previously saved results are not
silently mixed into these averages.

### Frozen-vector logistic failure classifier

RynnBrain remains fully frozen. After collecting labeled vectors, a small
L2-regularized logistic-regression head can be trained directly on the complete
hidden vectors (PCA coordinates are not used). Each vector is L2-normalized and
centered using training-set statistics. The saved classifier contains only one
weight per hidden feature, an intercept, preprocessing values, and metadata.

Choose a representative training set containing multiple nominal and
failure bags. Store the fixed training split
in `pipeline_config.json`; this avoids rebuilding the classifier during normal
testing:

```json
"cop_classifier": {
  "enabled": false,
  "method": "logistic",
  "model_paths": {
    "raw": "/outputs/experiments/example/cop_classifier/raw_logistic.npz"
  },
  "training": {
    "input_dir": "/outputs/experiments/example/rynnbrain_benchmark",
    "input_mode": "raw",
    "normal_bags": [
      "/data/Nominal/setting 1/nominal_1.2",
      "/data/Nominal/setting 1/nominal_1.3"
    ],
    "failure_bags": [
      "/data/Failure_1_grasp_miss/failure_1.1",
      "/data/Failure_1_grasp_miss/failure_1.2"
    ],
    "class_weight": "balanced",
    "regularization_c": 1.0,
    "threshold": 0.5
  }
}
```

Run the dedicated service once:

```bash
docker compose run --build --rm cop-classifier-train
```

With `--config` (including the Docker service default), this command now prepares
its selected training vectors before fitting the classifier. It loads the VLM
once and generates the nominal reference response to check the full feature
signature. Compatible saved vectors are reused; missing, invalid, or stale
selected vectors are extracted into the existing `training.input_dir`, then the
classifier is fitted. It does **not** score every saved benchmark bag or update
benchmark summaries. Extraction uses the configured model and LoRA adapter, if
enabled. It may extract classifier training features from LoRA training bags;
the reference demonstration cannot be a supervised training sample.

For your `source: "rosbag"`, `input_mode: "raw"` setup, no preliminary benchmark
or DINO run is required. Heatmap/generated-video modes still need the existing
per-bag videos under `training.input_dir/<bag_name>/`. Global video overrides
cannot identify separate training bags and are rejected during preparation.

With `rynnbrain.cop_classifier.enabled: true` and your existing explicit
classifier training lists, the **normal benchmark command** now prepares the
classifier automatically when `train_on_benchmark` is true (the default):

```bash
docker compose run --rm --build benchmark
```

The benchmark prepares only the configured training bags, fits the classifier, then
evaluates each test bag once with probabilities already available. One VLM
instance serves both phases. Compatible training vectors are reused; missing or
stale vectors are refreshed. It enables vector capture for classifier scoring
without editing the config. Training output paths must match `model_paths`.
With `train_on_benchmark: true`, the selected classifier is fitted on each run;
the VLM/LoRA is not retrained. With `train_on_benchmark: false`, it loads the
existing `model_paths` artifacts and skips all classifier preparation/training.
Setting `cop_classifier.enabled: false` disables automatic classifier preparation
and scoring. The older `--prepare-classifier` flag remains an optional way to
enable them for one run. `--skip-vlm` does not prepare a classifier. Raw-rosbag-only
VLM benchmarks skip DINO automatically; use `--with-dino` to include its results.
Classifier-training bags (including those recorded in the saved classifier)
are excluded from benchmark test results, even with `--include-nominal-bags`.
The clean table includes `failure_probability`; statistics include a separate
`classifier` section alongside the VLM's decision statistics. Single-bag
RynnBrain evaluation continues to load the saved classifier without training.

Classifier preparation also saves the complete generated responses in each
training bag's existing `rynnbrain_multiturn/` folder:
`rynnbrain_results_multiturn.csv`, `rynnbrain_responses_multiturn.json`, and
`selected_vlm_frames.csv`, alongside the selected images and CoP vectors.
These bags are marked `sample_role: classifier_training` and still excluded
from held-out benchmark statistics. Older preparation runs saved only images
and vectors; their unsaved answers cannot be recovered from those vectors.
The next normal preparation/benchmark run regenerates selected training bags
whose response files are missing or cannot be matched to their vector. This
includes legacy files without an `evaluation_id`; complete matching outputs
are then reused. Multiple input modes retain their separate saved responses.

In the response JSON and CSV, `nominal_response` describes the nominal
reference (turn 1); `response` is the complete execution answer (turn 2).
`generation.explain_decision: false` disables only the optional separate
evidence pass. It never removes explanation text generated in the execution
answer itself. Each new result records its generation time, evaluation ID,
base model ID, loaded adapter path/SHA256, and response source. The vector
metadata shares the same evaluation ID so the answer can be traced to its
vector. Adapter information is also printed at startup and included in
`benchmark_summary.csv`. A classifier compatibility error leaves probability
empty and saves `classifier_error`, retaining the VLM answer and vector;
other benchmark failures are recorded in `evaluation_error` in the summary.

Each benchmark also creates `benchmark_failure_probability_<mode>.png`, a
boxplot of classifier failure probabilities (0–100%) for ground-truth successful
and failed executions. Groups use the recorded/manual ground truth, not the
VLM or classifier decision. Boxes show the middle 50%, the line is the median,
whiskers extend to observations within 1.5 IQR, and dots show individual bags.
Boxes and individual markers are drawn side by side to keep dense clusters
readable. Successful executions use circles; each failure category has its own
marker shape and colour, with a separate legend using the same category styles
as PCA. Categories come from `failure_category` or the existing bag folder path;
missing categories are shown as uncategorized failures. Only horizontal marker
positions are spread for readability; probabilities remain unchanged. Group
labels show the number of bags and median probability.
Input modes are plotted separately. `benchmark_failure_probability_summary.csv`
and the `failure_probability` section in `benchmark_statistics.json` contain
counts, mean, median, quartiles, minimum, and maximum in percent. Missing or
invalid probabilities and unknown labels are excluded and counted. An absent
class is shown as having no scored bags; a mode with no valid scores has no
plot (and its previous plot is removed). These remain classifier estimates,
not a guarantee of calibrated real-world failure frequency.

To re-extract selected training vectors when bag/video contents changed without
their paths changing, run:

```bash
docker compose run --rm cop-classifier-train --config /config/pipeline_config.json --refresh-vectors
```

To retain the old saved-vectors-only behavior with no VLM loading, use
`--config /config/pipeline_config.json --saved-vectors-only`. This also supports
intentional `training.allow_all_labeled` runs. Automatic preparation requires
explicit bags and deterministic generation (`do_sample: false`).

The trainer uses only
the names in `training.normal_bags` and `training.failure_bags`; these lists
explicitly override recorded labels. Empty lists are rejected to prevent an
accidental train-on-everything run. Command-line options remain available
as explicit overrides. The trainer rejects mixed model/prompt/
reference/camera/frame settings, duplicate bags, unknown labels, and datasets
with fewer than two bags per class. Several dozen diverse bags per class are
strongly preferable to the minimum.

For nested datasets, full container paths such as
`/data/Nominal/setting 1/nominal_1.2` are recommended. Data-root-relative paths
such as `Nominal/setting 1/nominal_1.2` and unique final names such as
`nominal_1.2` are also accepted. JSON paths need no escaping for underscores;
spaces are preserved inside the quoted string.

`class_weight: "balanced"` gives the normal and failure classes equal total
influence even when their bag counts differ. Use `"none"` to optimize ordinary
unweighted logistic loss. Balanced weighting is usually preferable for failure
detection with an uneven training set, but its percentage reflects an equal-
class training prior and is not automatically calibrated to the real failure
rate.

Training creates:

- `raw_logistic.npz`: weights, intercept, training mean, and threshold.
- `raw_logistic.json`: exact representation signature, class counts, fit
  diagnostics, and deterministic stratified cross-validation metrics.
- `raw_logistic_training_predictions.csv`: out-of-fold probability for every
  training bag. These predictions are more informative than fitted-set scores.

After that one training run, enable the saved head for future tests:

```json
"cop_classifier": {
  "enabled": true,
  "method": "logistic",
  "model_paths": {
    "raw": "/outputs/experiments/example/cop_classifier/raw_logistic.npz"
  }
}
```

Every later RynnBrain result and benchmark row then includes
`classifier_failure_probability`, `classifier_failure_percent`,
`classifier_decision`, and `classifier_decision_correct`. The original
RynnBrain text decision remains alongside it. Evaluate performance on bags that
were not used to train the classifier; scoring the training bags only measures
memorization. The percentage is a logistic estimate, not a guaranteed calibrated
real-world probability. Calibration becomes credible only with a sufficiently
large, representative, independently evaluated dataset.

The default RynnBrain base is NVIDIA's PyTorch 25.08 container for Jetson AGX
Thor. It can be overridden with `RYNNBRAIN_BASE_IMAGE` when running on a
different NVIDIA platform.

## Benchmark across every demonstration

`run_benchmark.py` scores the configured RynnBrain (multi-turn) setup against
every bag under the data root that was **not** used as a DINO nominal bag
(`nominal_bags`) or a RynnBrain nominal-demonstration bag
(`rynnbrain.reference_bags`) — both are read straight from `pipeline_config.json`
and automatically excluded. For raw rosbag-only VLM inputs, no DINO memory is
needed. For heatmap/video inputs or an explicit `--with-dino` run, build the
DINO nominal cache for the configured cameras once first:

```bash
docker compose run --rm vision-memory
```

Then run the benchmark itself:

```bash
docker compose run --rm benchmark
```

For each remaining bag it writes a self-contained folder (videos, DINO CSVs,
selected VLM frames, prompts/responses) under
`<rynnbrain.output_dir>_benchmark/<bag_name>/`, and appends one row per
`(bag, input_mode)` to a single `benchmark_summary.csv` in that same directory.
It also writes `benchmark_clean.csv`, containing only `bag_name`, `label`,
`model_decision`, `correct`, and the logistic classifier's
`failure_probability`, plus
`benchmark_statistics.json` with overall counts, accuracy, and breakdowns by
label, decision, and input mode.
Ground truth per bag comes from `--normal-bag`/`--failure-bag` when supplied;
otherwise it uses the recorded-stage-marker heuristic from
`build_dataset_manifest.py`. Bags without a clear `normal`/`fail` marker still
run but are excluded from the printed accuracy numbers. The RynnBrain model is
loaded once for the whole run rather than once per bag. Pass `--limit N` to
smoke-test on a handful of bags, or `--skip-dino`/`--skip-vlm` to rerun only
one stage.

### CoP probability ROC and AUROC by failure category

With `rynnbrain.cop_classifier.enabled: true` and `method: "logistic"`, the usual benchmark command also
produces probability-based ROC/AUROC reports automatically, using the scores
already computed for test bags. No extra model calls or second benchmark run
are needed. These reports live in `benchmark_probability_roc/` inside the
existing benchmark output directory:

- `roc_<input_mode>.png`: ROC curves for all failures and each failure category
  versus nominal bags, with AUROC values in the legend.
- `auroc_<input_mode>.png`: a horizontal AUROC comparison chart.
- `auroc.csv`: AUROC, scored class counts, score coverage, missing/invalid-score
  counts, and reasons for skipped comparisons.
- `roc_points.csv`: every distinct probability threshold and its false/true
  positive rates. Thresholds use 0–1 probabilities; `inf` is the initial
  endpoint where no bags are predicted as failures.
- `roc_summary.json`: report metadata and metrics, also saved under
  `probability_roc` in `benchmark_statistics.json`.

This report uses `classifier_failure_probability` directly, without converting
it to a success/failure decision at 0.5 or the configured classifier threshold.
Failure is the positive class. Tied scores enter together; a constant score
has AUROC 0.5 when both classes are present. A curve may still have few points
if only a few distinct probabilities are available.

All distinct score thresholds are already tested; no intermediate ROC points
are dropped. Testing additional thresholds between adjacent scores cannot
change a prediction and therefore cannot add ROC detail. False-positive rates
change in multiples of `1 / normal_count`, and recalls in multiples of
`1 / failure_count`; several such changes can lie on the same straight segment.
Several perfect-ranking categories can also overlap on the same curve.

Both probability and kNN ROC plots emphasize **Overall (all evaluated bags)**
as a thick dark line drawn above the thinner, lighter category curves. This
overall ROC pools all scored held-out nominal and failure bags directly; it is
not an average of category AUROCs. The empirical points are not smoothed.

The plots show each comparison's sample counts. ROC dots mark observed score
thresholds; diamonds mark the saved `classifier_threshold`. The compact legend
keeps every AUROC and shows threshold recall/false-alarm details for the overall
curve; per-category threshold details remain in the CSV/JSON reports.
The common threshold must be present and valid
for every scored bag in that comparison; missing, invalid or mixed thresholds
are reported instead of assuming 0.5. `auroc.csv` and `roc_summary.json` also
save the configured threshold and TP/FP/TN/FN counts and rates at that threshold.
These additions do not change AUROC or tune the classifier on test bags.
AUROC 1 means perfect ranking on the evaluated bags, not necessarily perfect
classification at the configured threshold. More independent test executions,
particularly in small categories, provide stronger evidence than a denser
threshold grid. Select any new threshold on separate validation data.

Each category is compared against the same held-out nominal bags; other failure
categories are excluded from that category's curve. Input modes are kept
separate. Both ground-truth classes and finite scores in [0, 1] are required.
Missing/invalid probabilities and unknown ground truth are excluded and counted;
an uncertain/unparsed VLM text decision does not exclude a valid classifier
score. AUROC measures ranking, not probability calibration or VLM text accuracy.
Classifier-training and LoRA-training bags remain excluded by the benchmark.

The existing report-regeneration command below regenerates both the binary
and probability ROC reports, plus the probability boxplots, from
`benchmark_summary.csv` without loading the VLM.
If that CSV has no classifier scores, probability curves are skipped with an
explicit reason rather than substituting the VLM's binary outputs.

### VLM decision evaluation and binary ROC by failure category

Each benchmark also writes `benchmark_roc/` inside its existing benchmark
output directory:

- `roc_<input_mode>.png`: overall and per-category binary ROC plots, with the
  VLM's operating point marked and binary AUROC in the legend. Each input mode
  is evaluated separately.
- `auroc.csv`: binary AUROC, confusion counts (TP/FP/TN/FN), accuracy, failure
  recall, specificity, false-alarm rate, decision coverage, abstention counts,
  and reasons for skipped curves.
- `roc_points.csv`: thresholds on the binary decision encoding and false/true
  positive rates. These thresholds are not model confidence thresholds.
- `roc_summary.json`: the same metrics and plot paths (also included under
  `roc` in `benchmark_statistics.json`).

Evaluation uses the VLM's generated `Decision: success` or `Decision: failure`
answer, as stored in the benchmark's `decision` column. The CoP classifier is
not needed and its probabilities do not enter this report; it can remain
disabled. Run the usual benchmark command. Prompts, configuration, paths, and
bag names are unchanged.

For ROC only, `success` is encoded as 0 and `failure` as 1. These are binary
decisions, not probabilities. They give one operating point (false-alarm rate,
failure detection rate); dotted lines connect it to the ROC endpoints. The
resulting **binary AUROC equals balanced accuracy on decided bags**, the mean
of failure recall and normal specificity. It does not measure confidence
ranking or provide a choice of model confidence thresholds.

Categories come directly from existing `Failure_<number>_<description>` parent
folders, such as `Failure_1_grasp_miss` and `Failure_2_slip_at_start`, and are
recorded in `benchmark_summary.csv` as `failure_category`. Each category is
compared against the evaluated normal bags; other failure categories are
excluded from that curve. The overall curve uses all labeled failures versus
normal bags. Existing manual/stage ground-truth labels remain authoritative;
failure bags without a matching category folder appear as
`uncategorized_failure`.

Include held-out normal and failure bags, keeping the existing reference-bag
exclusions. Each ROC needs at least one normal and one failure bag with a parsed
binary decision. `uncertain`, `not_parsed`, and missing/invalid decisions are
excluded from ROC and confusion counts, and counted separately by ground-truth
class. `normal_count` and `failure_count` count only decided bags; columns ending
in `_decided` also refer only to this subset. Always read AUROC alongside
`decision_coverage`: excluding many uncertain answers can make it look better.
`accuracy` counts abstentions as incorrect over all labeled rows in the
comparison. Unknown labels are excluded from decision metrics and counted.
Runs without an input mode (e.g. failed/skipped VLM runs) are counted separately
as `rows_without_input_mode` in the JSON report. If either decided class is
missing, AUROC is empty with a skip reason; available decision metrics are
still saved.

To regenerate just the ROC reports from an existing summary, without loading
the VLM or processing bags again (replace the example summary path with your
existing benchmark output):

```bash
docker compose run --rm --entrypoint python benchmark \
  -m rynnbrain_vlm.benchmark_roc \
  --summary /outputs/experiments/example/rynnbrain_benchmark/benchmark_summary.csv
```

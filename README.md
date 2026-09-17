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
# Uses nominal_bags only; test_bag is not required and no test is run.
docker compose run --rm vision-memory

# Uses test_bag plus an existing named memory; nominal_bags are not read.
# This command never builds or rebuilds memory.
docker compose run --rm vision-test
```

After changing Python source code pulled from Git, rebuild the shared pipeline
image once with `docker compose build vision-memory`. Both services then use the
same updated image. Keep only one top-level `output_dir` entry in the JSON.

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

### Frozen-vector logistic failure classifier

RynnBrain remains fully frozen. After collecting labeled vectors, a small
L2-regularized logistic-regression head can be trained directly on the complete
hidden vectors (PCA coordinates are not used). Each vector is L2-normalized and
centered using training-set statistics. The saved classifier contains only one
weight per hidden feature, an intercept, preprocessing values, and metadata.

First generate a representative training set containing multiple nominal and
failure bags with identical RynnBrain settings. Store the fixed training split
in `pipeline_config.json`; this avoids rebuilding the classifier during normal
testing:

```json
"cop_classifier": {
  "enabled": false,
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

This is the only command that trains or overwrites the classifier. Benchmark
and single-bag RynnBrain runs only load the saved model. The trainer uses only
the names in `training.normal_bags` and `training.failure_bags`; these lists
explicitly override recorded labels. Empty lists are rejected to prevent an
accidental train-on-everything run. Set `training.allow_all_labeled` to `true`
only when that behavior is intentional. Command-line options remain available
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

`run_benchmark.py` scores the full DINO + RynnBrain (multi-turn) setup against
every bag under the data root that was **not** used as a DINO nominal bag
(`nominal_bags`) or a RynnBrain nominal-demonstration bag
(`rynnbrain.reference_bags`) — both are read straight from `pipeline_config.json`
and automatically excluded. It reuses the existing DINO nominal cache, so build
that once first:

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

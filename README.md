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
if bags are nested more deeply. The label is derived only from recorded stage
messages:

- A stage marker containing `Error`, `Anomaly`, `Fail`, or a related spelling
  gives the whole bag the label `fail`.
- Recorded stage messages without a failure marker give the bag `normal`.
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
output directory. `dataset_labels_details.csv` shows timestamped ignored and
used markers for debugging. By default, stage messages earlier than 0.1 seconds
are treated as cached startup history and do not affect the label.

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

By default, `source` is `generated_videos`, so RynnBrain reads the existing
`<camera>_model_input.mp4` and `<camera>_heatmap.mp4` files in the configured
vision `output_dir`. It does not rerun DINO or decode the test rosbag. Set
`source` to `rosbag` only when direct bag sampling is specifically wanted.

The configuration selects the checkpoint, number of uniformly sampled time
steps, and input modes.
Supported modes are `raw`, `heatmap`, and paired `raw_heatmap`.
`rynnbrain.camera_topics` selects the test viewpoints sent to the VLM and does
not change the top-level DINO camera topics. Set `rynnbrain.task_description`
to the expected behavior in plain language; this avoids processing nominal
videos with the VLM.
The total visual load is approximately `num_frames x number_of_cameras`, or
twice that for paired raw/heatmap input.

Build and run the initial x86 CUDA test service with:

```bash
docker compose build rynnbrain
docker compose run --rm rynnbrain
```

If `task_description` is omitted, RynnBrain can optionally generate one from
`reference_bags` (one bag by default) and save it at `task_memory_path`. This
does not run DINO. A saved description is reused unless
`rebuild_task_memory` is set to `true`.
Selected model inputs, parsed decisions, confidence, full responses, and frame
timestamps are saved under `output_dir`. Each input mode also contains
`vlm_input_storyboard.jpg`, showing the exact images and order sent to the VLM.

The default RynnBrain base is NVIDIA's PyTorch 25.08 container for Jetson AGX
Thor. It can be overridden with `RYNNBRAIN_BASE_IMAGE` when running on a
different NVIDIA platform.

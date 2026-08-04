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
  ..\data --output dataset_manifest.csv
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
Windows and Ubuntu. The large `data` directory is mounted read-only and is not
copied into the image. Generated files are mounted back to `Script_VS/outputs`.

The Docker container is headless. Native Matplotlib windows are therefore not
displayed, but calculations, CSV output, and annotated videos still work.

### Build once

Run from the `Script_VS` directory:

```powershell
docker compose -p alrezomi-ad build
```

The default image installs CPU PyTorch and works without an NVIDIA GPU. Later
builds reuse Docker's cache when dependencies have not changed.

### Run without GPU

Both sections:

```powershell
docker compose -p alrezomi-ad run --rm pipeline
```

Only vision or time series:

```powershell
docker compose -p alrezomi-ad run --rm pipeline vision
docker compose -p alrezomi-ad run --rm pipeline time-series
```

The Hugging Face model cache is stored in a Docker volume, so DINOv2 is not
downloaded again for every run.

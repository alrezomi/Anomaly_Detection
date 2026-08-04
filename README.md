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

## Run with Docker

Docker provides the same Linux, Python, PyTorch, and package environment on
Windows and Ubuntu. The large `data` directory is mounted read-only and is not
copied into the image. Generated files are mounted back to `Script_VS/outputs`.

The Docker container is headless. Native Matplotlib windows are therefore not
displayed, but calculations, CSV output, and annotated videos still work.

### Build once

Run from the `Script_VS` directory:

```powershell
docker compose build
```

The default image installs CPU PyTorch and works without an NVIDIA GPU. Later
builds reuse Docker's cache when dependencies have not changed.

### Run without GPU

Both sections:

```powershell
docker compose run --rm pipeline
```

Only vision or time series:

```powershell
docker compose run --rm pipeline vision
docker compose run --rm pipeline time-series
```

The Hugging Face model cache is stored in a Docker volume, so DINOv2 is not
downloaded again for every run.

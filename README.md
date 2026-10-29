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

### One DINO scoring pipeline

DINO uses one nominal-adjusted patch score and a progress-bin threshold.
There is no local/notebook mode selector. Remove `dino_score_mode` from your local
config; old entries are ignored with a migration notice in build and replay.
The VLM, LoRA, classifier, GMR and camera selection are unaffected.

For each sampled frame (normally 2 fps):

1. Resize to `dino_input_size`, apply the model image normalization, and extract
   patch features, the CLS token and last-layer CLS-to-patch attention averaged
   across heads. L2-normalize every patch vector and the CLS vector.
2. Select the five closest nominal frames by CLS cosine distance, restricted to
   the same recorded stage when `stage_constrained_matching` is true.
3. Compare corresponding patches using RMS distance: Euclidean distance divided
   by the square root of the feature dimension. Combine query/reference attention
   by their maximum, normalize to [0, 1], raise to power 1.5, and apply the existing
   minimum weight 0.10. Attention is a soft weight, not a scissors/object mask.
4. Compute `sum(patch_distance * weight) / sum(weight)` for each candidate and
   retain the lowest-scoring frame as the progress anchor and reference preview.
5. Within the selected frame only, find each observation patch's closest nominal
   patch in its clipped 3x3 neighborhood (one patch in each direction). Also search
   nominal-to-observation. At each grid position take the maximum of the two
   minimum RMS distances, retaining extra and missing-content evidence. There is
   no border wraparound, search across other frames, or unrestricted spatial search.
   At the matched progress, subtract each patch's expected nominal local residual and
   divide positive excess by its regularized nominal standard deviation:
   `evidence = max(distance - mean, 0) / scale`. Multiply directly by the
   attention weight: `weighted_evidence = evidence * weight`. Similar local content contributes little through its small
   residual; weights are not redistributed onto the remaining patches.
6. Produce two attention-weighted maps: the original patch evidence, and its
   edge-padded **3x3 spatial average**. For each map average the highest **1%** of
   values into a separate frame score. Both Euclidean and Mahalanobis mode show
   these two scores; baseline-subtracted/scaled-only scores are no longer charts.
7. Interpolate each score's own smoothed threshold at the matched nominal frame's
   progress. Each chart marks its own `score > threshold` crossings. The existing
   summary, exported video and epsilon use the score **without pooling**. The two
   experimental alarms are not combined.

Set just one new parameter manually in the ignored `pipeline_config.json`:

```json
"dino_metric": "euclidean"
```

Omitting it defaults to `euclidean`. Use `"mahalanobis"` for the alternative.
Only these exact spellings are accepted. Both modes keep CLS/stage frame
retrieval and the one-selected-frame restriction. Both report pooled and unpooled
scores. The 1% fraction is fixed, not another configuration knob.

Mahalanobis mode uses a **regularized diagonal covariance**, not a full covariance
matrix. A fixed, evenly spaced subset of at most 32 DINO channels bounds storage
and fitting cost. Within the selected frame, find forward and reverse 3x3 local
matches by Euclidean distance in that subspace; retain their signed feature
residual vectors. Learn per-patch/per-progress vector means and variances from
other nominal runs, with equal-run moments and sparse-bin shrinkage. Runtime
computes `sqrt(sum((residual - mean)**2 / regularized_variance))` in each direction
and takes their maximum at each patch. Multiply **directly by the existing attention
weight**, without `0.5 + 0.5 * weight`. This is the normalized/powered query-reference
attention weight with the existing minimum of 0.10, rather than raw attention
probability. Euclidean also multiplies directly by this weight. Aggregate the top 1% before
and after 3x3 pooling. This centers/scales the vector
once; it does **not** subtract or divide by the scalar Euclidean profile again.
The diagonal approximation does not model correlations between channels, and
channel subsampling may lose useful information. Its detection accuracy needs lab
validation; it is not guaranteed to improve over Euclidean scoring.

Euclidean thresholds calibrated with softened attention must be recalibrated once.
The internal signature records `attention_weighting: direct`; compatible extracted
features remain reusable. Existing direct-attention Mahalanobis calibration remains
compatible. No new user config parameter is introduced.

The calibration protocol is
`matched_patch_metrics_progress_v7_top1_pooling_smooth`. Old profiles/thresholds and
thresholds from the other metric are incompatible. With `rebuild_nominal_cache:
false`, model-compatible extracted features are reused. Recalibrate with
`vision-memory` after switching metrics, then restart replay. Use distinct
`vision_memory_name` values if you want to retain each metric's calibration;
changing the name to an empty cache requires extraction again. The two scores
share the same feature extraction, matching and normality fitting.
Nominal-only calibration still uses nested whole-recording holdouts.

Replay displays two score/threshold charts: **After attention (no pooling)** and
**After attention (3x3 pooling)**. Each title names the active metric (Euclidean or Mahalanobis) and its pooling
choice. Each has its own dimensionless score and limit.
The per-variant scores, thresholds and alarms are saved as flat CSV columns and
in JSON; exported calibration/replay score plots also show separate axes.

Euclidean heatmap views show raw local residuals, baseline-subtracted residuals,
variability-scaled evidence, attention-weighted evidence, and the effective
attention multiplier. Both pooled and unpooled heatmaps are available; the default
view remains unpooled to match the summary score.
Mahalanobis shows its distance before attention, both attention-weighted maps, and
the attention multiplier; scalar baseline/scaling views are hidden. Raw/subtracted
Euclidean views share one zero-based colour scale, and scaled/weighted views share
another. Do not compare brightness across groups or timestamps. Selecting a
heatmap never changes any graph or decision.

Stop replay and click the object to inspect its numerical patch values. Diagnostics
come from the decision calculation and reject inspection if the observation changes
while fetching. They are retained only for the latest frame and do not rerun DINO.
The diagnostic array `weighted` is the unpooled map; `pooled` contains the actual
3x3 averaged map. Pooling uses edge replication, as in the earlier implementation.

This tolerates movement by one patch, including an object moved by that amount;
it cannot distinguish intentional task-critical small displacements from nuisance
motion. Spatial searches are many-to-one, not object tracking or geometric alignment.
At a 52x52 grid the tolerance is about 1/52 of the image width/height. Recalibration
is slower because it repeats local comparisons during whole-recording holdouts.

The progress axis divides nominal execution into 30 bins; it is not elapsed test
seconds or the test bag's eventual duration. Recorded stages align nominal
progress when available. This accommodates different execution speeds while
retaining the existing same-stage restriction. DINO bag-stage alignment is
retrospective; this replay is not a live ROS subscriber.

The variability profile is learned from `nominal_bags` only. Each training frame
is matched to other complete recordings, using the runtime CLS/attention matcher.
This measures normal *matched residuals*, not unaligned feature variance. Moments
are combined with equal recording weights in each of the existing 30 bins.
Sparse bins shrink toward global per-location moments with three recording
equivalents of regularization. The scale is `sqrt(variance + floor**2)`; the floor
is the larger of the median patch standard deviation, the median nonzero mean
residual, and `1e-6`. Thus nearly constant patches cannot get unbounded weights
from a tiny estimated variance. No region is permanently masked.

This adaptation is informed by [PaDiM's location-specific normal statistics](https://arxiv.org/abs/2011.08785)
and [PatchCore's localized anomaly scoring](https://arxiv.org/abs/2106.08265);
it is not a reproduction of either algorithm or full-covariance Mahalanobis
distance. [DINOv2](https://arxiv.org/abs/2304.07193) provides general visual
features, not task-specific scissors segmentation. [The registers paper](https://arxiv.org/abs/2309.16588)
documents feature/attention artifacts, but its DINOv2 size comparison finds the
high-norm outliers in the large models, not Small. It does not establish that
those outliers caused this experiment's arm hotspots. Keeping the existing Small
extractor preserves reusable features. This method can still miss changes in highly variable or occluded
regions, or alarm under unseen lighting/camera shifts. Lab accuracy remains to
be measured on independent runs.

### Calibrate and reuse DINO memory

To test calibration from the memory demonstrations themselves, set
`"dino_calibration_bags": []` in your local config. Keep the existing
`nominal_bags`, DINOv2 model, input size, and memory name for this first experiment.
Keep `rebuild_nominal_cache: false`: extracted features are reused. Each whole
recording is held out from both matching and patch-normality fitting when its
calibration scores are measured (at least three nominal recordings required).
This is intentionally not self-matching. The nested fitting can take substantially
longer than separate calibration. Save your previous calibration bag list outside
the active config; restoring that list and running `vision-memory` restores the
separate-calibration experiment without re-extracting compatible features.
Use the same `dino_threshold_scale` (start with 1.0) for both comparisons.

For a second, separate experiment, the same scorer supports DINOv3 ViT checkpoints.
Manually change these local config entries:

```json
"dino_model_name": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
"dino_input_size": 832,
"vision_memory_name": "pick_place_dinov3_splus832_nominal",
"dino_calibration_bags": [],
"dino_threshold_scale": 1.0,
"rebuild_nominal_cache": false
```

Use a fresh `output_dir` too, to preserve the earlier graphs. Small+ is a practical
first trial alongside the existing VLM; larger ViTs increase compute and memory
and are not established to be better for this task. 832 / 16 gives the same 52x52
patch grid as DINOv2 at 728 / 14. The extractor removes CLS/register tokens,
retains CLS-to-patch attention, and L2-normalizes patch and CLS features for both
families. ConvNeXt checkpoints are not supported by this CLS-based matcher.
The GPU image already pins Transformers 4.57.1, which supports DINOv3; no VLM
dependency upgrade is needed. DINOv2 remains the default when the new key is absent.

After Hugging Face approves your account, log in **inside the shared Docker cache**
on the lab PC using a read token belonging to that account. Run this interactive
command and paste the token only into its hidden prompt (not into JSON or Git):

```bash
docker compose run --rm --build --entrypoint hf vision-memory auth login
docker compose run --rm --build vision-memory
docker compose up --build online-replay
```

Weights download automatically on the first build and persist in the existing
`huggingface-cache` volume. DINOv3 needs fresh extracted features and calibration;
DINOv2 features cannot be reused across backbones. To roll back, restore the
DINOv2 model name, 728 input size, original memory name and calibration list,
then run `vision-memory` and restart replay. Do not delete either cache.

Evaluate the same **unseen** nominal and failure bags after each experiment.
Select **Nominal-adjusted differences** in the heatmap menu to see scored evidence;
attention weights alone are not an anomaly map. Record false alarms on nominal
runs and whether the object-related failure crosses its threshold. These changes
enable a controlled comparison; neither a newer backbone nor nominal-only
calibration guarantees better detection. Do not tune repeatedly against the final
evaluation set and then report it as an independent test.

References: [DINOv3 paper](https://arxiv.org/abs/2508.10104),
[Meta model collection](https://huggingface.co/collections/facebook/dinov3),
[Transformers 4.57.1 DINOv3 support](https://huggingface.co/docs/transformers/v4.57.1/model_doc/dinov3).
The paper's dense-feature improvements motivate this experiment; it does not
validate our robot anomaly scorer or its thresholds.

`nominal_bags` supplies retrieval memory. Optional `dino_calibration_bags` supplies
separate successful recordings, scored by the exact same matcher. Those queries
never enter retrieval memory or the patch variability profile. At least two
nominal memory recordings are required. Without a separate calibration list,
at least three nominal recordings are needed: each outer held-out recording is
excluded from both retrieval and variability fitting, then the final profile is
fit on all memory recordings. Separate calibration is preferable and avoids the
extra fitting cost and reference-size difference of this fallback.
The previous local method's minimum of 18 calibration bags no longer applies.
The supplied 16 calibration recordings can be used by this method.

Within each progress bin, compute a 95th percentile for each recording, then
the 95th percentile across those values. These quantiles select score values;
they do not multiply the scores by 0.95. Every recording has equal
weight. Bins without two recordings/two frames interpolate populated bins; if all
bins are sparse, the same two-level quantile is computed globally. The floor is
the median of recording-level 20th-percentile frame scores. Three-bin
smoothing retains every valid nominal-bin bound. It raises neighbors of high bins
until adjacent thresholds differ by at most a factor of 1.25, then rounds the ramps
with a short symmetric moving average, keeping the upper envelope. This can raise
thresholds around nominal peaks. It smooths along progress, not elapsed replay time;
a jump in the matched progress can still jump over several bins.
These are empirical frame/run quantiles, not a conformal or guaranteed
whole-execution false-alarm rate.
Evaluate the result on held-out normal and failure recordings, including delay.

`dino_threshold_scale` is the existing sensitivity control: 1.0 uses calibration,
0.8 lowers the bound by 20%, and 1.1 raises it by 10%. It changes the decision
limit only and requires a replay restart, not recalibration. Use 1.0 for an initial
baseline. This is the sole multiplicative gain; the previous fixed 1.10 margin
has been removed. Compatible legacy calibration is converted on load by rebuilding
the small quantile tables from saved scores, with no feature extraction or patch
profile fitting. Old files and benchmark reports remain unchanged. Multiplying
the previous scale by 1.1 compensates for the removed margin where the quantile
bound dominates; floor-dominated bins can differ because the floor was never
multiplied by that margin.
The scored heatmap now shows the same unpooled attention-weighted evidence used for the
score. Replay uses shared scales for the diagnostic groups described above;
exported batch videos retain their threshold-relative scale. Colors are not
probabilities, and the final decision uses the aggregated score.

After pulling this update, stop replay and recalibrate once:

```bash
docker compose stop online-replay
docker compose run --rm --build vision-memory
docker compose up --build online-replay
```

Keep the existing `vision_memory_name`, `dino_input_size`, bag lists and
`rebuild_nominal_cache: false`. Previous thresholds, including the former unified
whole-image score, are rejected by the new protocol signature. Compatible
extracted features are reused, including
per-recording calibration features under `calibration_features`; missing features
are extracted once. Recalibration does not retrain LoRA or the CoP classifier.
A fully compatible cache loads without initializing DINO during memory building.
Test/replay never builds memory or silently uses an incompatible threshold.

| Change | Feature extraction | Score/threshold calibration |
| --- | --- | --- |
| GUI, logging, output experiment folder | Reuse | Reuse |
| `dino_threshold_scale` | Reuse | Reuse; runtime multiplier |
| Attention weights, CLS shortlist, stage matching restriction | Reuse | Recompute |
| Threshold-fitting settings | Reuse | Recompute |
| Model, resolution, camera, sources, sampling, stage assignment | Extract incompatible features | Recompute |

Keep evaluation bags separate from memory and calibration; replay excludes both.
`calibration.csv` records scores, matched progress and crossings;
`adaptive_threshold.csv` records the curve and bin counts. These are calibration
diagnostics, not independent evaluation results. `patch_normality.npz` stores the
learned means/scales; `patch_reliability.png` shows their relative reliability
for every progress bin, using one fixed color scale. Missing/incompatible profiles
cannot silently fall back to the old score. `raw_global_score` in result rows
retains the former whole-image distance for diagnosis. Changing the memory name selects
another directory rather than searching for reusable files elsewhere. Replacing
bag contents under the same path requires `rebuild_nominal_cache: true` because
feature signatures track sources/settings rather than hashing all bag contents.

### DINO benchmark for selected folders

Use the separate `dino-benchmark` service for pooled Euclidean/Mahalanobis
statistics. It does not load RynnBrain/LoRA or run the existing VLM benchmark.
Manually add this one selection block to your ignored `pipeline_config.json`:

```json
"dino_benchmark": {
  "folders": [
    "Nominal/setting 1",
    "Failure_1_grasp_miss"
  ],
  "labels_csv": "/outputs/dataset_labels.csv"
}
```

Paths are relative to the container data root `/data`; absolute paths inside
that root are also accepted. List your actual desired folders. Subfolders are
searched recursively for ROS 2 `metadata.yaml` files. A bag folder itself can
also be selected. There is **no default scan of the entire data root**: empty,
root-level, overlapping or outside-root selections are rejected. Bags in
`nominal_bags` or `dino_calibration_bags` are excluded by full resolved path,
even if their parent folder is selected. No folder-name-based labels are used.

```bash
docker compose stop online-replay
docker compose run --rm --build dino-benchmark
```

Alternatively, override the selection for one run without editing config:

```bash
docker compose run --rm --build dino-benchmark --config /config/pipeline_config.json --folder "Nominal/setting 1" --folder "Failure_1_grasp_miss"
```

The command first builds/reuses nominal features and calibrates **both** metrics
using your current `nominal_bags`, optional `dino_calibration_bags`, DINO model,
input size and camera topics. `dino_metric` does not restrict this comparison.
Keep `rebuild_nominal_cache: false` for reuse. Both metrics share nominal and
separate-calibration extracted features. Compatible existing online calibration
can be reused; otherwise benchmark calibrations are stored independently under
`<nominal_cache_dir>/<vision_memory_name>/<camera>/benchmark_calibration/<metric>`.
The benchmark does not replace the online metric's calibration. Subsequent runs
reuse both benchmark calibrations. Nominal-only calibration still performs the
existing expensive nested recording holdouts when a calibration is missing.

Each sampled test image is extracted once and its representative nominal frame
is selected once. Both metrics score the same features/match. Decisions use
**attention-weighted 3x3 pooled evidence only**, averaging its highest **1%**.
The no-pooling score is not used for any benchmark classification or ROC point.
A bag is predicted **failure only after two consecutive sampled images in the same
camera exceed their pooled thresholds**, strictly before the recorded **start of
stage 3**. The alarm timestamp is the second crossing, when it can be confirmed.
One spike, equal-to-threshold values, cross-camera alternating crossings, or
crossings at/after stage 3 do not confirm an alarm. Any one camera can confirm.
Scores after stage 3 remain visible but are shaded/ignored for evaluation.
This camera OR is a benchmark bag decision, separate from independent replay
camera plots. All images run at the existing DINO sampling rate (normally 2 fps),
without replay pacing, VLM intervals, early stopping, or an eight-image limit.

When `labels_csv` is set (or `--labels-csv` is supplied), that CSV is the
**authoritative ground truth**. It needs `bag_path,label` columns; paths may be
absolute or relative to `/data`. A unique `bag_name` can be used when paths are
unavailable, but duplicate basenames across selected folders are rejected.
Labels are `normal`/`fail` (success/nominal/failure aliases are accepted). The
benchmark automatically creates/updates the labels CSV before model preparation:
missing selected bags are added using their recorded stage labels. Existing
entries, manual corrections, onset columns, and unselected rows are preserved.
No separate `bag-labels` command is needed. Unknown or duplicate labels still
stop the benchmark. Stage-derived labels
remain in the audit columns and cannot silently overwrite CSV labels.

When no CSV path is configured, it defaults to `dataset_labels.csv` under
`ANOMALY_OUTPUT_ROOT` (Docker `/outputs`), or the parent of the experiment output
directory outside Docker. New rows use the existing final-three eligible
stage-message policy from `build_dataset_manifest.py`. A `Failure_*` folder name is never a label.
Only selected, eligible bags are added; other dataset folders are not scanned.
To regenerate existing recorded labels deliberately, use the separate `bag-labels`
command, which overwrites its output rather than preserving edits.
**Preflight requires known labels before memory preparation or model inference.**
Single-class evaluation is supported: decisions, alarm times and the applicable
class rate are reported, while ROC and best-scale selection remain unavailable.
Task success and visual normality are different ground truths. A completed task
with an unrelated object may correctly have a normal task label and still trigger
DINO. Against task labels that alarm counts as a false positive; this does not
establish whether it correctly detected a visual change. For visual-anomaly recall
and ROC, supply a separate labels CSV identifying visually changed cases as fail
and visually normal cases as normal. Preserve the original task-outcome CSV.
Never change task labels merely to obtain an ROC.
For DINO, actual onset is the first explicit Error/Fail/Failure/Anomaly-family
marker at or after the corrected stage-1 start. Earlier unrelated markers are
ignored, even if later than the startup-ignore interval. All stage messages,
the corrected stage boundaries and the number of ignored early failure markers
are retained for audit. Optionally, the
labels CSV can supply an explicit bag-relative `failure_time_sec` (or
`recorded_failure_time_sec`) for failure bags. Missing markers/timestamps remain
missing; a failure label or folder name does not supply an onset time. Both clocks are
relative to bag start. When stage-constrained matching is enabled, scoring begins
at the corrected stage-1 marker and retains retrospective numbered-stage matching.
Failure markers are used for evaluation, never for matching or metric fitting.
A normal final label with an earlier failure marker is flagged in `selection.csv`.
Unknown labels are excluded from correctness/recall/FPR, and missing camera/data
errors invalidate the entire bag rather than becoming a success or a partial-camera
result. Those rows and errors remain visible. Failures without a detection or a
valid onset never get a fabricated zero timing error.

Reports are written into a new timestamped directory under
`<output_dir>/dino_benchmark/` (Docker `/outputs` maps to `ANOMALY_OUTPUT_DIR`).
Open **`SUMMARY.md` first**. Report format v4 creates separate `euclidean/` and
`mahalanobis/` folders. Each contains `summary.md`, evaluation CSVs,
`roc.png`, `alarm_rate_vs_scale.png`, and `bag_index.csv`.
At the root, **`failure_time_difference.png`** shows annotated demos from all
folders and both metrics together on one signed-delay axis, using the configured
scale. A valid recorded onset is compared regardless of the task-success label;
any label/marker conflict remains flagged without relabelling. If no onset is
available, **`first_alarm_time.png`** replaces the difference graph and shows
actual first-alarm times, never invented delays. **`metrics.png`** shows only
accuracy and FPR for normal-only runs; irrelevant recall/precision/F1 panels and
unavailable ROC images are omitted. Mixed-class runs retain the full metric graph.
`timing_diagnostics.csv` records raw stage values/times, selected onset, ignored
pre-stage-1 marker count, first alarm, comparison status and reader/refresh errors.
The console prints onset availability before expensive inference starts.
Every evaluated/error bag has `bags/<name>_<path-hash>/score_threshold.png`, its
frame CSV and its decision CSV. Separate camera panels show the actual pooled
score and configured threshold in bag-relative seconds, crossings, first alarm,
and the recorded failure onset when available. Errors remain visibly incomplete.
Training/calibration bags remain excluded and are listed in the selection/results.
The original combined CSVs remain at the root for compatibility.

Single-class sets still produce an alarm-rate-versus-scale curve and per-folder
alarm rates. They cannot produce a true ROC, recall for an absent positive class,
or an optimal scale balancing sensitivity/specificity. The ROC image explains
this without drawing empty axes or a chance line. `metrics.csv` now separates
training exclusions and errors and includes `alarm_bags`, `no_alarm_bags` and
`alarm_rate`, so these can be read independently of the chosen label semantics.

The console prints report format/source identity, cache directories, rebuild
status, exact incompatible signature keys, and calibration reuse/fitting times.
`report_manifest.json` lists generated artifacts and the report writer hash;
`run.json` records the format version and benchmark source hash. A log claiming
fixed calibration `x 1.1` is from code before the single-gain update. Compatible
old margin-only caches are reused via saved scores, without fitting patch profiles.

| File | Contents |
| --- | --- |
| `selection.csv` | Selected folders/bags, training exclusions, recorded labels, onset markers and errors |
| `bag_results.csv` | Both metrics, configured and exploratory-best scale decisions, correctness, first alarm, signed timing error and status |
| `frame_results.csv` | Per-camera sampled pooled scores, unscaled thresholds, configured/best alarms, matched frame/progress and processing times |
| `metrics.csv` | Overall and each selected folder: TP/FP/TN/FN, recall, FPR, precision, accuracy, F1, balanced accuracy, Wilson 95% intervals and failure-timing statistics |
| `roc.png`, `<metric>_roc.csv` | Bag-level ROC/AUC, with configured and exploratory-best scale marked |
| `<metric>_scale_sweep.csv`, `roc_summary.json` | Threshold-scale sweep and suggested scale for each metric |
| `failure_time_difference.png` or `first_alarm_time.png` | Signed delay when actual onsets exist; otherwise first-alarm times only |
| `timing_diagnostics.csv` | Recorded stage messages and timing audit, including ignored pre-stage-1 errors and comparison status |
| `metrics.png` | Accuracy/FPR for normal-only data; full metric comparison for mixed-class data |
| `runtime.csv`, `runtime_summary.csv` | Model load, feature preparation/reuse, calibration, actual bag-inference wall times, and frame mean/median/p95 by metric/folder |
| `run.json` | Effective model/memory signatures, selection/label policies, run status, ROC selection and total wall time |
| `alarm_diagnostics.csv` | Alarm-frame counts/fractions, first-frame crossings and first-alarm camera, for distinguishing isolated crossings from systematic startup alarms |

Both the configured and suggested scales are reported overall and per selected
folder. Recall is TP/(TP+FN); FPR is FP/(FP+TN). Rates are blank when their class
is absent, rather than incorrectly reported as zero. Include unseen nominal bags
as well as failure bags to obtain an ROC curve. Runtime feature/matching values
are shared by the metrics and should not be added together; per-metric processing
time is feature extraction + representative-frame comparison + that metric's
scoring. Read/decode time is separate. Phase durations can overlap (for example model load
inside preparation or first-bag inference); `run.json` records the actual total
wall time. These are offline throughput measurements,
including cold first-frame effects, not live end-to-end ROS alarm latency.

ROC ranks each bag by the largest minimum `score/base_threshold` of adjacent
eligible frame pairs in one camera. This represents the two-crossing rule, not
an isolated maximum. The scale sweep uses the actual strict comparisons for both
frames before stage 3 (equality is no alarm), so tuning needs no repeated model inference. The
suggested scale maximizes Youden J = recall - FPR, preferring lower FPR, usable
interior/configured scales, and then proximity to the configured scale when
performance ties. The JSON/CSV preserve full precision. Standard ROC endpoints may be
unreachable by a positive scale for zero scores/zero thresholds; such cases are
counted explicitly and the suggested operating point is evaluated using real
positive finite scales. Scale suggestions are **exploratory tuning on the same
selected bags**, not independently validated performance. Nothing rewrites your
config. Use another held-out set to evaluate a chosen scale. The combined timing
figure uses the configured scale; both operating points remain in `bag_results.csv`
because successful demonstrations have no actual failure onset to subtract.

### Recover DINO reports without repeating inference

If an existing run has valid `frame_results.csv`, regenerate the full report
layout without repeating inference. This also supports correcting labels with
an authoritative CSV; omit `--labels-csv` when keeping saved labels and no CSV
is configured:

```bash
docker compose run --rm --build dino-benchmark \
  --config /config/pipeline_config.json \
  --report-only /outputs/experiments/YOUR_EXPERIMENT/dino_benchmark/YOUR_RUN \
  --labels-csv /outputs/dataset_labels.csv
```

This reads the saved selection, pooled frame scores and unscaled thresholds.
It checks known labels, saved frame counts and complete camera coverage. It does
not extract features, calibrate memory or load DINO. It refreshes only recorded
stage metadata so old startup-based onset annotations are corrected to stage 1;
the original bags must be available for this refresh. If a bag is unavailable,
valid saved annotation data is retained and the refresh error is explicit.
Explicit CSV onset annotations are preserved.
It preserves the original configured scale, selection and original outputs, writing a new
`reports_<timestamp>/` subdirectory with corrected metrics, ROC, timings and alarm
diagnostics. CSV labels change evaluation truth, never detector scores. If the
CSV labels all evaluated bags normal, reports still show the detector decisions
and task-label false-positive rate; ROC and best-scale selection are unavailable.
An absent onset marker remains unavailable unless explicitly supplied in the CSV.
High nominal false-alarm rates can remain after relabelling: the bag rule still
uses confirmed two-frame crossings before stage 3, and the suggested scale is still exploratory tuning.
Report-only recovery uses the original stored scale and thresholds, even if the
current config or calibration has changed. It does not retroactively apply the
single-gain change to an old experiment. Generating many plot images takes CPU
time, but requires neither DINO inference nor nominal-memory loading.
Report-only applies the current alarm confirmation/window policy to saved raw
scores, so bag decisions, onset errors, metric graphs and ROC are recomputed together.
If a labels CSV is supplied/configured for report-only, missing entries are filled
from saved stage annotations without ROS reads. This does not add new recordings
to the old run; newly selected recordings require a fresh benchmark.

### Independent top and flange DINO replay

DINO uses the **top-level** `camera_topics` list, just like `vision-memory`.
Every configured camera gets its own memory, calibration, heatmaps and graph.
One loaded DINO model is shared; frames from both cameras are scored sequentially
in timestamp order, including simultaneous timestamps. Each camera keeps its
native 2 fps sampling schedule. No camera scores are fused or compared against
another viewpoint's memory. A missing camera/topic or incompatible cache is an
explicit error. Processing lag is shown rather than skipping sampled images.

For this experiment, manually change these existing top-level entries:

```json
"camera_topics": [
  "/top_camera/cam_135222070433/color/image_raw/compressed",
  "/flange_camera/cam_230422272745/color/image_raw/compressed"
],
"dino_input_size": 896,
"vision_memory_name": "pick_place_dino896_v1",
"dino_threshold_scale": 0.9
```

`0.9` lowers the calibrated bound by 10% for the requested more sensitive test.
It can increase false alarms. Use `1.0` for the unscaled comparison; the existing multiplier
requires only a replay restart, not recalibration. Changes to the scoring formula
require fresh calibration. Never change scores without recalibrating the same
metric on nominal recordings.

**Keep the VLM top-only.** Inside your existing `rynnbrain` object, explicitly set
both entries below, retaining all its other settings. If these are omitted,
camera inheritance would also send the flange view to the VLM.

```json
"camera_topics": ["/top_camera/cam_135222070433/color/image_raw/compressed"],
"memory_camera_topics": ["/top_camera/cam_135222070433/color/image_raw/compressed"]
```

Keep the nominal/calibration bag lists separate and unchanged; those recordings
must contain both camera topics. Run `docker compose run --rm --build vision-memory`
once, then `docker compose up --build online-replay` and select **DINO vision**.
The new memory name preserves the old resolution's cache. Both cameras need
new features at 896 and independent recalibration. Two cameras increase total
inference work and host RAM for memory; GPU weights are shared. Measure processing
delay on the lab GPU; no real-time throughput guarantee is implied.

Saved `online_scores.csv`/`online_results.json` label every row with `camera_topic`;
timeline PNGs are exported separately for each camera. The browser view selector
only changes the displayed heatmap, not the score. No user config is overwritten
by this code update.

Attention does not identify the target object or guarantee detection under
occlusion. Validate on separate successful and failed recordings. Regression
tests check normalization, attention weighting, stage restrictions, progress
thresholds, cache reuse and calibration/replay/batch/video agreement.

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
| Benchmark evaluation outputs | `rynnbrain_benchmark/timeline/` or `rynnbrain_benchmark/whole_demo/` |
| Shared classifier-training vectors and responses | `rynnbrain_benchmark/classifier_training/` |
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
never accumulates a queue of old frames. By default, the eight selected observations may
cover only part of a long bag. Change the interval deliberately for your task;
it is not automatically chosen from the bag duration. There is no rolling
window or cache eviction in this version.

To experiment with adaptive VLM sampling beyond eight observations, add this
object inside `rynnbrain` (keep `num_frames: 8` and your existing model/classifier paths):

```json
"online_replay": {
  "interval_mode": "adaptive",
  "max_observations": null,
  "stop_on_alarm": true
}
```

`adaptive` selects the next interval from the latest logistic failure score:
below 30%: 10 seconds; 30% to below 50%: 7 seconds; 50% to below 60%: 5 seconds;
60% and above: 3 seconds. The browser interval field is disabled in this case.
`constant` uses the browser interval instead. `stop_on_alarm` stops replay at the
first score greater than or equal to the classifier's configured threshold,
independently of the interval bands. All observations up to and including that
alarm, vectors, CSV/JSON and plots are saved. This stops replay, not a physical robot.

`max_observations: null` removes the observation-count limit; replay ends at the
bag/window end, manual stop, or alarm. A positive integer sets a limit. Omitting
the object preserves constant intervals, the `num_frames` observation limit and
no automatic alarm stop. These options affect browser VLM replay only; DINO,
GMR, training and offline benchmarks are unchanged. Use `inference.feature_only: true`
for the intended fast feature-only experiment. Adaptive replay requires a compatible
logistic classifier; kNN prefix-index detectors do not support extended histories.

The nominal reference retains its trained image count, while the execution
cache keeps every observed frame, including observations beyond eight. There
is no rolling window or reset at observation eight. Context length and memory
usage therefore grow; scores beyond the training history length are experimental.
No retraining is launched by replay. Saved rows include `next_interval_sec`, and
the JSON records `termination_reason` and `alarm_timestamp_sec` when applicable.

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

Base-model VLM browser replay labels every nominal-reference and execution image
with its actual recorded time in seconds from that image's own bag start. Temporal
LoRAs retain their trained timestamp format. Both paths use `temporal=True`;
reference and execution clocks remain separate.

When explicitly loading a legacy LoRA without temporal metadata and its matching
frame-order-only classifier (for example v102/v131), replay restores the exact
reference/evaluation prompts saved in the classifier and the original frame-order
labels. It prints this selection at startup; it does not edit the prompt script,
configuration or model files. All other feature settings, the loaded adapter
fingerprint, and the classifier's prefix companion are still checked normally.
Actual capture times remain in replay logs and saved rows; legacy inputs do not
inject new seconds labels into the model. This preserves the saved feature
extractor while allowing adaptive sampling, extended histories and alarm stopping.

When `rynnbrain.inference.feature_only` is `false` (also the default when omitted),
browser VLM replay additionally generates an answer for **each sampled observation
history**, starting with the first frame. The server prints the bag name,
observation number, timestamp, and unmodified model answer to its terminal/Docker
logs. It also saves `vlm_response`, `vlm_response_source`, `vlm_response_error`, and
`vlm_generation_sec` in `online_results.json` and `online_scores.csv`.
This is a separate generation pass over the same reference and frames observed
so far; it adds latency and can change which elapsed-time observations are sampled.
The classifier score still comes from the cached prompt vector. Generated answers
are not fed back into the scoring history or used to replace its decisions.
An optional text-generation failure is logged without discarding the score.
`feature_only: true` skips these extra calls. The learned adapter determines the
answer format; enabling text does not itself add visual explanations to an adapter
trained only on decisions and onset times. This does not enable per-prefix text
generation in the offline benchmark, whose existing final-answer behavior is unchanged.

With text enabled, a missing, disabled or incompatible classifier is reported at
startup and replay continues with model text only. Scores, thresholds and alarm
decisions are unavailable (saved as null), not zero or invented probabilities.
The selected base model or explicit LoRA is unchanged. A fresh nominal-reference
response is generated with that model rather than reusing an incompatible
classifier's reference response. With `feature_only: true`, classifier errors
still stop replay. Existing compatible classifiers continue scoring normally.

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

To evaluate only particular folders, manually add this optional top-level block
to your ignored `pipeline_config.json` (independent of `dino_benchmark`):

```json
"vlm_benchmark": {
  "folders": ["Nominal/setting 1", "Failure_1_grasp_miss"],
  "decision_mode": "whole_demo"
}
```

Only those subfolders under the data root are scanned for evaluation. Omit the
block, omit `folders`, or use `"folders": []` to retain the existing scan of all
bags under the data root. Existing memory/reference/LoRA/classifier-training
exclusions still apply. Paths cannot escape the data root or overlap; CLI
`--folder` (repeatable) overrides the config list. Existing `--bag`, manual label
overrides, `--limit` and `--no-recursive` further restrict the selected folders.
Folder validation occurs before classifier/model preparation. Classifier training
and nominal reference settings remain independent of evaluation-folder selection.

### Unattended LoRA, classifier, and benchmark run

To train a fresh LoRA adapter and then automatically train the classifier(s)
and evaluate the selected folders, add `train_lora: true` to the existing
top-level `vlm_benchmark` object:

```json
"vlm_benchmark": {
  "train_lora": true,
  "decision_mode": "timeline",
  "folders": ["Nominal/setting 1", "Nominal/setting 2", "Failure_1_grasp_miss"]
}
```

Keep your own folder selection, prompt variant, task description, LoRA/classifier
training lists and hyperparameters. In particular, keep
`rynnbrain.lora.training.temporal_supervision: true` if you want recorded onset
times and prefix supervision. This workflow uses the existing trainer and targets;
it does not alter prompts, numeric-onset supervision, or graph generation.
The flag is off when omitted/false, preserving the ordinary benchmark behavior.

Use the usual benchmark command (stop the browser's model first to release GPU memory):

```bash
docker compose stop online-replay
docker compose run --rm --build benchmark
```

No additional mode or container name is needed. Terminal output is also saved
to `benchmark_run.log`.

Each automatic run creates a unique directory under
`<rynnbrain.output_dir>_benchmark/pipeline_runs/<UTC timestamp>_prompt_<a|b>/`
(or `<--benchmark-dir>/pipeline_runs/...`). Inside it are:

- `requested_config.json`: the original resolved configuration for provenance.
- `lora_training_config.json`: the training snapshot, starting from the base model.
- `lora_temporal_adapter/` or `lora_adapter/`: the newly trained adapter.
- `cop_classifier/` and `classifier_training/`: fresh classifier models and features.
- `timeline/` or `whole_demo/`: the usual per-bag outputs, graphs and reports.
- `pipeline_config.json`: the effective configuration pointing to the new models.

An existing `model.lora_adapter_path`, LoRA output directory, or classifier path
in the user's config is replaced **only in the isolated run's snapshots**. Existing
artifacts and the user's config file are not overwritten. The workflow enables
classifier preparation for this run even if `train_on_benchmark` was false.
The LoRA process must exit successfully and produce a trained checkpoint before
the classifier or benchmark starts; failure never falls back to an older adapter.
Training runs in a child process to release GPU memory before inference loads.
`benchmark_runtime.json` records `lora_training_status`, `lora_training_sec`,
the trained adapter path and the effective configuration path, alongside the
existing classifier-preparation and inference timings.

After successful completion, the saved `pipeline_config.json` disables both
`vlm_benchmark.train_lora` and `cop_classifier.train_on_benchmark`, allowing you
to load that configuration to reuse the new adapter/classifiers without training
again. Your original config retains its flag; running it again with
`train_lora: true` creates another fresh training run. The configured reference,
LoRA-training and classifier-training bags remain excluded from test evaluation.

The regular browser command can use your original configuration after automatic
training finishes:

```bash
docker compose up --build online-replay
```

With `train_lora: true`, replay discovers the newest **completed** automatic run under this experiment's
`pipeline_runs/` for the selected prompt variant and classifier method. It loads
the adapter and classifier paths together, and prints the chosen run and paths
at startup. This also works for runs completed before this discovery feature was
added; no retraining or manual timestamp-path editing is needed. Replay never
trains, regardless of the benchmark training flags. Your current inference
options, threshold, prompts, cameras and task description remain in effect;
the existing feature/classifier compatibility checks still apply.

With `train_lora: false`, replay and ordinary benchmark runs use only the configured
adapter and classifier paths. `lora_adapter_path: null` selects the base model;
no saved LoRA is discovered, even when configured classifier files are missing.
Missing files or incompatible classifiers produce an error rather than selecting
another model. Use classifiers trained with the selected base model or adapter.
The original configuration is not rewritten.

Video/heatmap LoRA inputs still require the existing prepared training videos;
this option does not create those videos or rebuild DINO memory.

### Compare prompt A and B with a fixed execution interval

The ordinary benchmark command can queue two independent fresh training runs.
Set these fields in your existing configuration (retain your bag lists, model
settings and other parameters):

```json
{
  "vlm_benchmark": {
    "train_lora": true,
    "prompt_variants": ["A", "B"],
    "decision_mode": "timeline",
    "threshold_search": false
  },
  "rynnbrain": {
    "prompt_variant": "A",
    "execution_interval_sec": 3.0,
    "inference": {"feature_only": true},
    "online_replay": {
      "interval_mode": "constant",
      "interval_sec": 3.0,
      "max_observations": null,
      "stop_on_alarm": true
    },
    "lora": {"training": {"temporal_supervision": true}},
    "cop_classifier": {
      "enabled": true,
      "method": "logistic",
      "logistic_onset": {"enabled": true}
    }
  }
}
```

Choose a new experiment `output_dir`. Then run:

```bash
docker compose stop online-replay
docker compose run --rm -T --build benchmark
```

Each variant starts from the base model, trains its own LoRA and classifier, and
evaluates held-out bags. The runs execute sequentially in separate processes so
the preceding VLM releases its GPU memory. Existing training/reference exclusions
apply, and the comparison checks that both evaluated bag/label sets are identical.
Failures stop the queue and retain completed outputs.

`execution_interval_sec` selects the first available camera image at or after the
next due recording time, then schedules the next deadline from that selected
image's actual time. LoRA examples include every growing prefix; classifier
training uses those same prefixes with onset labels. Evaluation uses the GUI's
cached detector without wall-clock waits. No eight-observation execution cap is
applied. `num_frames` continues to control the uniformly sampled nominal reference.
The current implementation supports one raw camera, temporal LoRA and logistic
onset classification. Training checks the longest history in every bag against
`lora.training.max_tokens` before optimizer updates; it never silently truncates
images. More observations increase training time and GPU memory requirements.

Edit the **temporal** branches of the existing A/B prompt functions before running.
The code does not rewrite the prompts. It saves the exact rendered text in
`prompt_texts.json` and each adapter's manifest, rejects identical A/B text before
training, and checks that the text has not changed between queue preparation and
training. The repository's default B delegates to A until you customize it.

Outputs are isolated under
`<rynnbrain.output_dir>_benchmark/prompt_comparisons/<timestamp>_interval_3s/`.
`prompt_comparison.csv` and `.json` link each variant's model pair, reusable config,
full reports, accuracy, threshold and mean processing time. Each prompt retains
the usual per-bag outputs and combined graphs under its own `pipeline_runs/` folder.
With `threshold_search:false`, both use your configured classifier threshold.
With `threshold_search:true`, each runs the existing timing-cost threshold search.
Threshold selection on these bags remains a development result; use separate
recordings to assess a final chosen threshold.

After both runs finish, the original config with `train_lora:true` lets replay
discover the completed comparison model pair matching `rynnbrain.prompt_variant`.
Choose `"A"` or `"B"` there to inspect either pair; keep `execution_interval_sec:3`.
Each saved `pipeline_config.json` also contains explicit paths with training disabled.

The VLM nearest-nominal preview checkbox has been removed from the browser and
defaults off. Configured nominal reference demonstrations remain part of VLM input.
DINO retains its own matching and scoring.

### Offline timeline decision mode

Set `vlm_benchmark.decision_mode` to `"timeline"` to benchmark **alarms during
execution**, instead of the final whole-demo classifier outcome. Omission or
`"whole_demo"` preserves the original final-score decision rule. Run the same command:

```bash
docker compose run --rm --build benchmark
```

Timeline results go into `<rynnbrain.output_dir>_benchmark/timeline/` (or
`<--benchmark-dir>/timeline/`). Whole-demo runs now use the parallel
`<rynnbrain.output_dir>_benchmark/whole_demo/` folder (or
`<--benchmark-dir>/whole_demo/`). Existing reports in the parent directory are
left in place; new runs do not move or rewrite them.
Classifier preparation has a separate, shared default directory:
`<rynnbrain.output_dir>_benchmark/classifier_training/<training_bag>/rynnbrain_multiturn/`.
Those bags are training inputs, not duplicate evaluated demonstrations. Keeping
them shared avoids extracting them again just because the decision mode changes.
Explicit `cop_classifier.training.input_dir` overrides are still honored. Compatible
training artifacts from the older loose-bag layout beside `timeline/` are read in
place when needed; refreshed/new training artifacts use the grouped directory.
Existing folders, graphs, and outputs are never deleted by this layout change.
Each observation is scored using only the selected current and preceding execution
images, without waiting for recording time to pass. This uses existing offline
prefix sampling (normally 1, 2, ... 8 observations), not the online replay's
elapsed-time sampling. It does not add model calls beyond the existing timeline
diagnostics, but prefix inference is more expensive than a final score alone.

A bag is predicted failure if **any prefix score is >= its own threshold**,
including the final prefix; otherwise a complete timeline predicts success.
This mode requires an enabled `rynnbrain.cop_classifier` and automatically enables
its timeline scoring, even if `plot_timeline` is false. It keeps the configured
classifier method and onset settings: logistic uses the separate onset classifier
when `logistic_onset.enabled` is true, otherwise the whole-demo classifier on
prefixes. The latter remains an offline diagnostic, not a validated onset model.
kNN uses its existing matched nominal-prefix detector and thresholds.
No two-alarm confirmation or stage-3 cutoff is applied to this VLM protocol.

`benchmark_summary.csv`, folder metrics/graphs and classifier ROC/score reports
use timeline decisions. `timeline_first_alarm_time_sec` records the first crossing;
existing timing reports compare it with a recorded failure marker when available.
`whole_demo_classifier_*` columns retain the original final score and decision.
Per-bag `rynnbrain_results_multiturn.csv` still contains the original whole-demo
result; the accompanying timeline CSV/PNG contains each prefix evaluation.
Generated-language decisions, if enabled, remain separately labelled and unchanged.
`benchmark_clean.csv` includes both classifier decisions for comparison.

For logistic ROC, each bag's score is its **maximum prefix failure score**;
this is not a calibrated probability of any failure in a bag. For kNN ROC, the
score is the maximum `prefix_score / prefix_threshold`, with alarm threshold 1,
because the threshold varies with prefix length. Incomplete no-alert timelines
are unavailable, not successes. A crossing on a partial timeline still proves an
alarm, but incomplete timelines are excluded from peak-score ROC and timing means.
`benchmark_protocol.json` records the rules. Switching decision modes alone does
not require retraining the VLM, LoRA or classifier; the existing
`cop_classifier.train_on_benchmark` setting still controls classifier preparation.

The logistic decision cutoff uses the existing
`rynnbrain.cop_classifier.training.threshold` setting, for example `0.6`.
It applies consistently to whole-demo, offline-prefix/onset, and online-replay
decisions even when `train_on_benchmark` is false. Changing only this cutoff does
not change feature vectors, weights or probabilities, and needs no retraining.
If omitted, the loaded classifier's saved threshold is used. Restart replay after
changing the config. kNN continues to use its learned nominal thresholds.

To keep two editable prompts, set `rynnbrain.prompt_variant` to `"A"` (default)
or `"B"`. Prompt A retains the existing instructions exactly, including temporal
and heatmap forms. Edit `task_context_prompt_b` and
`evaluation_prompt_multiturn_b` in `rynnbrain_vlm/prompts.py` to write B's nominal
and evaluation turns; B initially returns A's text. Keep the generated-answer
format if language decisions/timing are enabled. The selector is used consistently
by training, feature extraction, scoring, replay and feature verification.

For B, the configuration loader automatically isolates result directories and
classifier files with `_prompt_b` names, including explicit configured paths;
training-vector and newly trained LoRA output paths are isolated too. The loaded
`model.lora_adapter_path` is unchanged, so B can first be tested with the same
adapter. Switch back to A in the original config to recover A's paths. The actual
rendered prompts are already part of classifier/vector compatibility signatures:
editing B requires extracting compatible B vectors and fitting its classifier
(`train_on_benchmark: true` prepares these). It does not automatically retrain LoRA.

`benchmark_metrics.csv` and `benchmark_metrics.png` compare **overall and every
evaluation folder**, separating generated VLM decisions, optional classifier
decisions, and input modes. They include TP/FP/TN/FN, recall, FPR, precision,
accuracy, F1, balanced accuracy, Wilson 95% intervals, decision coverage, timing
summaries for available classifier timelines, and mean bag/model runtimes.
Recall/FPR and confusion counts use parsed binary decisions; uncertain,
unavailable/error cases are counted separately and reflected in coverage.
Accuracy includes all known labels, with unavailable decisions counted incorrect;
`accuracy_decided` and `failure_recall_all` make the denominators explicit.
Undefined rates are N/A. `benchmark_metrics_notes.txt` records these definitions.
The existing ROC/score reports and signed classifier-timing graphs remain unchanged;
benchmark timing annotations ignore failure markers before the corrected stage-1
start, using the already computed classifier timeline without rescoring images.
Feature-only runs omit the generated-response metric series, since no response
was generated. The metric graph retains all six panels; unavailable rates are
marked N/A. This adds no inference or new
calibration. DINO's two-frame/stage-3 rule is specific
to the DINO benchmark and is not silently applied to generated VLM decisions.

Each VLM benchmark writes `REPORTS.md` in its decision-mode output folder. It
links all PNGs, lists each demo's timeline status/path, and explains unavailable
ROC and PCA reports. Timelines remain inside each demo's `rynnbrain_multiturn/`
folder. Prompt B results are under the separate `rynnbrain_prompt_b_benchmark/`
root; they are not written into A's results directory.

ROC/AUROC PNGs are retained even for single-class runs, as clearly labelled
unavailable panels with the reason. No ROC curve or AUROC value is invented.
Generated-response ROC is unavailable when `feature_only` skips text generation;
classifier ROC is a separate report. With normal and failure test labels, the
existing valid ROC/AUROC plots are generated normally. PCA now also plots a
single ground-truth class when there are at least two compatible, nonzero vectors
with at least two features. This shows within-class variation, not class separation.
If PCA or score plots cannot be calculated, a labelled unavailable PNG explains why.

For each remaining bag it writes a self-contained folder (videos, DINO CSVs,
selected VLM frames, prompts/responses) under
`<rynnbrain.output_dir>_benchmark/<decision_mode>/<bag_name>/`, and appends one row per
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

The probability report also writes `benchmark_threshold_sweep/`, comparing
**0% through 100%, inclusive, in 1% steps** using those already computed scores:

- `threshold_metrics.csv`: accuracy, balanced accuracy, failure recall,
  false-positive rate, precision, F1 and TP/FP/TN/FN at every threshold.
- `threshold_metrics_<mode>_<decision_mode>.png`: curves for accuracy, balanced
  accuracy, recall and FPR; the overall best-accuracy thresholds are highlighted.
- `threshold_summary.json`: the highest observed accuracy and **all tied
  thresholds**, plus class counts and exclusions. Comparisons with only one
  ground-truth class do not produce a best threshold.

Input modes, whole-demo/timeline protocols and failure categories remain separate.
In timeline mode the sweep uses each bag's maximum saved prefix score, matching
the existing any-crossing decision rule. It reuses the saved observation sequence;
it does not simulate different adaptive replay trajectories or extend stopped
recordings. Missing/invalid scores and unknown labels are excluded and counted.
This report does not change the config, model weights or classifier files.
Accuracy depends on the normal/failure class mix; use validation bags to choose
a threshold, then evaluate that chosen threshold on separate test bags.

Both the ordinary benchmark and the report-regeneration command below produce
this sweep automatically, with no additional model inference or training.

For the **sequential GUI-style search followed by one best-threshold report**, add
`"threshold_search": true` inside the existing `vlm_benchmark` object, alongside
`"decision_mode": "timeline"` and your selected `folders`. Keep the existing
LoRA/classifier paths and set `rynnbrain.inference.feature_only: true`. This workflow
currently supports a raw, single-camera rosbag VLM with a logistic classifier.
It does not run DINO or whole-demo inference. Use the ordinary `benchmark` command.

For long runs, disable the interactive pseudo-terminal with `-T`:

```bash
docker compose run --rm -T --build benchmark
```

`benchmark_run.log` is flushed to disk on each write. Terminal mirroring runs
separately so a stalled or disconnected terminal cannot block benchmark progress;
if the terminal stops accepting output, only terminal copies may be dropped.
The file log remains complete. During bag selection, `benchmark_runtime.json`
records the current bag and `selection_completed_bags` / `selection_total_bags`;
after selection it advances to model loading and then threshold scoring.

The benchmark uses the same `OnlineDetector` and the `rynnbrain.online_replay`
constant/adaptive interval rules as the GUI, but reads the first recorded frame
at or after each due bag timestamp without waiting for wall-clock replay. Only
the current frame and cached preceding observations reach the model. For an
extended trajectory, keep `online_replay.max_observations: null`; `num_frames: 8`
still describes the nominal reference. `online_replay.interval_sec` sets the
constant interval (default 10 seconds); the GUI also uses it as its initial value.
Adaptive bands remain 10/7/5/3 seconds.

The first pass deliberately continues after alarms so that all thresholds from
0% through 100% can be compared fairly. It caches each observation **once**, not
once per threshold. `threshold_selection/` contains compact per-bag NPZ scan
caches, selection scores, ROC/AUROC, threshold curves and `selected_threshold.json`.
Both nominal and failure labels are required. Sequential threshold selection now
minimizes a balanced first-alarm timing cost. The accuracy-only optimum and all
its ties remain available for comparison; the probability ROC is unchanged.

For a normal bag, any alarm costs 1 and no alarm costs 0. For a failure bag with
recorded onset `t0` and last scored time `T`, the first alarm at `a >= t0` costs
`(a - t0) / (T - t0)`. A missed failure or first alarm before `t0` costs 1;
a later correct crossing cannot rescue a premature stop. Selection minimizes
`0.5 * mean(normal costs) + 0.5 * mean(failure costs)` over 0–100%, in 1% steps.
Costs tied within absolute tolerance `1e-12` choose the lowest threshold.
There is no added configuration parameter or delay deadline. Normal and failure
classes each have half the weight, so class imbalance alone cannot favor one.

This is a transparent application-specific starting cost, inspired by delay-
sensitive evaluation in [NAB](https://arxiv.org/abs/1510.03336) and
[quickest change detection](https://arxiv.org/abs/1210.5552); it is **not** the
official NAB score or a universally optimal economic cost. Normalizing by the
remaining observed horizon makes the same delay less costly in a longer bag.
The cost of a detector that never alarms is 0.5. The report includes improvement
over that baseline; a minimum near 0.5 is weak evidence for a useful detector.
Annotations unavailable or outside the scored window, and incomplete/invalid
histories, are explicitly excluded and counted. If no evaluable normal/failure
comparison remains, selection stops with a saved explanation. No failure time
is invented and annotations never enter inference.

`benchmark_temporal_cost/` contains the cost/false-alarm/delay plots, CSV metrics
for every threshold, and every bag's first alarm, status, raw delay and cost.
Delay in this objective uses observation timestamps; it excludes processing,
transport and display latency. The separately measured frame-processing report
is retained. Mean/median/90th-percentile raw delays describe post-onset detections
only; missed and premature detections remain full penalties in the cost.

Only **one** full output tree is exported, under `best_threshold/`, using saved
frames/features rather than repeating model inference. With
`online_replay.stop_on_alarm: true`, each exported timeline ends at its first
crossing of the selected threshold; negative runs retain the observation window.
Usual metrics, timelines, failure-timing graphs, boxplots, ROC/AUROC, PCA, selected
input images, response/vector records and report index are retained. ROC ranking
uses `threshold_selection_peak_score` from the uncensored scan, as does the
comparison boxplot. Per-bag reported probabilities and PCA describe the selected
stopping point. There are no
per-threshold output trees. The original config, prompts and trained model files
are not edited; `best_threshold/pipeline_config.json` is a separate reusable
configuration with the chosen cutoff and training/search disabled. Use a fresh
output directory for another selection run; existing best-threshold reports are
not overwritten.

`benchmark_frame_runtime.csv`, `.json` and `.png` report mean cached feature and
classifier processing time per observation and a weighted mean over the selected
observations. These timings exclude model/reference loading, frame decoding,
waiting and report export. They are actual processing measurements from the
single scan, not nominal sampling intervals. Threshold selection on these bags
is validation, not an independent test of the chosen cutoff.

To refresh a completed sequential search using its saved observations, without
loading the VLM or repeating inference:

```bash
docker compose run --rm -T --build --entrypoint python benchmark \
  -m rynnbrain_vlm.benchmark_replay \
  --input-dir /outputs/experiments/YOUR_EXPERIMENT/rynnbrain_benchmark/timeline
```

The refresh writes `threshold_selection_timing/` and `best_threshold_timing/`,
preserving the original best-threshold reports, previous `*_0_100` reports,
and NPZ observation caches. No additional model run or retraining is needed.
The new best-threshold tree includes the complete usual report set and the
originally measured processing times. Its timelines are re-exported at the new
first-alarm stopping point from cached images/features. Refreshing requires the
saved scan NPZ files and original best-threshold provenance/configuration.
Existing refreshed reports are not overwritten. Recorded failure annotations
are read again, but model weights and raw execution images are not loaded.

Probability plots identify the actual source folder for both successful and
failed executions, including each nominal setting and nonstandard failure
folders. `benchmark_failure_probability_points.csv` maps every plotted score to
its bag, category, marker and colour. The two boxplots still group actual
normal/failure outcomes. Scores are failure scores even for successful bags;
timeline peaks are not calibrated execution-level probabilities.

An N/A folder accuracy with `selected_bags=0` means no cases from that folder
were evaluated. `benchmark_bag_discovery.csv` and
`benchmark_folder_selection.csv` distinguish missing bag metadata, training/
reference exclusions, and explicit bag filters or limits. Known-label counts
and availability explanations remain in `benchmark_metrics.csv`.

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
  --summary /outputs/experiments/example/rynnbrain_benchmark/whole_demo/benchmark_summary.csv
```

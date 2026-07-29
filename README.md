# DINOv2 vision anomaly detection

This project runs the vision portion of the anomaly-detection pipeline from
`vision_dinov2/run_vision_test.py`.

## Required directory layout

The program derives all paths from the location of `run_vision_test.py`.
It does not depend on the current Windows username, drive letter, clone
location, or terminal working directory.

Place the source code and data in this layout:

```text
AD/
|-- data/
|   `-- Exp_1/
|       `-- Exp_1/
|           `-- Exp_1_Videos_u_Fotos/
|               |-- i6_pick_place.MP4
|               |-- i7_pick_place.MP4
|               `-- i11_pick_place.MP4
`-- Script_VS/
    |-- .venv/
    |-- outputs/
    `-- vision_dinov2/
        |-- run_vision_test.py
        |-- dinov2_features.py
        |-- anomaly_detection.py
        |-- threshold.py
        `-- visualization.py
```

The nominal and test filenames are configured near the top of
`vision_dinov2/run_vision_test.py`. To use other videos, change only the
filenames in `NOMINAL_VIDEO_PATHS` and `TEST_VIDEO_PATH`. Change
`VIDEO_DIRECTORY` only if the data uses a different directory structure.

## Run on Windows

From the `Script_VS` directory, run:

```powershell
.\.venv\Scripts\python.exe .\vision_dinov2\run_vision_test.py
```

Activating the virtual environment is optional because this command invokes
its Python interpreter directly.

Generated CSV files and the annotated MP4 are saved in `Script_VS/outputs`,
regardless of the directory from which the command is executed.


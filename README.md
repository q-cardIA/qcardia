# qcardia

Python library for cardiac MRI segmentation inference. Provides series classes
(`CineSeries`, `LGESeries`), an inference pipeline, and visualisation utilities
for short-axis (SAX) and long-axis (LA) cine and LGE data.

---

## Installation

### 1. Install this package

```bash
pip install -e .
```

### 2. Install dependencies

`qcardia` depends on two companion packages:

| Package | GitHub | Notes |
|---|---|---|
| `qcardia-data` | <https://github.com/q-cardIA/qcardia-data> | Data utilities |
| `qcardia-models` | <https://github.com/q-cardIA/qcardia-models> | Model architectures |

Install them from GitHub:

```bash
pip install git+https://github.com/q-cardIA/qcardia-data.git
pip install git+https://github.com/q-cardIA/qcardia-models.git
```

Or clone and install as editable installs if you intend to modify them:

```bash
git clone https://github.com/q-cardIA/qcardia-data.git
pip install -e qcardia-data/

git clone https://github.com/q-cardIA/qcardia-models.git
pip install -e qcardia-models/
```

### 3. Context-aware transformer model

The context-aware `UNet_Transformer` architecture is **not included** in the
published `qcardia-models` package. To use it you need the development version
of the package, which must be obtained separately.

```bash
# install the dev version instead of (or replacing) the published one
pip install -e /path/to/qcardia-models-dev/
```

Without this, `qcardia` will still run but will only support `UNet2d` models.
Attempting to load a Transformer model will raise an `ImportError`.

---

## Model weights

Trained weights are not bundled with the package. Point `--model` at a
directory containing the config and weights. Two layouts are supported:

```
# Flat layout (e.g. weights/LAx/):
<model-dir>/
    config-copy.yaml
    best_model.pt       (or last_model.pt)

# WandB run layout:
<model-dir>/
    files/
        config-copy.yaml
        best_model.pt   (or last_model.pt)
```

`best_model.pt` is preferred over `last_model.pt` when both are present.

---

## Data directory structure

The inference scripts expect data sorted into per-view subfolders. This
structure is produced automatically by
[cardisort](https://github.com/q-cardIA/cardisort), but you can also organise
files manually.

```
<subject-dir>/
    CINE_SAX/            # Short-axis cine stack
        sa stack/        # DICOM files (subfolder name may vary)
    CINE_2CH/            # Two-chamber long-axis view
        <subfolder>/
    CINE_3CH/            # Three-chamber long-axis view
        <subfolder>/
    CINE_4CH/            # Four-chamber long-axis view
        <subfolder>/
```

**If your folder names differ** from `CINE_SAX`, `CINE_2CH`, etc., edit the
`LA_CHAMBERS` and `SAX_CHAMBERS` lists at the top of `run_multiview.py`.

---

## Running inference

### Multi-view pipeline (`run_multiview.py`)

The main entry point for running a model across cardiac views.

```bash
# Run on all available CINE_* views found in the subject folder:
python run_multiview.py \
    --model    /path/to/wandb/run \
    --data-dir /path/to/CardiSorted_QLGE01_

# Run on specific views only:
python run_multiview.py \
    --model    /path/to/wandb/run \
    --data-dir /path/to/CardiSorted_QLGE01_ \
    --views CINE_SAX CINE_2CH

# Custom output location and label for the model:
python run_multiview.py \
    --model    /path/to/wandb/run \
    --data-dir /path/to/CardiSorted_QLGE01_ \
    --name MyModel \
    --output-dir /path/to/results
```

`--views` defaults to every `CINE_*` subfolder found in `--data-dir`. Pass it
explicitly if you only want to process a subset. `--name` sets the label used in
output folder naming and defaults to the model directory name.

### Output directory layout

Results are written under `<data-dir>_results/` by default (or the directory
you specify with `--output-dir`):

```
<data-dir>_results/
    UNet-LA/
        CINE_2CH/
            CINE_2CH_data_check.png
            segmentation_frame<N>.png
            segmentation_animation.gif
            volume_curves.png
            marker_points_frame<N>.png
        CINE_2CH_segmentation/
            segmentation_1.nii  ...   # one file per slice
        CINE_3CH/  ...
        CINE_4CH/  ...
    UNet-SAX/
        CINE_SAX/
            CINE_SAX_data_check.png
            segmentation_frame<N>.png
            segmentation_animation_slice00.gif  ...
            sax_volume_curves.png
            sax_volume_heatmaps.png
            sax_3d_animation.gif
        CINE_SAX_segmentation/
            segmentation_1.nii  ...
    Transformer-Context/
        CINE_SAX/  ...
```

### Simple script (`run.py`)

`run.py` is an early development / exploratory script showing the low-level API
for `CineSeries` and `LGESeries`. It is not intended to be run as-is; use it
as a reference for using the library programmatically.

---

## Library API (quick reference)

```python
from pathlib import Path
from qcardia.series import CineSeries, LGESeries

# Load cine SAX data and run segmentation
cine = CineSeries(Path("/path/to/DICOM/folder"), batch_size=50)
seg = cine.predict_segmentation(Path("/path/to/wandb/run"))
cine.save_predictions(Path("/path/to/output"))

lv_vol = cine.compute_volume_curve(structure="lv")
ef = cine.compute_ejection_fraction(lv_vol)

# Load LGE data
lge = LGESeries(Path("/path/to/LGE/DICOM"))
lge_seg = lge.predict_segmentation(Path("/path/to/wandb/run"))
lge.save_predictions(Path("/path/to/output"))
```

# qcardia

Code for an AI-based quantitative cardiac image analysis package. Provides
series classes (`CineSeries`, `LGESeries`), cardisort-based sequence
classification/disambiguation, and inference for the plain, context-aware and
LA-conditioned segmentation models.

---

## Installation

```bash
pip install -e .
```

### A note on companion packages

`qcardia` depends on `qcardia-models` (model architectures) and `qcardia-data`
(data utilities). The context-aware model (`ContextUNet2d`, routed via
`is_transformer_model`) and the LA/SAx intersection geometry LA conditioning
needs (`qcardia_data.pipeline.la_sa_intersection`) have not landed on either
package's `main` branch yet — `pyproject.toml` currently pins them to the
branches that have this code (`qcardia-models@context-unet`,
`qcardia-data@context-la-sampling`). Once those branches merge upstream,
update the pins back to bare `main`.

The LA-conditioning geometry import in `qcardia.inference.la_conditioning` is
lazy, so plain `UNet2d` and context-aware-without-LA inference don't need that
particular branch — but `qcardia_models.training_utils.is_transformer_model`
is required unconditionally by model routing, so the `context-unet` pin (or
its eventual merge to main) is needed for any inference at all.

---

## Model weights

Trained weights are not bundled with the package. Point a model path at a
directory containing the config and weights; both a flat layout and a WandB
`files/` layout are supported:

```
<model-dir>/
    config-copy.yaml   (or config.yaml)
    last_model.pt      (or best_model.pt)

# or, WandB run layout:
<model-dir>/
    files/
        config-copy.yaml   (or config.yaml)
        last_model.pt      (or best_model.pt)
```

`last_model.pt` is preferred over `best_model.pt` when both are present, and a
direct `<model-dir>/...` match is preferred over one under `files/`.

---

## Data directory structure

Cardisort produces subject folders with one subdirectory per view, each
holding the DICOM series for that view:

```
<subject-dir>/
    CINE_SAX/            # Short-axis cine stack
        sa stack/        # DICOM files (subfolder name varies)
    CINE_2CH/            # Two-chamber long-axis view
    CINE_3CH/            # Three-chamber long-axis view
    CINE_4CH/            # Four-chamber long-axis view
```

`qcardia.utils.get_data_directory` resolves the actual DICOM folder inside a
view directory (some exports keep DICOMs directly inside it, others one level
down). If your own data doesn't follow the `CINE_*` naming, pass
`--as-chamber` to `run_inference.py` to say explicitly whether a folder is
short- or long-axis.

---

## Running inference

### `run_inference.py` — segment one study directory

The CLI entry point for running a trained model over a study laid out as
above. It's independent of cardisort: it only needs the view folders to
exist, so it also works on data that was organised by hand.

```bash
# Segment every CINE_* view found under a study directory:
python run_inference.py --model wandb/my-run --data-dir study

# Segment specific views only:
python run_inference.py --model wandb/my-run --data-dir study --views CINE_SAX

# LA-conditioned models additionally need the long-axis view they were
# conditioned on:
python run_inference.py --model wandb/wla-run --data-dir study \
    --views CINE_SAX --lax-dir study/CINE_4CH

# Measure how much the LA vectors change the result:
python run_inference.py --model wandb/wla-run --data-dir study \
    --views CINE_SAX --lax-dir study/CINE_4CH --la-ablation
```

Output is written to `<output-dir>/<model-name>/`, one subfolder per
segmented view:

```
results/my-run/
    CINE_SAX_segmentation/
        segmentation_1.nii   ...   # one file per slice
        spacing.json               # pixel spacing / slice count, for volumes
    CINE_2CH_segmentation/
        ...
```

`run_inference.py` exits non-zero if any requested view failed to segment, so
it can be used as a batch step in a larger pipeline (e.g. warming a cache of
segmentations across `data/new_data/`).

### `run.py` — full cardisort pipeline, for reference

`run.py` runs cardisort classification, LLM disambiguation and segmentation
end-to-end over a hardcoded dataset directory (`data copy/`) and hardcoded
WandB run paths. It's an exploratory/demo script for the full pipeline, not a
general-purpose CLI — there's no argument parsing and the paths are meant to
be edited in place. Use `run_inference.py` (or the library API below) when you
want to run inference against an arbitrary study directory or trained model;
use `run.py` as a reference for wiring cardisort + disambiguation +
segmentation together end-to-end.

### Library API (quick reference)

```python
from pathlib import Path
from qcardia.series import CineSeries, LGESeries

# Plain segmentation
cine = CineSeries(Path("/path/to/DICOM/folder"), batch_size=50)
seg = cine.predict_segmentation(Path("/path/to/wandb/run"))
cine.save_predictions(Path("/path/to/output"))

# LA-conditioned segmentation: pass the 4CH DICOM directory the model
# expects. cine.la_vectors holds the computed conditioning vectors afterward.
seg = cine.predict_segmentation(
    Path("/path/to/wandb/wla-run"),
    lax_dicom_dir=Path("/path/to/CINE_4CH"),
)

lv_vol = cine.compute_volume_curve(structure="lv")
ef = cine.compute_ejection_fraction(lv_vol)

# Load LGE data
lge = LGESeries(Path("/path/to/LGE/DICOM"))
lge_seg = lge.predict_segmentation(Path("/path/to/wandb/run"))
lge.save_predictions(Path("/path/to/output"))
```

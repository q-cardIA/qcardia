# LA Inference

Multi-model inference pipeline for long-axis cardiac cine segmentation.

## Overview

Compares segmentation predictions from multiple UNet models on 2-chamber, 3-chamber, and 4-chamber cine MRI sequences.

## Usage

```bash
conda activate qcardia-dev
cd qcardia/infer_LA
python run_LA.py
```

## Configuration

Edit model paths in `run_LA.py`:

```python
MODELS = {
    "UNet-LA": Path("path/to/UNet-LA/model"),
    "UNet-SA": Path("path/to/UNet-SA/model")
}
DATA_PATH = Path("path/to/CardiSorted_QLGE04_")
```

## Outputs

For each chamber and model combination:

### Visualizations
- `segmentation_frame15.png` - Mid-frame static visualization (3-panel)
- `segmentation_animation.gif` - Animated segmentation across cardiac cycle
- `volume_curves.png` - LV/MYO/RV volume curves with ejection fractions
- `marker_points.png` - Anatomical landmarks (LV/RV centers, RV insertions)
- `CINE_*CH_data_check.png` - Input data verification (first model only)

### Segmentation Masks
- `*CH_segmentation_{model}/` - NIfTI format segmentation files

Saved to: `DATA_PATH/CINE_*CH/*CH_results_{model}/`

## Segmentation Classes

- **0**: Background
- **1**: LV cavity
- **2**: Myocardium  
- **3**: RV / other structures

## Notes

- Volume calculations use area × slice thickness (single-slice LA views)
- For accurate LA volume measurements, consider Simpson's biplane or area-length methods
- Uses `CineSeries` class from qcardia.series (compatible with both SA and LA sequences)

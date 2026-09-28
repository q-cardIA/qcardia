# qcardia-app

This repo contains code for an interactive DICOM viewer.

## Features

- Multi-planar slice viewing (axial, coronal, sagittal)
- Dynamic sagittal segmentation contour overlay (classes 1/2/3) regenerated when slice changes
- Crosshair synchronization across views

## Development Notes

- Segmentation volume is stored and contours are extracted (morphological gradient) in `CentralDisplayAreaWidget.__update_segmentation_overlay`.
- Overlays are composited in `CustomQGraphicsView.update_slice_view` and flipped horizontally to match base slice orientation.
- Atlas code has been removed; remaining unused methods were cleaned.

## Dependencies

See `pyproject.toml` for Python dependencies. SciPy is required for morphological operations.

## Subject data

Each subfolder of the data folder (default `data/`, next to this repo) is one subject. Change it on the welcome screen under `···` → **Edit**.
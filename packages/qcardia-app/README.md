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

## LLM assistant

The chat assistant needs [Ollama](https://ollama.com/download) on this machine. Install it, then run `ollama pull qwen3:8b` to fetch the model the app defaults to: Ollama does not download models on demand, so without that step the assistant appears but every reply fails. Any other model you have pulled can be picked from the dropdown in the chat panel.

To serve the models from another machine instead, copy `config/llm_devices.example.json` to `config/llm_devices.json` and move one of its `examples` entries into `devices`. That file is gitignored, so host addresses and SSH aliases stay out of the repo, and a plain copy changes nothing: only `devices` is read, and "This computer" is always offered. You then pick the machine on the welcome screen before starting, and can switch from the chat panel mid-session if it drops.

The example documents every field. The two worth knowing up front are `mode`, where `tunnel` runs `ssh -N -L` against an alias from your `~/.ssh/config` so the app stores no credentials and the remote Ollama can stay bound to localhost, while `http` needs that server exposed on the network; and `patient_data_ok`, which defaults to false and keeps DICOM header text on your local Ollama even while chat runs elsewhere. Set `QCARDIA_LLM_DEVICES` to keep the config file somewhere else.

## Dependencies

See `pyproject.toml` for Python dependencies. SciPy is required for morphological operations.

## Subject data

Each subfolder of the data folder (default `data/`, next to this repo) is one subject. Change it on the welcome screen under `···` → **Edit**.
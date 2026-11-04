# qcardia-app

Desktop viewer for cardiac MR studies, built on [`qcardia`](../qcardia). It sorts a study with CardiSort, opens the series in hanging protocols (function, LGE, perfusion, flow), and shows segmentation contours and results such as volumes and ejection fraction.

## Run

From the repository root:

```bash
uv sync
uv run qcardia-app
```

## Subject data

Each subfolder of the data folder (default `data/`, next to the repository) is one subject. Change it on the start screen under the options button, **Edit**.

Instead of a folder, the start screen can open a **patient list**: a `.csv` with a made-up name and hospital ID per subject folder, which the app shows and searches instead of the folder name. Only the listed patients are offered, so a study session can have its own list. The paths in it are relative to the list, so a zip of the list and its subject folders still works. Its columns are `hospital_id,name,folder`. A data folder's own `patients.csv` also names its subfolders; without one, the app uses the folder names.

## Model weights and segmentations

The app reads model weights from `wandb/` in the folder it is started from (see [Model weights](../../README.md#model-weights)). Segmentations are cached per subject in `.segmentation_cache/`. Without the weights, the app still shows a cached segmentation of the same series, so a dataset with segmentations can be viewed without the models.

## Plugins

Other packages can extend the app through entry points in the group `qcardia_app.plugins`. A plugin derives from `qcardia_app.plugins.Plugin` and can add options to the start screen, a side panel (`MainWindow.add_panel`), key bindings (`bindings.register_action`), styles (`theme.add_stylesheet`) and icons (`theme.register_line_icon`). See `src/qcardia_app/plugins.py` and `tests/ui/test_plugins.py` for an example.

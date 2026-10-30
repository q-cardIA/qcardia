# qcardia

Deep learning tools for quantitative cardiac MR image analysis, developed by the
[q-cardIA](https://github.com/q-cardIA) group at Eindhoven University of Technology.

This repository holds the `qcardia` package family. Each package installs on its own,
and the packages share one issue tracker, one test suite, and one release process.

<!-- TODO: add badges once CI, docs and PyPI are live
[![CI](https://github.com/q-cardIA/qcardia/actions/workflows/ci.yml/badge.svg)](...)
[![Docs](https://readthedocs.org/projects/qcardia/badge/)](...)
-->

## Packages

| Package | Purpose |
|---|---|
| [`qcardia-data`](packages/qcardia-data) | Data pipelines. PyTorch and MONAI based dataloaders, augmentation, caching, and reformatting for supported public datasets. Used to train models. |
| [`qcardia-models`](packages/qcardia-models) | Model pipelines. U-Net and encoder-MLP architectures, losses, and metrics. Used to train models. |
| [`qcardia`](packages/qcardia) | Analysis and deployment. Sorts and disambiguates DICOM studies with CardiSort, builds series objects, and applies trained models. |
| [`qcardia-app`](packages/qcardia-app) | Desktop viewer for cardiac MR studies: hanging protocols, cine, LGE, perfusion and flow, segmentation contours and results. Uses `qcardia`. |

## How the packages fit together

```
qcardia-data ──┐
               ├──→ training (in study repos) ──→ trained weights
qcardia-models ┘                                        │
                                                        ↓
                                                    qcardia
                                           (sorting, inference, analysis)
                                                        ↓
                                                  qcardia-app
                                              (viewing, interaction)
```

`qcardia-data` and `qcardia-models` are training infrastructure. Study repositories
such as [ssp-cmr-cine-segmentation](https://github.com/q-cardIA/ssp-cmr-cine-segmentation)
use them to produce trained weights. `qcardia` consumes those weights and applies them
to clinical DICOM studies. The viewer presents the results.

Each layer depends only on the layer below it. The libraries stay free of any
user-interface dependency.

## Requirements

- Python 3.10 or later. Tested most thoroughly on 3.11.9.
- PyTorch 2.3.1 is the tested version. Install it first and confirm GPU availability.
  See the PyTorch [previous versions guide](https://pytorch.org/get-started/previous-versions/#v231).

## Installation

Install a single package from a release tag:

```bash
pip install "git+https://github.com/q-cardIA/qcardia@v1.0.0#subdirectory=packages/qcardia-data"
```

Replace the subdirectory to install a different package. Available tags are listed
in the [releases tab](https://github.com/q-cardIA/qcardia/releases).

Most users need only `qcardia`, which pulls the libraries it depends on:

```bash
pip install "git+https://github.com/q-cardIA/qcardia@v1.0.0#subdirectory=packages/qcardia"
```

### Development install

The repository is a [uv](https://docs.astral.sh/uv/) workspace. Clone it and sync:

```bash
git clone https://github.com/q-cardIA/qcardia.git
cd qcardia
uv sync
```

Every package installs in editable mode, and changes in one package take effect
immediately in the others.

## Model weights

`qcardia` applies trained models. It does not contain them.

Weights currently load from a local Weights & Biases run directory:

```python
from qcardia.cardisort import load_cardisort_model
model, config = load_cardisort_model(Path("wandb/cardisort"))
```

<!-- TODO: replace this section once weights are published.
     Planned: a Zenodo deposit with a DOI, plus qcardia.weights.fetch() with
     checksum verification and a local cache. Until then, contact the maintainers. -->

Published weights with a permanent download path are in preparation. Until they are
available, contact the maintainers for access.

## Documentation

Each package has its own README with installation and usage detail:

- [`qcardia-data` README](packages/qcardia-data/README.md) — data setup, supported datasets, configuration
- [`qcardia-models` README](packages/qcardia-models/README.md) — model configuration and training
- [`qcardia` README](packages/qcardia/README.md) — sorting and inference
- [`qcardia-app` README](packages/qcardia-app/README.md) — the viewer: running it, data, model weights and plugins

Worked examples are in `packages/qcardia-data/demo/` and `packages/qcardia-models/demo/`.

<!-- TODO: link the hosted documentation site once ReadTheDocs is configured. -->

## Development

```bash
uv sync                  # install all packages and dev dependencies
uv run pytest            # run the test suite
uv run ruff check .      # lint
uv run ruff format .     # format
uv run lint-imports      # check the package layering
```

Package boundaries are enforced by [import-linter](https://import-linter.readthedocs.io/).
A change that makes a library import the viewer, or that reverses the layering, fails
the build.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute, report a problem, or ask
for support. Please open an issue before starting substantial work, so we can agree the
approach first.

## Citation

Each package carries its own `CITATION.cff`. Cite the packages you used.

If you use the CardiSort sorting model, please also cite:

> Lim RP, Kachel S, Villa ADM, Kearney L, Bettencourt N, Young AA, Chiribiri A,
> Scannell CM. CardiSort: a convolutional neural network for cross vendor automated
> sorting of cardiac MR images. *arXiv preprint* arXiv:2109.08479, 2021.

## Related repositories

- [ssp-cmr-cine-segmentation](https://github.com/q-cardIA/ssp-cmr-cine-segmentation) — self-supervised pretraining for cine segmentation, built on `qcardia-data` and `qcardia-models`
- [cardisort](https://github.com/q-cardIA/cardisort) — the original CardiSort model and paper

## License

Apache License 2.0. See [LICENSE](LICENSE).

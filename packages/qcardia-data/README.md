## Quantitative cardiac image analysis data module: `qcardia-data`

A PyTorch and MONAI based library to build and handle medical imaging data pipelines. Can be used to quickly get highly customizable Dataloaders for deep learning purposes, especially for already supported datasets. Currently supported public datasets (i.e. reformatting is available):
| # | Name | Description | link |
|-|-|-|-|
| 1 | M&Ms   | Cardiac short axis cine MRI (4D) | https://www.ub.edu/mnms/
| 2 | M&Ms-2 | Cardiac short axis cine MRI (4D) | https://www.ub.edu/mnms-2/

*Instructions on how to setup these supported datasets can be found in the Getting started section*

### Installation
#### Environment and PyTorch
It is recommended to make a new environment (requires Python>=3.10 but only extensively tested for Python 3.11.9) and first installing PyTorch and checking GPU availability. It is recommended to install the PyTorch version the package was tested for ([PyTorch 2.3.1](https://pytorch.org/get-started/previous-versions/#v231)), which should limit warnings or unexpected behaviours. Alternatively, installation instructions for the latest stable PyTorch version can also be found in [their "get started" guide](https://pytorch.org/get-started/locally/).

#### Stable version of `qcardia-data`
Install from GitHub using pip to get the latest version:
```
pip install git+https://github.com/q-cardIA/qcardia-data
```

Or if you want a specific version, include the release version in the link, e.g.:
```
pip install git+https://github.com/q-cardIA/qcardia-data@v1.0.0
```

Available versions can be found in the [releases tab](https://github.com/q-cardIA/qcardia-data/releases), where the release [tags](https://github.com/q-cardIA/qcardia-data/tags) are used in the install command.

#### Editable version of `qcardia-data`
You can install a local copy in `editable` mode to make changes to the package that instantly get reflected in your environment. After getting a local copy (download/clone/fork), install using:
```
pip install -e "path/to/qcardia-data"
```
(It may be neccessary to first upgrade pip using `python3 -m pip install --upgrade pip`)

### Getting started

#### Data setup
Public datasets must first be reformatted so the qcardia-data pipeline can use the data. For the supported public datasets, this can be achieved by:
1. Downloading the public data.
2. Saving the data with the expected folder hierarchy.
3. Updating the configs to point to your local data folder.
4. Running the relevant data setup functions.

**Expected data folder hierarchy for reformatting of supported public datasets :**
```
data
└── original_data
    ├── MnM
    │   ├── dataset
    │   │   ├── A0S9V9
    │   │   │   ├── A0S9V9_sa_gt.nii.gz
    │   │   │   └── A0S9V9_sa.nii.gz
    │   │   └── ...
    │   └── dataset_information.csv
    ├── MnM2
    │   ├── dataset
    │   │   ├── 001
    │   │   │   ├── 001_SA_CINE.nii.gz
    │   │   │   ├── 001_SA_ED_gt.nii.gz
    │   │   │   ├── 001_SA_ED.nii.gz
    │   │   │   ├── 001_SA_ES_gt.nii.gz
    │   │   │   ├── 001_SA_ES.nii.gz
    │   │   │   └── ... (001_LA... -> long axis data unused for now)
    │   │   └── ...
    │   └── dataset_information.csv
    └── ...
```
**Setup functions:**
The data setup script can be found in `qcardia_data/setup/data_setup`, and require a path to your local data folder. This data_path should also be updated in any config files. Data setup functions reformat the relevant available original datasets, and can generate default test data splits.

Example cine data setup:
```python
from qcardia_data.setup import setup_cine
from pathlib import Path

data_path = Path("path/to/your/data_folder")
setup_cine(data_path)
```

#### Dataloaders
The qcardia-data package uses a configuration dictionary. We provide an example `demo/data-config.yaml` configuration file. The file's data path (`config.paths.data`) should be updated to your local data folder, and a valid subject split file name/path (`config.dataset.split_file`) is required. Dataloaders can then be initialized as shown below:

```python
from pathlib import Path
from qcardia_data import DataModule

import yaml


# get config dict from file
data_config_path = Path("data-config.yaml")
config = yaml.load(data_config_path.open(), Loader=yaml.FullLoader)

# setup data module with config
data = DataModule(config)
data.setup()

# get dataloaders for training and validation
train_dataloader = data.train_dataloader()
valid_dataloader = data.valid_dataloader()
```
*Note that this code snippet assumes a current working directory that contains the example config file*

You can take a look in the included demo folder for a notebook with more information and examples. To find the config files, the notebook assumes that the demo folder (where the notebook is located in) is the current working directory.

#### Custom dataset
Datasets that aren't supported first need their own reformatting to work with the `qcardia-data` pipeline. You can look at the reformatting scripts (`qcardia_data\setup\reformat\...`) of supported public datasets for inspiration, as well as the folder hierarchy of reformatted datasets below. Note that the reformatted dataset folder and file names determine how they can be selected in a `qcardia-data` config.

Full example data folder structure:
```
data
├── cached_data
│   ├── dev2D-mm1_285-mm2_85-sa_cine=sa_cine_gt-b5296fe704398a111570e2fafa44ae9a
│   │   ├── mm1-A0S9V9-00-00.pt
│   │   └── ...
│   ├── dev2D-mm1_285-mm2_85-sa_cine=sa_cine_gt-b5296fe704398a111570e2fafa44ae9a.csv
│   └── ...
├── original_data
│   ├── MnM
│   │   ├── dataset
│   │   │   ├── A0S9V9
│   │   │   │   ├── A0S9V9_sa_gt.nii.gz
│   │   │   │   └── A0S9V9_sa.nii.gz
│   │   │   └── ...
│   │   └── dataset_information.csv
│   ├── MnM2
│   │   ├── dataset
│   │   │   ├── 001
│   │   │   │   ├── 001_SA_CINE.nii.gz
│   │   │   │   ├── 001_SA_ED_gt.nii.gz
│   │   │   │   ├── 001_SA_ED.nii.gz
│   │   │   │   ├── 001_SA_ES_gt.nii.gz
│   │   │   │   ├── 001_SA_ES.nii.gz
│   │   │   │   └── ... (001_LA... -> long axis data unused for now)
│   │   │   └── ...
│   │   └── dataset_information.csv
│   └── ...
├── reformatted_data
│   ├── mm1
│   │   ├── A0S9V9
│   │   │   ├── A0S9V9_sa_cine_gt.nii.gz
│   │   │   └── A0S9V9_sa_cine.nii.gz
│   │   └── ...
│   ├── mm2
│   │   ├── 001
│   │   │   ├── 001_sa_cine_gt.nii.gz
│   │   │   └── 001_sa_cine.nii.gz
│   │   └── ...
│   ├── mm1.csv
│   ├── mm2.csv
│   └── ...
└── subject_splits
    ├── default-cine-split.yaml
    └── default-cine-test-split.yaml
```
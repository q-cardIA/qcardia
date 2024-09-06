## Quantitative cardiac image analysis data module: `qcardia-data`

A PyTorch and MONAI based library to build and handle medical imaging data pipelines. Can be used to quickly get highly customizable Dataloaders for deep learning purposes, especially for already supported datasets. Currently supported public datasets (i.e. reformatting is available):
| # | Name | Description | link |
|-|-|-|-|
| 1 | M&Ms   | Cardiac short axis cine MRI (4D) | https://www.ub.edu/mnms/
| 2 | M&Ms-2 | Cardiac short axis cine MRI (4D) | https://www.ub.edu/mnms-2/


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


### Getting started
The qcardia-data package uses a configuration dictionary. Dataloaders can then be initialized as shown below:

```python
from qcardia_data import DataModule


config = ...

# setup data module with config
data = DataModule(config)
data.setup()

# get dataloaders for training and validation
train_dataloader = data.train_dataloader()
valid_dataloader = data.valid_dataloader()
```

You can take a look in the included demo folder for a notebook with more information and examples, and some example configs.

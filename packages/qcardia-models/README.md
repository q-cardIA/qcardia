## Quantitative cardiac image analysis data module: `qcardia-models`

A PyTorch based library to build and handle medical imaging model pipelines. Can be used to quickly get highly customizable U-Net or EncoderMLP models for deep learning purposes. Currently supported models:
| # | Name | Dimensionality | Use case |
|-|-|-|-|
| 1 | U-Net   | 2D | Dense predictions (e.g. segmentation)
| 2 | EncoderMLP | 2D | Encoder pretraining tasks


### Installation
#### Environment and PyTorch
It is recommended to make a new environment (tested for Python 3.11.9) and first installing PyTorch and checking GPU availability. It is recommended to install the PyTorch version the package was tested for ([PyTorch 2.3.1](https://pytorch.org/get-started/previous-versions/#v231)), which should limit warnings or unexpected behaviours. Alternatively, installation instructions for the latest stable PyTorch version can also be found in [their "get started" guide](https://pytorch.org/get-started/locally/).

#### Stable version of `qcardia-models`
Install from GitHub using pip to get the latest version:
```
pip install git+https://github.com/q-cardIA/qcardia-models
```

Or if you want a specific version, include the release version in the link, e.g.:
```
pip install git+https://github.com/q-cardIA/qcardia-models@v1.0.0
```

Available versions can be found in the [releases tab](https://github.com/q-cardIA/qcardia-models/releases), where the release [tags](https://github.com/q-cardIA/qcardia-models/tags) are used in the install command.

#### Editable version of `qcardia-models`
You can install a local copy in `editable` mode to make changes to the package that instantly get reflected in your environment. After getting a local copy (download/clone/fork), install using:
```
pip install -e "path/to/qcardia-models"
```


### Getting started
While it is recommended to use a confgiuration file to initialize models, models can be initialized as shown below:

```python
from qcardia_models.models import EncoderMLP2d, UNet2d

# U-Net model
unet_model = UNet2d(
    nr_input_channels=1,
    channels_list=[32, 64, 128, 256, 512, 512, 512],
    nr_output_classes=4,
    nr_output_scales=1,  # number of decoder scales/blocks that give an output
)

# Encoder + MLP model
encodermlp_model = EncoderMLP2d(
    nr_input_channels=1,
    encoder_channels_list=[32, 64, 128, 256, 512, 512, 512],
    mlp_channels_list=[512, 512, 512],
)
```

You can take a look in the demo folder for the included demo notebooks for more information and examples. Tutorials for each component of `qcardia-models` in isolation can be found in the `demo/tutorial.ipynb` notebook. The `demo/model_visualization.ipynb` notebook visualizes two available models and their components (requires the `torchview` package, see notebook for instructions).

The `train_demo.py` training script showcases a bare-bones example of training a model. The training demo uses our [qcardia-data](https://github.com/q-cardIA/qcardia-data) package to get training data, but other data(loaders) can also be used. To find the config files, the training demo script assumes a main/root package folder level current working directory (i.e. parent folder of demo folder), while the demo notebooks assume the demo folder (where the notebooks are located) as the current working directory. Make sure to update your file/folder hierarchy and current working directories when running scripts/notebooks that cannot find config files, or change config paths in the code directly to reflect the correct (global) path of the config.

For more advanced training scripts, you can look at our [ssp-cmr-cine-segmentation](https://github.com/q-cardIA/ssp-cmr-cine-segmentation) repository, showcasing baseline training from scratch, as well as self-supervised pretraining.

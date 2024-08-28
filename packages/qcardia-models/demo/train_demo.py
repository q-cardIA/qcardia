import warnings
from pathlib import Path

import torch
import yaml
from qcardia_data import DataModule

from qcardia_models.losses import DiceCELoss
from qcardia_models.models import UNet2d
from qcardia_models.utils import seed_everything


def main():
    # The config contains all the model hyperparameters and training settings for the
    # experiment. Additionally, it contains data preprocessing and augmentation
    # settings, as well as paths to data.
    config_path = Path("demo/train-demo-config.yaml")
    config = yaml.load(Path.open(config_path), Loader=yaml.FullLoader)

    # seed for reproducibility
    seed_everything(config["experiment"]["seed"])

    # Initialises the qcardia-data DataModule with the configuration specified in the
    # config yaml file. The DataModule handles the data loading, preprocessing, and
    # augmentation. The setup method is used to cache uncached datasets and to build
    # resampling transforms including augmentations and intensity pre-processing.
    # It also builds the dataset and splits to be used by the dataloaders.
    data = DataModule(config)
    data.setup()

    # get the MONAI DataLoader objects for the training datasets
    train_dataloader = data.train_dataloader()

    # definition of training and model settings based on the information in config yaml
    max_epochs = config["training"]["max_nr_epochs"]
    image_key, label_key = config["dataset"]["key_pairs"][0]  # get dict key pairs

    # the device is set to GPU if available, otherwise CPU is used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        warnings.warn("No GPU available; using CPU", stacklevel=1)

    # get custom loss function and model from qcardia-models
    loss_function = DiceCELoss(
        cross_entropy_loss_weight=config["loss"]["cross_entropy_loss_weight"],
        dice_loss_weight=config["loss"]["dice_loss_weight"],
        dice_classes_weights=config["loss"]["dice_classes_weights"],
    )
    unet_model = UNet2d(
        nr_input_channels=config["unet"]["nr_image_channels"],
        channels_list=config["unet"]["channels_list"],
        nr_output_classes=config["unet"]["nr_output_classes"],
        nr_output_scales=config["unet"]["nr_output_scales"],
    ).to(device)

    # optimizer and learning rate scheduler from torch
    optimizer = torch.optim.SGD(
        unet_model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        momentum=config["optimizer"]["momentum"],
        nesterov=config["optimizer"]["nesterov"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    learning_rate_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, max_epochs, config["training"]["polynomial_scheduler_power"]
    )

    # training loop
    for epoch_nr in range(max_epochs):
        for x in train_dataloader:
            # training step
            optimizer.zero_grad()  # reset gradients
            outputs = unet_model(x[image_key].to(device))
            labels = x[label_key].to(device)

            loss, _, _ = loss_function(outputs[0], labels)  # calculate loss
            loss.backward()
            optimizer.step()

        learning_rate_scheduler.step()  # update learning rate

        # save model weights
        weights_dict = {k: v.cpu() for k, v in unet_model.state_dict().items()}
        torch.save(weights_dict, "last_model.pt")


if __name__ == "__main__":
    main()

from pathlib import Path

import yaml

from qcardia_data.pipeline import DataModule
from qcardia_data.utils import print_dict


def main():
    data_config_path = Path("data-config.yaml")
    config = yaml.load(data_config_path.open(), Loader=yaml.FullLoader)

    data = DataModule(config)
    data.setup()

    dataloader = data.train_dataloader()
    for x in dataloader:
        print_dict(x, max_len=128)
        break

    valid_dataloader = data.valid_dataloader()
    for x in valid_dataloader:
        print_dict(x, max_len=128)
        break


if __name__ == "__main__":
    main()

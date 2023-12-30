import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision
from dataset.model import CityScapeAnnotatedImage, CityScapeRegion, CityScapeLabel
from torch.utils.data import Dataset


def load_json_dataset(filepath):
    with open(filepath, 'r') as f:
        data = f.read()
    dataset = json.loads(data, object_hook=lambda d: SimpleNamespace(**d))
    return CityScapeAnnotatedImage(dataset.imgWidth, dataset.imgWidth,
                                   [CityScapeRegion(CityScapeLabel.parse(o.label), o.polygon)
                                    for o in dataset.objects])


def load_cityscape_dataset(dataset_root_folder, dataset_type, city_name):
    path_prefix_train = f'{dataset_root_folder}/leftImg8bit/train/{city_name}'
    filepaths_train = []
    for item in Path(path_prefix_train).iterdir():
        filepaths_train.append(f'{path_prefix_train}/{item.name}')
    path_prefix_target = f'{dataset_root_folder}/{dataset_type}/train/{city_name}'
    filepaths_target = []
    for item in Path(path_prefix_target).iterdir():
        if item.is_file() and item.name.endswith('labelIds.png'):
            filepaths_target.append(f'{path_prefix_target}/{item.name}')
    return CityScapeDataset(filepaths_train, filepaths_target)


class CityScapeDataset(Dataset):
    def __init__(self, filepaths_train, filepaths_target):
        super(CityScapeDataset, self).__init__()
        self.filepaths_train = filepaths_train
        self.filepaths_target = filepaths_target

    def __getitem__(self, idx):
        in_tensor = torch.stack([torchvision.io.read_image(filepath) for filepath in self.filepaths_train[idx]], dim=0)
        target = torch.stack([torchvision.io.read_image(filepath) for filepath in self.filepaths_target[idx]], dim=0)
        # Convert the target image into the 2D tensor of labels
        return in_tensor, target

    def __len__(self):
        return len(self.filepaths_train)

import os
from dataclasses import dataclass
from datetime import datetime

import torch
import torchvision
from torch import cuda
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


@dataclass(init=True)
class LearningParameters:
    lr = 5e-4
    weight_decay = 2e-4
    betas = (0.9, 0.99)
    eps = 1e-6


@dataclass(init=True)
class EnetConfig:
    pretrained_model_path: str
    train_full_model: bool
    max_epoch: int
    dataset_root_folder: str = 'datasets/cityscapes/data_unzipped'
    image_mode = 'fine'
    target_types = ['semantic']
    batch_size = 10
    custom_weight_scaling_const = 1.02
    image_size = (256, 512)
    scaling_props_range = (1.0, 50.0)
    learning_params = LearningParameters()

    def __post_init__(self):
        self.target_image_size = self.image_size if self.train_full_model else (
            self.image_size[0] // 8, self.image_size[1] // 8)
        self.pretrained_model_path = self.pretrained_model_path.strip() if self.pretrained_model_path is not None else None

    def load_dataset(self, split):
        resize_input = transforms.Resize(self.image_size)
        resize_target = transforms.Resize(self.target_image_size)
        input_transforms = transforms.Compose([resize_input, transforms.ToTensor()])
        target_transforms = transforms.Compose([resize_target, transforms.PILToTensor()])
        dataset = torchvision.datasets.Cityscapes(self.dataset_root_folder,
                                                  split=split, mode=self.image_mode,
                                                  target_type=self.target_types,
                                                  transform=input_transforms,
                                                  target_transform=target_transforms)
        num_workers = 2 if cuda.is_available() else os.cpu_count() // 2
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=num_workers, pin_memory=False)

    def load_checkpoint(self, model, optimizer, device):
        if self.pretrained_model_path is not None and self.pretrained_model_path != '':
            checkpoint = torch.load(self.pretrained_model_path, map_location=device)
            print(f'Checkpoint file {self.pretrained_model_path} successfully loaded')
            strict = True
            if self.train_full_model and "encoder" in self.pretrained_model_path:
                # Remove the last full_convolution layer
                # The checkpoint contains an encoder-only model
                del checkpoint['model_state_dict']['full_conv.weight']
                strict = False
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    def save_checkpoint(self, model, optimizer, loss, epoch, root_folder_path=''):
        print(f'Epoch={epoch + 1}/{self.max_epoch}, Loss={loss.item()}')
        model_name = 'enet_model' + ('' if self.train_full_model else '_encoder')
        filepath = f'{root_folder_path}pretrained_model/{model_name}_{epoch}.pt'
        print(f'Saving training checkpoint as {filepath} at {datetime.now()}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, filepath)

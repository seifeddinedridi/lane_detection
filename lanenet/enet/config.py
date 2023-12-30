from dataclasses import dataclass
from datetime import datetime

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


@dataclass
class EnetConfig:
    batch_size = 10
    max_epoch = 1000
    custom_weight_scaling_const = 1.02
    dataset_root_folder = 'datasets/cityscapes/data_unzipped'
    train_full_model = bool
    pretrained_model_path: str
    image_size = (256, 512)
    scaling_props_range = (1.0, 50.0)
    image_mode = 'fine'
    target_types = ['semantic']

    def __init__(self, pretrained_model_path, train_full_model):
        self.train_full_model = train_full_model
        self.target_image_size = self.image_size if self.train_full_model else (
            self.image_size[0] // 8, self.image_size[1] // 8)
        self.pretrained_model_path = pretrained_model_path.strip() if pretrained_model_path is not None else None
        self.checkpoint = None

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
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=0, pin_memory=True)

    def load_checkpoint(self, model, optimizer, device):
        if self.pretrained_model_path is not None and self.pretrained_model_path != '':
            self.checkpoint = torch.load(self.pretrained_model_path, map_location=device)
            print(f'Checkpoint file {self.pretrained_model_path} successfully loaded')
            strict = True
            if self.train_full_model and "encoder" in self.pretrained_model_path:
                # Remove the last full_convolution layer
                # The checkpoint contains an encoder-only model
                del self.checkpoint['model_state_dict']['full_conv.weight']
                strict = False
            model.load_state_dict(self.checkpoint['model_state_dict'], strict=strict)
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

    def save_training_checkpoint(self, model, optimizer, loss, epoch):
        print(f'Epoch={epoch + 1}/{self.max_epoch}, Loss={loss.item()}')
        model_name = 'enet_model' + '' if self.train_full_model else '_encoder'
        filepath = f'pretrained_model/{model_name}_{epoch}.pt'
        print(f'Saving training checkpoint as {filepath} at {datetime.now()}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, filepath)
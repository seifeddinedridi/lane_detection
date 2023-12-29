import itertools
from dataclasses import dataclass
from datetime import datetime
from time import time

import torch as torch
import torchvision
import tqdm as tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import Enet
from dataset.labels import labels


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

    def __init__(self, pretrained_model_path, train_full_model):
        self.train_full_model = train_full_model
        self.target_image_size = self.image_size if self.train_full_model else (
            self.image_size[0] // 8, self.image_size[1] // 8)
        self.pretrained_model_path = pretrained_model_path.strip()

    def load_model(self):
        return self.pretrained_model_path is not None and self.pretrained_model_path != ''


def load_dataset(dataset_root_folder, scale_input_size, scale_target_size, batch_size=4, mode='coarse', split='train'):
    resize_input = transforms.Resize(scale_input_size)
    resize_target = transforms.Resize(scale_target_size)
    input_transforms = transforms.Compose([resize_input, transforms.ToTensor()])
    target_transforms = transforms.Compose([resize_target, transforms.PILToTensor()])
    dataset = torchvision.datasets.Cityscapes(dataset_root_folder, split=split, mode=mode,
                                              target_type='semantic',
                                              transform=input_transforms,
                                              target_transform=target_transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)


def eval_model(model, test_dataset, custom_weight_scaling_const, scaling_props_range):
    model.train(False)
    torch.manual_seed(int(time()))
    dataset_iter = itertools.cycle(iter(test_dataset))
    average_loss = 0
    max_epoch = 10
    for epoch in range(max_epoch):
        in_tensor, target = next(dataset_iter)
        logits = model(in_tensor)
        loss = compute_loss(logits, target, custom_weight_scaling_const, scaling_props_range)
        average_loss += loss.item()
    average_loss /= max_epoch
    print(f'Evaluation Loss={average_loss}')
    model.train()
    return average_loss


def save_training_checkpoint(model, optimizer, loss, epoch, max_epoch):
    print(f'Epoch={epoch + 1}/{max_epoch}, Loss={loss.item()}')
    filepath = f'pretrained_model/enet_model_{epoch}.pt'
    print(f'Saving training checkpoint as {filepath} at {datetime.now()}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


def load_model(model, optimizer, device, checkpoint_filepath, train_full_model):
    checkpoint = torch.load(checkpoint_filepath, map_location=device)
    print(f'Checkpoint file {checkpoint_filepath} successfully loaded')
    if train_full_model is True:
        # Remove the last full_convolution layer
        del checkpoint['model_state_dict']['full_conv.weight']
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


@torch.no_grad()
def batched_bincount(x, dim, num_classes):
    target = torch.zeros((x.shape[0], num_classes), dtype=x.dtype, device=x.device)
    values = torch.ones_like(x, dtype=torch.int64)
    target.scatter_add_(dim, x, values)
    return target


def compute_loss(logits, target, custom_weight_scaling_const, scaling_props_range):
    target_flat = target.type(torch.int64).view(target.shape[0], -1)
    probabilities = batched_bincount(target_flat, 1, target_flat.shape[1]) / target_flat.sum(dim=1).unsqueeze(dim=1)
    weights = torch.clamp(torch.div(1.0, (torch.log(torch.add(custom_weight_scaling_const, probabilities)))),
                          scaling_props_range[0], scaling_props_range[1])
    loss = torch.nn.functional.cross_entropy(logits, target.squeeze().type(torch.int64), reduction='none').view(
        target.shape[0], -1)
    loss = (loss * weights / weights.sum()).sum()
    return loss


def main():
    config = EnetConfig('pretrained_model/enet_model_encoder_only.pt', True)
    out_channels = len(labels)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = Enet(config.image_size, out_channels, config.train_full_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-4, betas=(0.9, 0.99), eps=1e-6)
    if config.load_model():
        load_model(model, optimizer, device, config.pretrained_model_path, config.train_full_model)
    model.train()
    train_dataset = load_dataset(config.dataset_root_folder, config.image_size, config.target_image_size,
                                 config.batch_size)
    dataset_iter = itertools.cycle(iter(train_dataset))
    progress_bar = tqdm.trange(0, config.max_epoch)
    for epoch in progress_bar:
        in_tensor, target = next(dataset_iter)
        logits = model(in_tensor)
        loss = compute_loss(logits, target, config.custom_weight_scaling_const, config.scaling_props_range)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.set_description(f"Epoch [{epoch}/{config.max_epoch}]")
        progress_bar.set_postfix(loss=loss.item())


if __name__ == '__main__':
    main()

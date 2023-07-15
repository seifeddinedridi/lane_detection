import itertools
from datetime import datetime
from time import time

import torch as torch
import torchvision
import tqdm as tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import Enet


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


def eval_model(model, test_dataset, out_channels, custom_weight_scaling_const):
    model.train(False)
    torch.manual_seed(int(time()))
    dataset_iter = itertools.cycle(iter(test_dataset))
    average_loss = 0
    max_epoch = 10
    for epoch in range(max_epoch):
        in_tensor, target = next(dataset_iter)
        logits = model(in_tensor)
        target_flat = target.view(-1)
        probabilities = torch.div(torch.bincount(target_flat, minlength=out_channels), target_flat.shape[0])
        weights = torch.div(1.0, (torch.log(custom_weight_scaling_const + probabilities)))
        loss = torch.nn.functional.cross_entropy(logits.view(-1, out_channels), target_flat, weights)
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


def main():
    batch_size = 10
    max_epoch = 1000
    custom_weight_scaling_const = 1.02
    dataset_root_folder = 'datasets/cityscapes/data_unzipped'
    from dataset.labels import labels
    out_channels = len(labels)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    image_size = (256, 512)
    train_full_model = False
    target_image_size = image_size if train_full_model is True else (image_size[0] // 8, image_size[1] // 8)
    model = Enet(image_size, out_channels, train_full_model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-4, betas=(0.9, 0.99), eps=1e-6)
    train_dataset = load_dataset(dataset_root_folder, image_size, target_image_size, batch_size, 'coarse')
    dataset_iter = itertools.cycle(iter(train_dataset))
    progress_bar = tqdm.trange(0, max_epoch)
    for epoch in progress_bar:
        in_tensor, target = next(dataset_iter)
        logits = model(in_tensor)
        target_flat = target.view(-1)
        probabilities = torch.div(torch.bincount(target_flat, minlength=out_channels), target_flat.shape[0])
        weights = torch.clamp(torch.div(1.0, (torch.log(torch.add(custom_weight_scaling_const, probabilities)))), 1.0,
                              50.0)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, out_channels), target_flat, weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.set_description(f"Epoch [{epoch}/{max_epoch}]")
        progress_bar.set_postfix(loss=loss.item())


if __name__ == '__main__':
    main()

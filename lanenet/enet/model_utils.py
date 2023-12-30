import torch

from lanenet.enet.dataset.labels import labels
from lanenet.enet.model import Enet


def eval_model(model, dataset_iter, custom_weight_scaling_const, scaling_props_range):
    model.train(False)
    average_loss = 0
    max_epoch = 10
    for epoch in range(max_epoch):
        in_tensor, target = next(dataset_iter)
        logits = model(in_tensor)
        loss = compute_loss(logits, target, custom_weight_scaling_const, scaling_props_range)
        average_loss += loss.item()
    average_loss /= max_epoch
    print(f'\nEvaluation Loss={average_loss}')
    model.train()
    return average_loss


def compute_loss(logits, target, custom_weight_scaling_const, scaling_props_range):
    target_flat = target.type(torch.int64).view(target.shape[0], -1)
    probabilities = batched_bincount(target_flat, 1, target_flat.shape[1]) / target_flat.sum(dim=1).unsqueeze(dim=1)
    weights = torch.clamp(torch.div(1.0, (torch.log(torch.add(custom_weight_scaling_const, probabilities)))),
                          scaling_props_range[0], scaling_props_range[1])
    loss = (torch.nn.functional.cross_entropy(logits, target.squeeze().type(torch.int64), reduction='none')
            .view(target.shape[0], -1))
    loss = (loss * weights / weights.sum()).sum()
    return loss


@torch.no_grad()
def batched_bincount(x, dim, num_classes):
    target = torch.zeros((x.shape[0], num_classes), dtype=x.dtype, device=x.device)
    values = torch.ones_like(x, dtype=torch.int64)
    target.scatter_add_(dim, x, values)
    return target


def load_model(config, device):
    out_channels = len(labels)
    model = Enet(config.image_size, out_channels, config.train_full_model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-4, betas=(0.9, 0.99), eps=1e-6)
    config.load_checkpoint(model, optimizer, device)
    model.train()
    train_dataset = config.load_dataset('train')
    test_dataset = config.load_dataset('val')
    return model, optimizer, train_dataset, test_dataset

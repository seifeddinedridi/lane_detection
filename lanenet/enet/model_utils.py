import torch
from torchvision import transforms

from lanenet.enet.dataset.labels import labels, id2label
from lanenet.enet.model import Enet


def eval_model(model, dataset_iter, custom_weight_scaling_const, scaling_props_range, device, max_epoch = 10):
    model.train(False)
    average_loss = 0
    for epoch in range(max_epoch):
        in_tensor, target = next(dataset_iter)
        in_tensor = in_tensor.to(device)
        target = target.to(device)
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
    loss = torch.nn.functional.cross_entropy(logits, target.squeeze().type(torch.int64), reduction='none').view(
        target.shape[0], -1)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_params.lr,
                                  weight_decay=config.learning_params.weight_decay, betas=config.learning_params.betas,
                                  eps=config.learning_params.eps)
    epoch, loss = config.load_checkpoint(model, optimizer, device)
    model.train()
    train_dataset = config.load_dataset('train_extra')
    test_dataset = config.load_dataset('val')
    return model, optimizer, train_dataset, test_dataset, epoch, loss


def segment_image(model, in_tensor, device):
    model.train(False)
    in_tensor = in_tensor.to(device)
    logits = model(in_tensor)
    model.train()
    # Map the labels to an image
    image = tensor_to_image(logits[0].detach())
    image.show()
    transforms.ToPILImage()(in_tensor[0]).show()


def tensor_to_image(logits):
    # Input has shape: (F, H, W)
    out_tensor = torch.zeros(3, logits.shape[1], logits.shape[2]).type(torch.uint8)
    for h in range(0, logits.shape[1]):  # height
        for w in range(0, logits.shape[2]):  # width
            # pixel_class = torch.distributions.Categorical(logits=logits[:, h, w] / 0.05).sample((1,)).item()
            pixel_class = torch.argmax(logits[:, h, w], dim=0).item()
            color = id2label[pixel_class][7] if pixel_class == 7 else [0, 0, 0]
            out_tensor[:, h, w] = torch.tensor([color[0], color[1], color[2]]).type(torch.int32)
    return transforms.ToPILImage()(out_tensor)


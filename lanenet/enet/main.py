from time import time

import torch
from tqdm import trange

from lanenet.enet.config import EnetConfig
from lanenet.enet.model_utils import load_model, eval_model, compute_loss


def main():
    # pretrained_model_path = 'pretrained_model/enet_model_encoder_only.pt'
    # pretrained_model_path = 'pretrained_model/enet_model_1000.pt'
    pretrained_model_path = None
    config = EnetConfig(pretrained_model_path, False)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model, optimizer, train_dataset, test_dataset = load_model(config, device)
    train_dataset_iter = iter(train_dataset)
    test_dataset_iter = iter(test_dataset)

    progress_bar = trange(0, config.max_epoch)
    last_checkpoint_saving_time = time()
    saving_period = 2 * 60  # 2 minutes
    best_eval_loss = float('inf')

    for epoch in progress_bar:
        try:
            in_tensor, target = next(train_dataset_iter)
        except StopIteration:
            # Iterator is exhausted
            train_dataset_iter = iter(train_dataset)
            in_tensor, target = next(train_dataset_iter)
        logits = model(in_tensor)
        loss = compute_loss(logits, target, config.custom_weight_scaling_const, config.scaling_props_range)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.set_description(f"Epoch [{epoch}/{config.max_epoch}]")
        progress_bar.set_postfix(loss=loss.item())
        if time() - last_checkpoint_saving_time >= saving_period:
            last_checkpoint_saving_time = time()
            try:
                eval_loss = eval_model(model, test_dataset_iter, config.custom_weight_scaling_const,
                                       config.scaling_props_range)
            except StopIteration:
                # Iterator is exhausted
                test_dataset_iter = iter(test_dataset)
                eval_loss = eval_model(model, test_dataset_iter, config.custom_weight_scaling_const,
                                       config.scaling_props_range)
            if eval_loss < best_eval_loss:
                config.save_training_checkpoint(model, optimizer, loss, epoch)


if __name__ == '__main__':
    main()

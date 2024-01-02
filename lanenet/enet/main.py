from time import time

import torch
from tqdm import tqdm
from lanenet.enet.config import EnetConfig
from lanenet.enet.model_utils import load_model, eval_model, compute_loss


def main():
    pretrained_model_path = 'pretrained_model/enet_model_894.pt'
    # pretrained_model_path = None
    config = EnetConfig(pretrained_model_path=pretrained_model_path, train_full_model=True)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.set_start_method('spawn')
    torch.set_flush_denormal(True)
    model, optimizer, train_dataset, test_dataset = load_model(config, device)
    train_dataset_iter = iter(train_dataset)
    test_dataset_iter = iter(test_dataset)

    last_checkpoint_saving_time = time()
    saving_period = 2 * 60  # 2 minutes
    best_eval_loss = float('inf')

    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ''{rate_noinv_fmt}{postfix}]'
    progress_bar = tqdm(range(0, config.max_epoch), bar_format=bar_format)
    for epoch in progress_bar:
        try:
            in_tensor, target = next(train_dataset_iter)
        except StopIteration:
            # Iterator is exhausted
            train_dataset_iter = iter(train_dataset)
            in_tensor, target = next(train_dataset_iter)
        in_tensor = in_tensor.to(device)
        target = target.to(device)
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
                                       config.scaling_props_range, device)
            except StopIteration:
                # Iterator is exhausted
                test_dataset_iter = iter(test_dataset)
                eval_loss = eval_model(model, test_dataset_iter, config.custom_weight_scaling_const,
                                       config.scaling_props_range, device)
            if eval_loss < best_eval_loss:
                config.save_checkpoint(model, optimizer, loss, epoch)
                best_eval_loss = eval_loss


if __name__ == '__main__':
    main()

import torch

from lanenet.enet.config import EnetConfig
from lanenet.enet.model_utils import load_model, segment_image


def test_model():
    pretrained_model_path = 'pretrained_model/enet_model_829.pt'  # 2116
    image_path = 'scripts/bielefeld_000000_000321_leftImg8bit.png'
    # pretrained_model_path = None
    config = EnetConfig(pretrained_model_path=pretrained_model_path, train_full_model=True, max_epoch=1)
    device = config.device
    torch.multiprocessing.set_start_method('spawn')
    torch.set_flush_denormal(True)
    model, optimizer, _, _, _, _ = load_model(config, device)
    in_tensor = config.load_image_as_tensor(image_path)
    in_tensor = in_tensor.to(device)
    segment_image(model, in_tensor, device)


if __name__ == '__main__':
    test_model()

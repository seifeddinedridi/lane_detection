from torch import nn

import copy
from lanenet.enet.model import Enet


class LanenetCommonModule(nn.Module):

    def __init__(self, enet: Enet):
        super().__init__()
        self.stage0 = enet.encoder.stage0
        self.stage1 = enet.encoder.stage1
        self.stage2 = enet.encoder.stage2

    def forward(self, x):
        return self.stage2(self.stage1(self.stage0(x)))


class BinarySegmentation(LanenetCommonModule):
    def __init__(self, enet: Enet):
        super().__init__(enet)
        self.stage3 = copy.deepcopy(enet.encoder.stage3)
        self.decoder = copy.deepcopy(enet.decoder)

    def forward(self, x):
        pass


class InstanceSegmentation(LanenetCommonModule):
    def __init__(self, enet: Enet, embedding_dimension=4, delta_v=0.5, delta_d=3.0):
        super().__init__(enet)
        self.stage3 = copy.deepcopy(enet.encoder.stage3)
        self.decoder = copy.deepcopy(enet.decoder)

    def forward(self, x):
        pass


class HNetBlock2d(nn.Module):
    def __init__(self, in_channels, size: tuple[int, int], stride, max_pool_size: tuple[int, int] = None,
                 max_pool_stride=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, size, stride, bias=False)
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        if max_pool_size is not None:
            self.max_pool = nn.MaxUnpool2d(max_pool_size, max_pool_stride)

    def forward(self, x):
        out_1 = self.conv(x)
        out_2 = self.batch_norm(out_1)
        out_3 = self.relu(out_2)
        return out_3 if self.max_pool is None else self.max_pool(out_3)


class HNetBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_1 = self.linear(x)
        out_2 = self.batch_norm(out_1)
        out_3 = self.relu(out_2)
        return out_3


class HNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = HNetBlock2d(16, (3, 3), 1)
        self.block2 = HNetBlock2d(16, (3, 3), 1, (2, 2), 2)

        self.block3 = HNetBlock2d(32, (3, 3), 1)
        self.block4 = HNetBlock2d(32, (3, 3), 1, (2, 2), 2)

        self.block5 = HNetBlock2d(64, (3, 3), 1)
        self.block6 = HNetBlock2d(64, (3, 3), 1, (2, 2), 2)

        self.linear1 = HNetBlock1d(128, 1024)
        self.linear2 = nn.Linear(1024, 6)

    def forward(self, x):
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)

        block3_out = self.block3(block2_out)
        block4_out = self.block4(block3_out)

        block5_out = self.block5(block4_out)
        block6_out = self.block6(block5_out)

        linear1_out = self.linear1(block6_out.view(block6_out.shape[0], -1))
        linear2_out = self.linear2(linear1_out)
        return linear2_out


class Lanenet(nn.Module):
    def __init__(self, enet: Enet):
        super().__init__()
        self.instance_segmentation = InstanceSegmentation(enet)
        self.binary_segmentation = BinarySegmentation(enet)
        self.hnet = HNet()

    def forward(self, x):
        pass

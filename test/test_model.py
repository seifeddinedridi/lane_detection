import unittest

import torch as torch
from torch import nn

import lanenet.enet as enet
from lanenet.enet.model import InitialBlock


class BottleStagesTestCase(unittest.TestCase):
    def test_stages_output_shapes(self):
        tensor = torch.zeros((1, 3, 512, 512))
        stage0 = enet.model.InitialBlock()
        tensor = stage0(tensor)
        self.assertEqual((1, 16, 256, 256), tensor.shape)
        stage1 = enet.model.EnetStage1((128, 128))
        tensor, indices1 = stage1(tensor)
        self.assertEqual((1, 64, 128, 128), tensor.shape)
        self.assertEqual((1, 64, 128, 128), indices1.shape)
        stage2 = enet.model.EnetStage2((64, 64))
        tensor, indices2 = stage2(tensor, indices1)
        self.assertEqual((1, 128, 64, 64), tensor.shape)
        self.assertEqual((1, 128, 64, 64), indices2.shape)
        stage3 = enet.model.EnetStage2((64, 64), False)
        tensor, _ = stage3(tensor, indices2)
        self.assertEqual((1, 128, 64, 64), tensor.shape)
        stage4 = enet.model.EnetStage4((128, 128))
        tensor, _ = stage4(tensor, indices2)
        self.assertEqual((1, 64, 128, 128), tensor.shape)
        stage5 = enet.model.EnetStage5((256, 256))
        tensor, _ = stage5(tensor, indices1)
        self.assertEqual((1, 16, 256, 256), tensor.shape)
        full_conv = nn.ConvTranspose2d(16, 3, 2, 2, bias=False)
        tensor = full_conv(tensor)
        self.assertEqual((1, 3, 512, 512), tensor.shape)

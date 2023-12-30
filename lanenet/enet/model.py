from enum import Enum, auto

import torch as torch
from torch import nn


class EnetSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class InitialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv2d(3, 13, 3, 2, padding=1, bias=False)
        self.max_pooling = nn.MaxPool2d((2, 2), (2, 2), return_indices=True)

    def forward(self, t):
        # shape of t is (B, 3, W, H)
        t1 = self.cnn(t)
        t2, _ = self.max_pooling(t)
        return torch.concat((t1, t2), dim=1)


class ConvolutionType(Enum):
    REGULAR = auto()
    DECONVOLUTION = auto()
    ASYMMETRIC = auto()


class BottleneckType(Enum):
    UPSAMPLING = auto()
    DOWNSAMPLING = auto()
    REGULAR = auto()


class Bottleneck(nn.Module):
    def __init__(self, bottleneck_type: BottleneckType, conv_type: ConvolutionType,
                 in_channels, out_channels,
                 kernel_size, stride,
                 regularizer_prob, output_size, dilation):
        super().__init__()
        # Bottleneck configuration
        conv_out_features = out_channels
        if bottleneck_type == BottleneckType.REGULAR:
            conv_out_features = in_channels
            self.projection = nn.Conv2d(in_channels, conv_out_features // 2, 1, 1, bias=False)
            self.batch_norm_1 = nn.BatchNorm2d(conv_out_features // 2)
            self.prelu_1 = torch.nn.PReLU(conv_out_features // 2)
            self.batch_norm_2 = nn.BatchNorm2d(conv_out_features // 2)
            self.prelu_2 = torch.nn.PReLU(conv_out_features // 2)
            self.expansion = nn.Conv2d(conv_out_features // 2, conv_out_features, 1, 1, bias=False)
            self.batch_norm_3 = nn.BatchNorm2d(out_channels)
            self.max_pooling = None
            self.padding = None
        elif bottleneck_type == BottleneckType.DOWNSAMPLING:
            conv_out_features = out_channels - in_channels
            self.projection = nn.Conv2d(in_channels, conv_out_features // 2, 2, 2, bias=False)
            self.batch_norm_1 = nn.BatchNorm2d(conv_out_features // 2)
            self.prelu_1 = torch.nn.PReLU(conv_out_features // 2)
            self.batch_norm_2 = nn.BatchNorm2d(conv_out_features // 2)
            self.prelu_2 = torch.nn.PReLU(conv_out_features // 2)
            self.expansion = nn.Conv2d(conv_out_features // 2, conv_out_features, 1, 1, bias=False)
            self.batch_norm_3 = nn.BatchNorm2d(out_channels)
            self.max_pooling = nn.MaxPool2d(2, 2, return_indices=True)
            self.padding = torch.nn.ConstantPad3d((0, 0, 0, 0, conv_out_features, 0), value=0)
        elif bottleneck_type == BottleneckType.UPSAMPLING:
            conv_out_features = out_channels // 2
            self.projection = nn.Conv2d(in_channels, conv_out_features // 2, 1, 1, bias=False)
            self.batch_norm_1 = nn.BatchNorm2d(conv_out_features // 2)
            self.prelu_1 = torch.nn.PReLU(conv_out_features // 2)
            self.batch_norm_2 = nn.BatchNorm2d(conv_out_features)
            self.prelu_2 = torch.nn.PReLU(conv_out_features)
            self.expansion = nn.Conv2d(conv_out_features, conv_out_features, 1, 1, bias=False)
            self.batch_norm_3 = nn.BatchNorm2d(out_channels)
            self.max_pooling = nn.MaxUnpool2d(2, 2)
            self.padding = nn.Conv2d(in_channels, conv_out_features, kernel_size=3, stride=1, bias=False,
                                     padding='same')
        # Convolution configuration
        if conv_type == ConvolutionType.REGULAR:
            self.conv = nn.Conv2d(conv_out_features // 2, conv_out_features // 2, kernel_size, stride,
                                  dilation=dilation, padding='same')
        elif conv_type == ConvolutionType.DECONVOLUTION:
            self.conv = nn.ConvTranspose2d(conv_out_features // 2, conv_out_features, kernel_size, stride,
                                           dilation=dilation, padding_mode='zeros', bias=False)
        elif conv_type == ConvolutionType.ASYMMETRIC:
            self.conv = nn.Sequential(
                nn.Conv2d(conv_out_features // 2, conv_out_features // 2, kernel_size, stride,
                          dilation=dilation, padding='same', bias=False),
                nn.Conv2d(conv_out_features // 2, conv_out_features // 2, (kernel_size[1], kernel_size[0]), stride,
                          dilation=dilation, padding='same', bias=False)
            )
        self.regularizer = nn.FeatureAlphaDropout(regularizer_prob)
        self.prelu_3 = torch.nn.PReLU(out_channels)
        self.output_size = output_size

    def forward(self, t, indices=None):
        out_0 = self.projection(t)
        out_0 = self.batch_norm_1(out_0)
        out_0 = self.prelu_1(out_0)
        out_1 = self.conv(out_0)
        out_1 = self.batch_norm_2(out_1)
        out_1 = self.prelu_2(out_1)
        out_2 = self.expansion(out_1)
        out_3 = self.regularizer(out_2)
        if self.max_pooling is not None:
            out_indices = None
            if isinstance(self.max_pooling, nn.MaxUnpool2d):
                out_4 = self.max_pooling(t, indices, output_size=self.output_size)
                out_4 = self.padding(out_4)  # Convolution without bias
            else:
                (out_4, out_indices) = self.max_pooling(t)
                if out_indices is not None and self.padding is not None:
                    out_indices = self.padding(out_indices)
            out_final = torch.concat((out_3, out_4), dim=1)
            out_final = self.batch_norm_3(out_final)
            out_final = self.prelu_3(out_final)
            return out_final, out_indices
        out_final = self.batch_norm_3(out_3)
        out_final = self.prelu_3(out_final)
        return out_final, indices


class EnetStage1(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        bottlenecks = list()
        bottlenecks.append(Bottleneck(
            BottleneckType.DOWNSAMPLING, ConvolutionType.REGULAR, 16, 64, 3, 1, 0.01, output_size, 1))
        for i in range(1, 5):
            bottlenecks.append(Bottleneck(
                BottleneckType.REGULAR, ConvolutionType.REGULAR, 64, 64, 3, 1, 0.01, output_size, 1))
        self.model = EnetSequential(*bottlenecks)

    def forward(self, t, indices=None):
        return self.model(t, indices)


class EnetStage2(nn.Module):
    def __init__(self, output_size, down_sample=True):
        super().__init__()
        bottlenecks = list()
        # 2.0
        if down_sample:
            bottlenecks.append(
                Bottleneck(BottleneckType.DOWNSAMPLING, ConvolutionType.REGULAR, 64, 128, 3, 1, 0.01, output_size, 1))
        # 2.1
        bottlenecks.append(
            Bottleneck(BottleneckType.REGULAR, ConvolutionType.REGULAR, 128, 128, 3, 1, 0.1, output_size, 1))
        # 2.2
        bottlenecks.append(
            Bottleneck(BottleneckType.REGULAR, ConvolutionType.REGULAR, 128, 128, 3, 1, 0.1, output_size, 2))
        # 2.3
        bottlenecks.append(
            Bottleneck(BottleneckType.REGULAR, ConvolutionType.ASYMMETRIC, 128, 128, (5, 1), 1, 0.1, output_size, 2))
        # 2.4
        bottlenecks.append(
            Bottleneck(BottleneckType.REGULAR, ConvolutionType.REGULAR, 128, 128, 3, 1, 0.1, output_size, 4))
        # 2.5
        bottlenecks.append(
            Bottleneck(BottleneckType.REGULAR, ConvolutionType.REGULAR, 128, 128, 3, 1, 0.1, output_size, 1))
        # 2.6
        bottlenecks.append(
            Bottleneck(BottleneckType.REGULAR, ConvolutionType.REGULAR, 128, 128, 3, 1, 0.1, output_size, 8))
        # 2.7
        bottlenecks.append(
            Bottleneck(BottleneckType.REGULAR, ConvolutionType.ASYMMETRIC, 128, 128, (5, 1), 1, 0.1, output_size, 1))
        # 2.8
        bottlenecks.append(
            Bottleneck(BottleneckType.REGULAR, ConvolutionType.REGULAR, 128, 128, 3, 1, 0.1, output_size, 16))
        self.model = EnetSequential(*bottlenecks)

    def forward(self, t, indices=None):
        return self.model(t, indices)


class EnetStage4(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        bottlenecks = list()
        # 4.0
        bottlenecks.append(
            Bottleneck(BottleneckType.UPSAMPLING, ConvolutionType.DECONVOLUTION, 128, 64, 2, 2, 0.1, output_size, 1))
        # 4.1
        bottlenecks.append(
            Bottleneck(BottleneckType.REGULAR, ConvolutionType.REGULAR, 64, 64, 2, 1, 0.1, output_size, 1))
        # 4.2
        bottlenecks.append(
            Bottleneck(BottleneckType.REGULAR, ConvolutionType.REGULAR, 64, 64, 2, 1, 0.1, output_size, 1))
        self.model = EnetSequential(*bottlenecks)

    def forward(self, t, indices=None):
        return self.model(t, indices)


class EnetStage5(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        bottlenecks = list()
        # 5.0
        bottlenecks.append(
            Bottleneck(BottleneckType.UPSAMPLING, ConvolutionType.DECONVOLUTION, 64, 16, 2, 2, 0.1, output_size, 1))
        # 5.1
        bottlenecks.append(
            Bottleneck(BottleneckType.REGULAR, ConvolutionType.REGULAR, 16, 16, 2, 1, 0.1, output_size, 1))
        self.model = EnetSequential(*bottlenecks)

    def forward(self, t, indices=None):
        return self.model(t, indices)


class EnetEncoder(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.stage0 = InitialBlock()
        self.stage1 = EnetStage1((image_size[0] // 4, image_size[1] // 4))
        self.stage2 = EnetStage2((image_size[0] // 8, image_size[1] // 8))
        self.stage3 = EnetStage2((image_size[0] // 8, image_size[1] // 8), False)

    def forward(self, t):
        out_0 = self.stage0(t)
        out_1, indices1 = self.stage1(out_0)
        out_2, indices2 = self.stage2(out_1, indices1)
        out_3, _ = self.stage3(out_2, indices2)
        return out_3, indices1, indices2


class EnetDecoder(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.stage4 = EnetStage4((image_size[0] // 4, image_size[1] // 4))
        self.stage5 = EnetStage5((image_size[0] // 2, image_size[1] // 2))

    def forward(self, out_3, indices1, indices2):
        out_4, _ = self.stage4(out_3, indices2)
        out_5, _ = self.stage5(out_4, indices1)
        return out_5


class Enet(nn.Module):
    def __init__(self, image_size, out_channels=3, include_decoder=True):
        super().__init__()
        self.encoder = EnetEncoder(image_size)
        if include_decoder:
            self.decoder = EnetDecoder(image_size)
            self.full_conv = nn.ConvTranspose2d(16, out_channels, 2, 2, bias=False)
        else:
            self.decoder = None
            self.full_conv = nn.Conv2d(128, out_channels, 1, 1, bias=False)

    def forward(self, t):
        out_3, indices1, indices2 = self.encoder(t)
        out_5 = self.decoder(out_3, indices1, indices2) if self.decoder is not None else out_3
        out_final = self.full_conv(out_5)
        return out_final

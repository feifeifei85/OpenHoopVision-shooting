"""RTMW3D network components and model architecture."""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import CONFIG


class ConvModule(nn.Module):
    """Conv-BN-activation block with OpenMMLab-compatible parameter names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        activation: str = "silu",
        bias: bool = False,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)
        self.activate = nn.SiLU(inplace=True) if activation == "silu" else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise-separable conv with upstream-compatible parameter names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation: str = "silu",
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        conv = partial(ConvModule, activation=activation, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.depthwise_conv = conv(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = conv(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.global_avgpool(x)))


class CSPNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        add_identity: bool,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = ConvModule(channels, channels, 3, padding=1, bn_eps=bn_eps, bn_momentum=bn_momentum)
        # CSPNeXt uses a 5x5 depthwise-separable second convolution.
        self.conv2 = DepthwiseSeparableConvModule(
            channels, channels, 5, padding=2, bn_eps=bn_eps, bn_momentum=bn_momentum
        )
        self.add_identity = add_identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return y + x if self.add_identity else y


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        add_identity: bool,
        channel_attention: bool = False,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        mid_channels = out_channels // 2
        conv = partial(ConvModule, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.main_conv = conv(in_channels, mid_channels, 1)
        self.short_conv = conv(in_channels, mid_channels, 1)
        self.final_conv = conv(mid_channels * 2, out_channels, 1)
        self.blocks = nn.Sequential(
            *[CSPNeXtBlock(mid_channels, add_identity, bn_eps, bn_momentum) for _ in range(num_blocks)]
        )
        self.channel_attention = channel_attention
        if channel_attention:
            self.attention = ChannelAttention(mid_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_main = self.blocks(self.main_conv(x))
        x_short = self.short_conv(x)
        x = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            x = self.attention(x)
        return self.final_conv(x)


class SPPBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        mid_channels = in_channels // 2
        conv = partial(ConvModule, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.conv1 = conv(in_channels, mid_channels, 1)
        self.poolings = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2) for size in (5, 9, 13)]
        )
        self.conv2 = conv(mid_channels * 4, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.poolings], dim=1)
        return self.conv2(x)


class CSPNeXtBackbone(nn.Module):
    """P5 CSPNeXt backbone returning C3, C4 and C5 feature maps."""

    def __init__(
        self,
        widen_factor: float = 1.0,
        deepen_factor: float = 1.0,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        conv = partial(ConvModule, bn_eps=bn_eps, bn_momentum=bn_momentum)
        stem_channels = int(64 * widen_factor // 2)
        channels = tuple(int(channel * widen_factor) for channel in (64, 128, 256, 512, 1024))
        depths = tuple(max(round(depth * deepen_factor), 1) for depth in (3, 6, 6, 3))
        self.stem = nn.Sequential(
            conv(3, stem_channels, 3, stride=2, padding=1),
            conv(stem_channels, stem_channels, 3, padding=1),
            conv(stem_channels, channels[0], 3, padding=1),
        )
        for index, (out_channels, depth) in enumerate(zip(channels[1:], depths), start=1):
            layers = [conv(channels[index - 1], out_channels, 3, stride=2, padding=1)]
            if index == 4:
                layers.append(SPPBottleneck(out_channels, out_channels, bn_eps, bn_momentum))
            layers.append(CSPLayer(out_channels, out_channels, depth, index < 4, True, bn_eps, bn_momentum))
            self.add_module(f"stage{index}", nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        features = []
        for index in range(1, 5):
            x = getattr(self, f"stage{index}")(x)
            if index > 1:
                features.append(x)
        return tuple(features)


class CSPNeXtPAFPN(nn.Module):
    """CSPNeXt path aggregation neck, optionally projecting output channels."""

    def __init__(
        self,
        in_channels: tuple[int, ...] = (256, 512, 1024),
        output_channels: int | None = None,
        output_indices: tuple[int, ...] = (1, 2),
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        conv = partial(ConvModule, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.in_channels = in_channels
        self.output_indices = output_indices
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce_layers = nn.ModuleList(
            conv(in_channels[index], in_channels[index - 1], 1) for index in range(len(in_channels) - 1, 0, -1)
        )
        self.top_down_blocks = nn.ModuleList(
            CSPLayer(
                in_channels[index - 1] * 2,
                in_channels[index - 1],
                2,
                False,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum,
            )
            for index in range(len(in_channels) - 1, 0, -1)
        )
        self.downsamples = nn.ModuleList(
            conv(channels, channels, 3, stride=2, padding=1) for channels in in_channels[:-1]
        )
        self.bottom_up_blocks = nn.ModuleList(
            CSPLayer(
                in_channels[index] * 2,
                in_channels[index + 1],
                2,
                False,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum,
            )
            for index in range(len(in_channels) - 1)
        )
        if output_channels is not None:
            self.out_convs = nn.ModuleList(conv(channels, output_channels, 3, padding=1) for channels in in_channels)

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        inner_outs = [inputs[-1]]
        for reduce, block, skip in zip(self.reduce_layers, self.top_down_blocks, reversed(inputs[:-1])):
            feat_high = reduce(inner_outs[0])
            inner_outs[0] = feat_high
            inner_out = block(torch.cat([self.upsample(feat_high), skip], dim=1))
            inner_outs.insert(0, inner_out)
        outs = [inner_outs[0]]
        for downsample, block, skip in zip(self.downsamples, self.bottom_up_blocks, inner_outs[1:]):
            down = downsample(outs[-1])
            outs.append(block(torch.cat([down, skip], dim=1)))
        if hasattr(self, "out_convs"):
            outs = [conv(feature) for conv, feature in zip(self.out_convs, outs)]
        return tuple(outs[index] for index in self.output_indices)


class ScaleNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Scale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class RTMCCBlock(nn.Module):
    """The self-attention GAU block used by the official RTMW3D-L head."""

    def __init__(self) -> None:
        super().__init__()
        self.o = nn.Linear(512, 256, bias=False)
        self.uv = nn.Linear(256, 1152, bias=False)
        self.gamma = nn.Parameter(torch.rand((2, 128)))
        self.beta = nn.Parameter(torch.rand((2, 128)))
        self.ln = ScaleNorm(256)
        self.act_fn = nn.SiLU(inplace=True)
        self.res_scale = Scale(256)
        self.sqrt_s = math.sqrt(128)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.res_scale(x)
        uv = self.act_fn(self.uv(self.ln(x)))
        u, v, base = torch.split(uv, [512, 512, 128], dim=2)
        base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
        q, k = torch.unbind(base, dim=2)
        qk = torch.bmm(q, k.permute(0, 2, 1))
        kernel = torch.square(F.relu(qk / self.sqrt_s))
        kernel = self.dropout(kernel)
        attended = u * torch.bmm(kernel, v)
        return shortcut + self.o(attended)


class RTMW3DHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ps = nn.PixelShuffle(2)
        self.conv_dec = ConvModule(256, 256, 7, padding=3, activation="relu")
        self.final_layer = ConvModule(1024, CONFIG["KEYPOINT_COUNT"], 7, padding=3, activation="relu")
        self.final_layer2 = ConvModule(768, CONFIG["KEYPOINT_COUNT"], 7, padding=3, activation="relu")
        self.mlp = nn.Sequential(ScaleNorm(108), nn.Linear(108, 128, bias=False))
        self.mlp2 = nn.Sequential(ScaleNorm(432), nn.Linear(432, 128, bias=False))
        self.gau = RTMCCBlock()
        self.cls_x = nn.Linear(256, 576, bias=False)
        self.cls_y = nn.Linear(256, 768, bias=False)
        self.cls_z = nn.Linear(256, 576, bias=False)

    def forward(self, feats: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_b, enc_t = feats
        feats_t = self.mlp(torch.flatten(self.final_layer(enc_t), 2))
        dec_t = self.conv_dec(self.ps(enc_t))
        enc_b = torch.cat([dec_t, enc_b], dim=1)
        feats_b = self.mlp2(torch.flatten(self.final_layer2(enc_b), 2))
        feats = self.gau(torch.cat([feats_t, feats_b], dim=2))
        return self.cls_x(feats), self.cls_y(feats), self.cls_z(feats)


class RTMW3DModel(nn.Module):
    """Predict three-dimensional human pose keypoints from image crops."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = CSPNeXtBackbone()
        self.neck = CSPNeXtPAFPN()
        self.head = RTMW3DHead()

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return x, y, and depth logits for the input image batch."""
        return self.head(self.neck(self.backbone(image)))

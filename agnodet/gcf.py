"""Gated complementary fusion (GCF) for the two-stream AgnoDet backbone.

GCF merges the semantic stream of a frozen DINOv3 ViT with the detail stream of a
lightweight convolutional prior module. Instead of concatenating the two streams,
it derives an explicit discrepancy map |a - b| and consistency map a * b, uses them
to predict a per-pixel per-channel softmax gate over the two streams, and adds a
scaled complementary residual built from the same two maps.
"""

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """Conv2d(bias=False) -> BatchNorm2d -> SiLU, padded to preserve spatial size."""

    def __init__(self, in_ch, out_ch, kernel_size=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=1,
                              padding=kernel_size // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _GateEncoder(nn.Module):
    """Maps the 4-way stacked interaction tensor to two-stream gate logits."""

    def __init__(self, channels):
        super().__init__()
        merged = channels * 4
        self.local = ConvBNAct(merged, merged, kernel_size=3, groups=merged)
        self.mix = ConvBNAct(merged, channels, kernel_size=1)
        self.out = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)

    def forward(self, x):
        return self.out(self.mix(self.local(x)))


class _ComplementaryResidual(nn.Module):
    """Builds a residual correction from the discrepancy and consistency maps."""

    def __init__(self, channels):
        super().__init__()
        self.discrepancy = ConvBNAct(channels, channels, kernel_size=3, groups=channels)
        self.consistency = ConvBNAct(channels, channels, kernel_size=1)
        self.project = ConvBNAct(channels, channels, kernel_size=1)

    def forward(self, discrepancy, consistency):
        return self.project(self.discrepancy(discrepancy) + self.consistency(consistency))


class GCF(nn.Module):
    """Gated complementary fusion of two equally-sized feature maps.

    Args:
        inc: two-element sequence with the channel count of each input stream.
        ouc: output channel count; both streams are first aligned to it.
        residual_scale_init: initial value of the learnable complementary residual scale.
    """

    def __init__(self, inc, ouc, residual_scale_init=0.1):
        super().__init__()
        if len(inc) != 2:
            raise ValueError(f"GCF takes exactly two input streams, got {len(inc)}")

        self.align_top = ConvBNAct(inc[0], ouc)
        self.align_bottom = ConvBNAct(inc[1], ouc)
        self.interaction = _GateEncoder(ouc)
        self.complementary = _ComplementaryResidual(ouc)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.output = ConvBNAct(ouc, ouc)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("GCF takes a list or tuple of two feature maps")

        x_top, x_bottom = x
        if x_top.shape[-2:] != x_bottom.shape[-2:]:
            raise ValueError("GCF requires both inputs to share the same spatial shape")

        top = self.align_top(x_top)
        bottom = self.align_bottom(x_bottom)

        discrepancy = torch.abs(top - bottom)
        consistency = top * bottom
        interaction = torch.cat((top, bottom, discrepancy, consistency), dim=1)

        gate_logits = self.interaction(interaction)
        b, two_c, h, w = gate_logits.shape
        gate_weights = torch.softmax(gate_logits.view(b, 2, two_c // 2, h, w), dim=1)
        top_weight, bottom_weight = gate_weights.unbind(dim=1)

        residual = self.complementary(discrepancy, consistency)
        fused = top_weight * top + bottom_weight * bottom + self.residual_scale * residual
        return self.output(fused)

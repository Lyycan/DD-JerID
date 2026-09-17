"""Low-rank adapters for the ProtoLoRA identity backbone.

A LoRA wrapper keeps the wrapped layer frozen and adds a trainable low-rank
bypass: out = base(x) + (alpha / r) * B(A(x)). B is zero-initialised, so a freshly
injected model is numerically identical to the frozen backbone and each period's
adaptation starts from the previous period's state rather than from noise.
"""

import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen nn.Linear plus a trainable rank-r bypass."""

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scale = alpha / r
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


class LoRAConv2d(nn.Module):
    """Frozen nn.Conv2d plus a trainable rank-r bypass (1x1 down, base-shaped up)."""

    def __init__(self, base: nn.Conv2d, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scale = alpha / r
        self.A = nn.Conv2d(base.in_channels, r, kernel_size=1, stride=1, bias=False)
        self.B = nn.Conv2d(r, base.out_channels, kernel_size=base.kernel_size,
                           stride=base.stride, padding=base.padding,
                           dilation=base.dilation, groups=1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


def _resolve_parent(root, name):
    """Resolve a dotted path to (parent_module, leaf_name), honouring numeric indices."""
    *parents, leaf = name.split(".")
    obj = root
    for p in parents:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj, leaf


def _get_child(obj, leaf):
    return obj[int(leaf)] if leaf.isdigit() else getattr(obj, leaf)


def _set_child(obj, leaf, value):
    if leaf.isdigit():
        obj[int(leaf)] = value
    else:
        setattr(obj, leaf, value)


def inject_lora_vit_blocks(blocks, r=8, alpha=16,
                           target=("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")):
    """Inject LoRA into the named linear layers of every transformer block."""
    lora_params = []
    for blk in blocks:
        for name in target:
            try:
                obj, leaf = _resolve_parent(blk, name)
                lin = _get_child(obj, leaf)
            except AttributeError:
                continue
            if isinstance(lin, LoRALinear):
                lin = lin.base
            if not isinstance(lin, nn.Linear):
                continue
            wrapped = LoRALinear(lin, r=r, alpha=alpha).to(next(lin.parameters()).device)
            _set_child(obj, leaf, wrapped)
            lora_params += [wrapped.A.weight, wrapped.B.weight]
    return lora_params


def inject_lora_conv(backbone, r=8, alpha=16, last_n=12):
    """Inject LoRA into the last `last_n` non-grouped Conv2d layers of a CNN backbone.

    Depthwise layers are skipped: a rank-r bypass across channels is not meaningful
    when the base layer does not mix channels in the first place.
    """
    conv_mods = [(n, m) for n, m in backbone.named_modules()
                 if isinstance(m, nn.Conv2d) and m.groups == 1]
    lora_params = []
    for name, conv in conv_mods[-last_n:]:
        obj, leaf = _resolve_parent(backbone, name)
        if isinstance(_get_child(obj, leaf), LoRAConv2d):
            continue
        wrapped = LoRAConv2d(conv, r=r, alpha=alpha).to(next(conv.parameters()).device)
        _set_child(obj, leaf, wrapped)
        lora_params += [wrapped.A.weight, wrapped.B.weight]
    return lora_params


def inject_lora(net, r=8, alpha=16):
    """Inject LoRA into net.backbone, dispatching on whether it is a ViT.

    r <= 0 injects nothing and returns an empty list, which gives the
    "frozen backbone, embedding head only" control condition.
    """
    if r <= 0:
        return []
    bb = net.backbone
    if getattr(bb, "is_vit", False):
        blocks = getattr(bb, "vit_blocks", None)
        if blocks is None:
            blocks = bb.model.blocks
        return inject_lora_vit_blocks(blocks, r=r, alpha=alpha)
    return inject_lora_conv(bb, r=r, alpha=alpha)

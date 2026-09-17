"""Identity embedding network for ProtoLoRA.

A frozen self-supervised backbone is followed by a pooling step and a small
projection head; identities are represented by prototypes in the resulting
256-d space rather than by classifier weights, so enrolling an animal never
requires touching the network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_transforms(img_size):
    train = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    evaluate = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, evaluate


class CropDS(torch.utils.data.Dataset):
    def __init__(self, items, tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tfm(Image.open(path).convert("RGB")), label


class ArcFace(nn.Module):
    """Additive angular margin head, used only while adapting; discarded at inference."""

    def __init__(self, in_dim, n_cls, s=32.0, m=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m

    def forward(self, emb, label):
        w = F.normalize(self.W, dim=1)
        cos = emb @ w.t()
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(label, cos.size(1)).float()
        return self.s * (onehot * target + (1 - onehot) * cos)


class TimmBackbone(nn.Module):
    """Wraps a timm model so it returns a list of feature maps in (B, C, H, W).

    Convolutional models go through features_only; transformers fall back to
    forward_features with the prefix tokens dropped and the patch tokens folded
    back onto a spatial grid.
    """

    def __init__(self, name, pretrained=True):
        super().__init__()
        import timm
        self.name = name
        last_err = None
        # timm>=1.0 ViTs accept features_only but need dynamic_img_size, otherwise
        # patch_embed asserts the input matches the pretraining resolution.
        for kw in (dict(features_only=True, dynamic_img_size=True),
                   dict(features_only=True),
                   dict(num_classes=0, dynamic_img_size=True),
                   dict(num_classes=0)):
            try:
                self.model = timm.create_model(name, pretrained=pretrained, **kw)
                self.features_only = kw.get("features_only", False)
                break
            except Exception as e:  # noqa: PERF203
                last_err = e
        if not hasattr(self, "model"):
            raise RuntimeError(f"could not build timm backbone {name}: {last_err}")

        # Detect transformer blocks structurally, independently of features_only:
        # with features_only=True timm nests the ViT, so the blocks are not at top
        # level, and timm stores them in an nn.Sequential rather than an nn.ModuleList.
        self.vit_blocks = None
        for m in self.model.modules():
            if isinstance(m, (nn.ModuleList, nn.Sequential)) and len(m) \
                    and hasattr(m[0], "attn"):
                self.vit_blocks = m
                break
        self.is_vit = self.vit_blocks is not None

        if self.features_only:
            chs = list(self.model.feature_info.channels())
            self.n_keep = min(3, len(chs))
            self.chs = chs[-self.n_keep:]
        else:
            self.chs = [self.model.num_features]
            self.n_keep = 1

    def forward(self, x):
        if self.features_only:
            outs = list(self.model(x))[-self.n_keep:]
            return [o.permute(0, 3, 1, 2)
                    if o.shape[-1] in self.chs and o.shape[1] not in self.chs else o
                    for o in outs]
        f = self.model.forward_features(x)
        npre = getattr(self.model, "num_prefix_tokens", 1)
        f = f[:, npre:]
        b, n, c = f.shape
        ph, pw = self.model.patch_embed.patch_size
        h, w = x.shape[2] // ph, x.shape[3] // pw
        if h * w != n:
            h = w = int(round(n ** 0.5))
        return [f.transpose(1, 2).reshape(b, c, h, w)]


class ReIDNet(nn.Module):
    """Backbone -> pooling -> Linear(256) + BN -> L2-normalised embedding.

    feat_scales='last' pools only the coarsest feature map; 'all' concatenates the
    pooled vector of every scale the backbone emits.
    """

    def __init__(self, backbone, in_dim, emb_dim=256, pool="gem", gem_p=3.0,
                 feat_scales="last"):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(in_dim, emb_dim), nn.BatchNorm1d(emb_dim))
        self.pool = pool
        self.feat_scales = feat_scales
        if pool == "gem":
            self.gem_p = nn.Parameter(torch.ones(1) * gem_p)

    def _pool(self, f):
        if self.pool == "gem":
            p = self.gem_p.clamp(min=1.0)
            return f.clamp(min=1e-6).pow(p).mean(dim=(2, 3)).pow(1.0 / p)
        return f.mean(dim=(2, 3))

    def forward(self, x):
        feats = self.backbone(x)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        use = list(feats) if self.feat_scales == "all" else [feats[-1]]
        g = torch.cat([self._pool(f) for f in use], dim=1)
        return F.normalize(self.head(g), dim=1)


def build(id_backbone, img_size, pool="gem", gem_p=3.0, feat_scales="last",
          pretrained=True, frozen=True):
    """Build the identity network on a timm backbone.

    pretrained=False skips the pretrained-weight download; use it whenever the
    caller immediately loads a trained checkpoint over the whole network.
    """
    if not id_backbone.startswith("timm:"):
        raise ValueError(f"id_backbone must be 'timm:<name>', got {id_backbone}")
    bb = TimmBackbone(id_backbone.split(":", 1)[1], pretrained=pretrained).to(DEVICE)

    bb.eval().to(DEVICE)
    with torch.no_grad():
        o = bb(torch.rand(1, 3, img_size, img_size).to(DEVICE))
        o = list(o) if isinstance(o, (list, tuple)) else [o]
        use = o if feat_scales == "all" else [o[-1]]
        in_dim = sum(f.shape[1] for f in use)

    if frozen:
        for p in bb.parameters():
            p.requires_grad = False
    return ReIDNet(bb, in_dim, pool=pool, gem_p=gem_p,
                   feat_scales=feat_scales).to(DEVICE)


@torch.no_grad()
def embed_paths(net, paths, eval_tfm, bs=128):
    """Embed a list of image paths; returns (N, 256) L2-normalised features."""
    net.eval()          # keep BatchNorm running statistics: re-estimating them on a
                        # handful of enrolment crops measurably degrades retrieval
    feats, batch = [], []
    for p in paths:
        batch.append(eval_tfm(Image.open(p).convert("RGB")))
        if len(batch) == bs:
            feats.append(net(torch.stack(batch).to(DEVICE)).cpu().numpy())
            batch = []
    if batch:
        feats.append(net(torch.stack(batch).to(DEVICE)).cpu().numpy())
    return np.concatenate(feats)

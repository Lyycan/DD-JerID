"""ProtoLoRA: per-period low-rank adaptation with a non-parametric prototype gallery.

For each period in turn the script adapts the LoRA bypass and the embedding head on
that period's update set, rebuilds one prototype per identity from the same crops,
and scores the period's held-out crops by cosine retrieval against those prototypes.
LoRA weights and the head carry over between periods; the gallery is rebuilt from
scratch, which is what lets a newly arrived animal be enrolled without training.

Crops are expected as <reid_root>/<period>/<identity>/<split>_<frame>_b<k>.jpg. The
leading split token is what keeps the update set and the evaluation set disjoint.

Example:
  python -m protolora.continual --reid-root /data/JerFace-LT/crops \
      --periods P1 P2 P3 P4 --save-dir runs/protolora
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .embedding import DEVICE, ArcFace, CropDS, build, embed_paths, make_transforms
from .lora import inject_lora

DINOV2_VITS = "timm:vit_small_patch14_dinov2.lvd142m"


def split_period(root, period, shot_ratio, seed=0, proto_cap=150, test_cap=200):
    """Split one period's crops into (update set, evaluation set), keyed by identity.

    The update set is drawn only from crops whose filename begins with `train_`, and
    the evaluation set only from `test_` crops, so no crop used to adapt the model or
    to build a prototype can reappear as a query. proto_cap bounds how many crops an
    operator would realistically label per period; test_cap bounds evaluation cost.
    """
    root = Path(root)
    rng = random.Random(seed)
    update, test = [], []
    identities = sorted(d.name for d in (root / period).iterdir() if d.is_dir())
    for ident in identities:
        imgs = sorted((root / period / ident).glob("*.jpg"))
        if not imgs:
            continue
        by = {"train": [], "val": [], "test": []}
        for p in imgs:
            prefix = p.name.split("_", 1)[0]
            if prefix in by:
                by[prefix].append(p)
        if not (by["train"] or by["test"]):
            raise ValueError(
                f"{root / period / ident}: crop names carry no split prefix; expected "
                f"files like train_<frame>_b0.jpg")
        k = max(1, int(len(imgs) * shot_ratio))
        pool = list(by["train"])
        rng.shuffle(pool)
        update.append((ident, pool[:k][:proto_cap]))
        test.append((ident, sorted(by["test"])[:test_cap]))
    return update, test


def build_protos(net, update, eval_tfm, bs=128):
    """One L2-normalised mean prototype per identity. No gradients, no training."""
    idents, protos = [], []
    for ident, paths in update:
        if not paths:
            continue
        mean = embed_paths(net, paths, eval_tfm, bs=bs).mean(0)
        idents.append(ident)
        protos.append(mean / (np.linalg.norm(mean) + 1e-9))
    return idents, np.stack(protos)


def evaluate(net, gallery_idents, protos, test, eval_tfm, bs=128):
    """Cosine retrieval against the prototypes; returns (rank-1, mAP, n_ids, n_query)."""
    n_correct = n_query = 0
    aps = []
    for ident, paths in test:
        if ident not in gallery_idents or not paths:
            continue
        qf = embed_paths(net, paths, eval_tfm, bs=bs)
        sims = qf @ protos.T
        gt = gallery_idents.index(ident)
        n_correct += int((sims.argmax(1) == gt).sum())
        n_query += len(qf)
        ranks = np.where(np.argsort(-sims, 1) == gt)[1] + 1
        aps.append(np.mean(1.0 / ranks))
    return (n_correct / max(n_query, 1),
            float(np.mean(aps)) if aps else 0.0, len(aps), n_query)


def adapt(net, lora_params, items, n_cls, epochs, train_tfm,
          lr_head=1e-3, lr_lora=1e-4, bs=128):
    """Train the embedding head and the LoRA bypass; the backbone stays frozen."""
    if not items:
        return
    dl = torch.utils.data.DataLoader(
        CropDS(items, train_tfm), batch_size=bs, shuffle=True,
        num_workers=8, drop_last=len(items) > bs)
    arc = ArcFace(256, n_cls).to(DEVICE)
    groups = [{"params": list(net.head.parameters()) + list(arc.parameters()),
               "lr": lr_head}]
    if lora_params:
        groups.append({"params": lora_params, "lr": lr_lora})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        net.train()
        net.backbone.eval()      # frozen weights and frozen BatchNorm statistics
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(arc(net(x), y), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reid-root", required=True,
                    help="crop root containing one directory per period")
    ap.add_argument("--periods", nargs="+", required=True,
                    help="period directory names, in chronological order")
    ap.add_argument("--id-backbone", default=DINOV2_VITS)
    ap.add_argument("--img-size", type=int, default=252,
                    help="must be a multiple of the backbone patch size (14 for DINOv2)")
    ap.add_argument("--pool", default="gem", choices=["gap", "gem"])
    ap.add_argument("--gem-p", type=float, default=3.0)
    ap.add_argument("--lora-r", type=int, default=8,
                    help="LoRA rank; 0 injects nothing and trains the head only")
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--shot-ratio", type=float, default=0.2)
    ap.add_argument("--replay", type=float, default=0.0,
                    help="fraction of earlier periods' update crops to mix back in")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default=None,
                    help="if set, writes <save-dir>/<period>.pth after each period")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    rng = random.Random(args.seed)
    train_tfm, eval_tfm = make_transforms(args.img_size)

    net = build(args.id_backbone, args.img_size, pool=args.pool,
                gem_p=args.gem_p, frozen=True)
    lora_params = inject_lora(net, r=args.lora_r, alpha=args.lora_alpha)
    n_lora = sum(p.numel() for p in lora_params)
    n_head = sum(p.numel() for p in net.head.parameters())
    print(f"[protolora] backbone={args.id_backbone} img={args.img_size} "
          f"pool={args.pool} lora_r={args.lora_r} replay={args.replay}")
    print(f"[protolora] trainable per period = {(n_lora + n_head)/1e6:.3f} M "
          f"(LoRA {n_lora/1e6:.3f} M + head {n_head/1e6:.3f} M)")
    print(f"{'period':10s} {'rank1':>8s} {'mAP':>8s} {'ids':>5s} {'query':>7s}")

    seen = []
    for period in args.periods:
        update, test = split_period(args.reid_root, period, args.shot_ratio, args.seed)
        present = [ident for ident, paths in update if paths]
        items = [(p, i) for i, (ident, paths) in enumerate(
            [u for u in update if u[1]]) for p in paths]

        if args.replay > 0 and seen:
            lab_of = {ident: i for i, ident in enumerate(present)}
            for ident, paths in seen:
                if ident in lab_of and paths:
                    pick = list(paths)
                    rng.shuffle(pick)
                    k = max(1, int(len(paths) * args.replay))
                    items += [(p, lab_of[ident]) for p in pick[:k]]

        epochs = max(args.epochs, 25) if period == args.periods[0] else args.epochs
        adapt(net, lora_params, items, len(present), epochs, train_tfm, bs=args.bs)

        gallery_idents, protos = build_protos(net, update, eval_tfm, bs=args.bs)
        r1, mean_ap, n_ids, n_query = evaluate(net, gallery_idents, protos, test,
                                               eval_tfm, bs=args.bs)
        print(f"{period:10s} {r1:8.3f} {mean_ap:8.3f} {n_ids:>5d} {n_query:>7d}")

        if args.save_dir:
            sd = Path(args.save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            torch.save({"net": net.state_dict(), "identities": gallery_idents,
                        "protos": protos, "period": period,
                        "id_backbone": args.id_backbone, "img_size": args.img_size,
                        "pool": args.pool, "gem_p": args.gem_p,
                        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                        "rank1": r1, "map": mean_ap}, sd / f"{period}.pth")

        seen += [(ident, paths) for ident, paths in update if paths]


if __name__ == "__main__":
    main()

# DD-JerID

Reference implementation of the two components described in *"Long-term Cattle
Face Recognition in a Changing Herd: A 29-month Evaluation of a Doubly Decoupled
Framework"*:

- **AgnoDet** — identity-agnostic single-class cattle-face detector. A frozen
  DINOv3 ViT-Tiny semantic stream and a convolutional detail stream are merged by
  **GCF**, a gated complementary fusion block, instead of being concatenated.
- **ProtoLoRA** — identity recognition on face crops. A frozen DINOv2 ViT-S/14
  backbone is adapted per period through a rank-8 LoRA bypass and a 256-d
  embedding head (0.689 M trainable parameters per period); identities live in a
  non-parametric prototype gallery.

## Layout

```
agnodet/
  gcf.py                      gated complementary fusion (torch only)
  dinov3_adapter.gcf.patch    diff adding fusion_type='gcf' to the DEIMv2 backbone
protolora/
  embedding.py                timm backbone, GeM pooling, embedding head, ArcFace
  lora.py                     LoRA adapters and injection
  continual.py                per-period adaptation + prototype gallery
configs/
  agnodet_dinov3t_gcf.yml     AgnoDet training recipe
  dataset_jerface_single.yml  single-class COCO template
```

## Data

This repository ships no imagery, annotations, splits or trained weights. The
dataset is released separately as **JerFace-LT**, at
[huggingface.co/datasets/Lyycan/JerFace-LT](https://huggingface.co/datasets/Lyycan/JerFace-LT)
under CC BY-SA 4.0, and becomes public once the paper is accepted.

It holds four collection periods spanning about 29 months, 17,804 frames and
96,430 boxes over 36 individuals, with identity labels keyed globally so that the
same animal carries the same id in every period.

```
P1/  2023-06     9,679 frames  50,092 boxes  23 identities
P2/  2024-10-28  2,630 frames  14,282 boxes  22 identities
P3/  2024-10-30  3,075 frames  17,581 boxes  21 identities
P4/  2025-11-02  2,420 frames  14,475 boxes  20 identities
```

Each period carries `images/`, an `annotations.json` in COCO format and a
`metadata.jsonl` for `datasets.load_dataset("imagefolder", ...)`; `identities.csv`
at the root lists which periods each animal appears in.

The release is organised by period and carries no train/val/test split. The
splits used in the paper are time-blocked rather than random, because frames were
sampled about three seconds apart and a randomly held-out frame therefore has a
near-duplicate in the training set. Split by contiguous time blocks or by video
source; the split files themselves are available from the corresponding author on
request.

## Licence

Apache-2.0 (`LICENSE`); upstream attribution in `NOTICE`.

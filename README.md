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

The JerFace-LT dataset is released separately and is not part of this
repository, which ships no imagery, annotations, splits or trained weights.

## Citation

```bibtex
@article{ddjerid,
  title   = {Long-term Cattle Face Recognition in a Changing Herd: A 29-month
             Evaluation of a Doubly Decoupled Framework},
  author  = {Lyycan},
  note    = {https://github.com/Lyycan},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026}
}
```

## Licence

Apache-2.0 (`LICENSE`); upstream attribution in `NOTICE`.

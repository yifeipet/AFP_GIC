# AFP_GIC

This repository provides the evaluation-only release for **AFP-GIC:
Adaptive Fused Prior Transfer for Controllable Generative Image
Compression**.

It contains the minimal code needed to load the released checkpoint and
reproduce the public Kodak evaluation entry used for the paper. Training
code, datasets, and most research artifacts are intentionally excluded
from this release.

Licensing and third-party attribution notes are summarized in
[`LICENSE`](LICENSE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Download the released checkpoint

Download the released model file from this
[Google Drive folder](https://drive.google.com/drive/folders/1qqPyKHtdiIVoiYWl3mZNGnJFXGzhRHTR?usp=drive_link).

Place the checkpoint at:

```text
checkpoint/afp_gic_release/model/afp_gic_release.pth.tar
```

This released checkpoint already contains the frozen prior component
used by the full AFP-GIC model, so no separate prior-weight download is
required.

The parameter counts reported in the paper correspond to the full
AFP-GIC model, including this frozen prior component.

## Environment

Install the dependencies first:

```bash
pip install -r public_release/requirements.txt
```

For results closest to the paper, use the versions pinned in
`public_release/requirements.txt` together with a compatible
PyTorch 2.1.0 / torchvision 0.16.0 build for your platform.

## Dataset layout

For a quick public test, prepare Kodak images under:

```text
datasets/kodak/*.png
```

## Quick test

```bash
python public_release/test.py -d cuda:0 --dataset kodak --qualities 0
```

Full Kodak evaluation across the five released operating points:

```bash
python public_release/test.py -d cuda:0 --dataset kodak --qualities 0 1 2 3 4
```

## Repository layout

```text
.
├── checkpoint/
├── datasets/
├── public_release/
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

`public_release/runtime/` contains the inference-time model code needed
to load the released checkpoint. No training workflow is required for
normal public testing.

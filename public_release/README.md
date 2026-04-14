# Public AFP-GIC Evaluation

This folder provides the public evaluation entry point for AFP-GIC.

Repository-wide licensing and third-party attribution notes are
available in `LICENSE` and `THIRD_PARTY_NOTICES.md` at the repository
root.

Download the released checkpoint from this
[Google Drive folder](https://drive.google.com/drive/folders/1qqPyKHtdiIVoiYWl3mZNGnJFXGzhRHTR?usp=drive_link).

## Environment / setup
Install the dependencies first:

```bash
pip install -r public_release/requirements.txt
```

For results closest to the paper, use the versions listed in `public_release/requirements.txt` and a compatible PyTorch 2.1.0 / torchvision 0.16.0 build for your platform.

Quick Kodak smoke test at the lowest operating point:

```bash
python public_release/test.py -d cuda:0 --dataset kodak --qualities 0
```

Full Kodak evaluation across the five released operating points:

```bash
python public_release/test.py -d cuda:0 --dataset kodak --qualities 0 1 2 3 4
```

The public entry uses the released model alias:

- model name: `afp_gic_release`

## Required files
Place the released checkpoint at:

```text
checkpoint/afp_gic_release/model/afp_gic_release.pth.tar
```

This checkpoint already includes the frozen prior component used by the full AFP-GIC model, so no separate prior-weight download is required.

The parameter counts reported in the paper correspond to the full AFP-GIC model, including this frozen prior component.

Kodak images:

```text
datasets/kodak/*.png
```

## Notes
- The public interface is evaluation-only.
- `public_release/runtime/` contains the inference-time model code needed to load the released checkpoint.
- The large model checkpoint should be distributed as a GitHub Release asset or via Git LFS.
- The research training scripts remain elsewhere in the repository and are not needed for normal public testing.

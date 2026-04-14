# Third-Party Notices

This repository contains original AFP-GIC code and evaluation scripts,
along with code that builds on or derives from upstream projects. The
list below highlights the main third-party components relevant to this
release.

## DC_VIC

AFP-GIC is developed on top of the DC_VIC codebase and keeps the
entropy-constrained codec backbone in that lineage. Parts of the runtime
structure, training/evaluation flow, and supporting code in this
repository are derived from or adapted from the DC_VIC project.

- Upstream project: `https://github.com/iwa-shi/DC_VIC`
- Upstream paper: *Dual-Conditioned Training to Exploit Pre-trained
  Codebook-based Generative Model in Image Compression*, IEEE Access

Please consult the upstream DC_VIC repository for citation guidance and
the applicable licensing terms for reused material.

## AdaCode

AFP-GIC uses AdaCode-related components and pretrained-prior machinery.
The public evaluation runtime includes AdaCode-derived code paths, and
the original project ships its license text in:

- `dc_vic_adacode_merge/external_libs/AdaCode_official/LICENSE`

At the time of preparing this release, that bundled upstream license is
Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0). Please review the upstream license
text directly before reuse or redistribution.

- Upstream project: `https://github.com/KAIST-VICLab/AdaCode`

## BasicSR

This repository also includes or depends on code derived from BasicSR.
The corresponding upstream license text is bundled under:

- `external_libs/BasicSR/LICENSE`
- `external_libs/BasicSR/LICENSE.txt`

- Upstream project: `https://github.com/XPixelGroup/BasicSR`

## CompressAI and other external dependencies

This release depends on external libraries such as CompressAI, PyTorch,
torchvision, timm, LPIPS, DISTS-pytorch, pytorch-msssim, and
pytorch-fid. These dependencies are installed separately by the user
and remain subject to their own licenses.

## Citation and attribution

If you use this repository, please cite the AFP-GIC paper and also cite
the upstream projects whose code or ideas materially contributed to your
usage, especially DC_VIC and AdaCode where appropriate.

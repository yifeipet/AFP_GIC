<h1 align="center">AFP-GIC</h1>
<h3 align="center">Adaptive Fused Prior Transfer for Controllable Generative Image Compression</h3>

<p align="center">
  Yifei Pei, Ying Liu, and Nam Ling<br>
  Santa Clara University<br>
  <strong>IEEE Access, 2026</strong>
</p>

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/11712133">Paper</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2605.16817">arXiv + Supplementary Material</a> &nbsp;|&nbsp;
  <a href="https://drive.google.com/drive/folders/1qqPyKHtdiIVoiYWl3mZNGnJFXGzhRHTR?usp=drive_link">Pretrained Model</a>
</p>

## [🤗 Live Interactive Demo on Hugging Face](https://huggingface.co/spaces/yifeipet/AFP-GIC)

[![AFP-GIC interactive demo: original and reconstructed image comparison, bitrate controls, and downloadable results.](figs/Hugging_Face_Screen.png)](https://huggingface.co/spaces/yifeipet/AFP-GIC)

<p>Upload your own image. Choose an operating point. Compress, reconstruct, and compare. <strong>The demo runs on CPU by default.</strong></p>

<p align="center"><a href="https://huggingface.co/spaces/yifeipet/AFP-GIC"><strong>Try the live demo →</strong></a></p>

**One trained model, five bitrate operating points.** AFP-GIC transfers an image-adaptive fused prior from a frozen AdaCode model for very-low-bitrate generative image compression. Encoder-side prior guidance and decoder-side prior prediction support reconstruction without transmitting the fused prior itself.

This is the official **evaluation-only release**, providing the inference code and a pretrained checkpoint. Training code and datasets are not included.

## Overview

<p align="center">
  <img src="figs/overview.png" width="900" alt="AFP-GIC architecture: adaptive fused-prior guidance at the encoder and prior prediction at the decoder.">
</p>

**Figure 1.** Overview of AFP-GIC. Blue and red indicate encoding and decoding, respectively; snowflakes and flames denote frozen and trainable modules.

## Highlights

- **Single-model bitrate control:** select from five reported operating points without loading a different model for each rate.
- **Adaptive fused-prior transfer:** combine encoder-side prior guidance with decoder-side prediction, without transmitting the fused prior.
- **Decoder efficiency:** 18.1% lower decoder latency and 20.5% fewer inference parameters than DC-VIC under the paper's unified benchmark.

| Method | Inference parameters | Encoder latency | Decoder latency |
| :--- | ---: | ---: | ---: |
| DC-VIC | 151.7M | 61.61 ms | 98.27 ms |
| **AFP-GIC** | **120.6M** | 81.34 ms | **80.47 ms** |

Table 5 of the paper: RTX 4090, 100 DIV2K patches of 256 x 256 pixels. AFP-GIC saves 31.1M inference parameters; its encoder is slower in this benchmark. Parameter counts include the frozen prior component. These are system-level comparisons using the evaluated models, not architecture-only comparisons.

## Visual Comparisons

<p align="center">
  <img src="figs/kodak_comparison.png" width="850" alt="Original images and low-bitrate reconstructions from VVC Intra, MS-ILLM, CRDR, DC-VIC, and AFP-GIC on Kodak, including enlarged details.">
</p>

**Figure 5.** Low-bitrate visual comparisons on Kodak. Baselines are shown at their closest available released bitrates; each image is labeled with its actual bpp. Images and annotations are reproduced from the paper.

## Paper Metrics

The [metrics directory](metrics/) provides CSV results for all five AFP-GIC operating points on Kodak, CLIC2020, and DIV2K: [dataset summaries](metrics/afp_gic_operating_points.csv), [paper-rounded values](metrics/afp_gic_paper_values.csv), and 2,760 per-image records including PSNR, SSIM, MS-SSIM, SNR, LPIPS, DISTS, and NIQE. See the [data description](metrics/README.md) for the evaluation records and aggregation checks. FID is provided as a dataset-level metric.

## Installation

The release was tested with **Python 3.9**, **PyTorch 2.1.0**, and **torchvision 0.16.0**. Create a separate environment and install the pinned dependencies:

```bash
git clone https://github.com/yifeipet/AFP_GIC.git
cd AFP_GIC
conda create -n afp-gic python=3.9 -y
conda activate afp-gic
python -m pip install -r public_release/requirements.txt
```

For GPU evaluation, use a PyTorch build compatible with your GPU and driver. Follow the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) for the pinned version if a platform-specific build is needed. Do not replace the pinned versions with the latest releases when reproducing the paper.

## Pretrained Model

Download the released checkpoint from [Google Drive](https://drive.google.com/drive/folders/1qqPyKHtdiIVoiYWl3mZNGnJFXGzhRHTR?usp=drive_link) and place it at:

```text
checkpoint/afp_gic_release/model/afp_gic_release.pth.tar
```

The checkpoint already includes the frozen prior component; no separate AdaCode weight download is required.

## Evaluation

### Kodak

Obtain the [Kodak dataset](https://r0k.us/graphics/kodak/) and place its 24 original PNG images directly in `datasets/kodak/`.

From the repository root, evaluate one operating point:

```bash
python public_release/test.py -d cuda:0 --dataset kodak --qualities 0
```

Evaluate all five operating points:

```bash
python public_release/test.py -d cuda:0 --dataset kodak --qualities 0 1 2 3 4
```

| Quality index | 0 | 1 | 2 | 3 | 4 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Nominal target bpp | 0.050 | 0.075 | 0.100 | 0.125 | 0.150 |

Actual bitrates vary with image content. All five indices use the same checkpoint.

### Other Datasets

The entry point also accepts `clic2020_test` and `div2k_valid_hr`. Place the original PNG images in the corresponding directories:

```text
datasets/
|-- kodak/                              # Kodak PNG images
|-- CLIC/
|   `-- clic_test_images/               # CLIC2020 test PNG images
`-- DIV2K_valid_HR/
    `-- DIV2K_valid_HR/                 # DIV2K validation PNG images
```

```bash
python public_release/test.py -d cuda:0 --dataset clic2020_test --qualities 0 1 2 3 4
python public_release/test.py -d cuda:0 --dataset div2k_valid_hr --qualities 0 1 2 3 4
```

### Outputs

The runner saves reconstructed images, actual bitrates, per-image metrics, and summary files. To choose an output directory:

```bash
python public_release/test.py -d cuda:0 --dataset kodak --qualities 0 --results-root results/kodak_demo
```

For this command, outputs include:

```text
results/kodak_demo/
|-- summary_all.csv
|-- comparison_pivot.csv
`-- kodak/afp_gic_release/q0/
    |-- <image_name>.png
    |-- _bitrates.csv
    |-- per_image_metrics.csv
    |-- _metrics.json
    `-- summary.json
```

`per_image_metrics.csv` records in-loop PSNR, MS-SSIM, and LPIPS. The separate `_metrics.json` records the post-processing metric pass on saved reconstructions; do not assume the two evaluation paths are numerically interchangeable. See the paper and Supplementary Material for the reported metric protocols. This release is an evaluation entry point, not a one-command reproduction of every experiment in the paper.

Run `python public_release/test.py --help` for the available options. Metric libraries may download their pretrained weights on first use.

## Citation

If you find our work useful in your research, please cite our official [IEEE Access paper](https://ieeexplore.ieee.org/document/11712133):

```bibtex
@article{pei2026adaptive,
  title   = {Adaptive Fused Prior Transfer for Controllable Generative Image Compression},
  author  = {Pei, Yifei and Liu, Ying and Ling, Nam},
  journal = {IEEE Access},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3737467},
  url     = {https://ieeexplore.ieee.org/document/11712133}
}
```

## Acknowledgments and License

AFP-GIC builds on [DC-VIC](https://github.com/iwa-shi/DC_VIC) and [AdaCode](https://github.com/KAIST-VICLab/AdaCode), with supporting components from [BasicSR](https://github.com/XPixelGroup/BasicSR) and [CompressAI](https://github.com/InterDigitalInc/CompressAI). We thank their authors for making these resources available.

Original AFP-GIC additions are provided for research and evaluation use. Third-party components retain their respective licenses; no single permissive license applies uniformly to this repository. Please consult [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before reuse or redistribution.

For questions about this release, please open a [GitHub issue](https://github.com/yifeipet/AFP_GIC/issues).

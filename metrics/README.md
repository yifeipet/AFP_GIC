# AFP-GIC paper metrics

These CSV files provide the AFP-GIC results used in Figure 3, Appendix A,
the FID tables in Appendix B, and the SNR/MS-SSIM point tables in the
Supplementary Material. They cover the released model's five operating points.

## Files

| File | Contents | Rows |
| --- | --- | ---: |
| `afp_gic_operating_points.csv` | Dataset means at original numerical precision, including LPIPS and dataset-level FID | 15 |
| `afp_gic_paper_values.csv` | The same results rounded as printed in the paper and supplementary tables | 15 |
| `afp_gic_per_image_kodak.csv` | 24 images at five operating points | 120 |
| `afp_gic_per_image_clic2020.csv` | 428 images at five operating points | 2140 |
| `afp_gic_per_image_div2k.csv` | 100 images at five operating points | 500 |
| `aggregation_check.csv` | Checks of per-image means against paper-source means | 120 |

## Columns

In each per-image CSV, `dataset` is the first column, followed by `image_name`,
which preserves the original filename, including its extension. The metric
columns follow the main paper: bpp, PSNR, SSIM, LPIPS, DISTS, NIQE; the
supplementary metrics SNR and MS-SSIM follow. Rows are sorted by image
name, then by `quality_index`, so the five operating points for each image
appear together. For example, the five rows for `kodim01.png` precede those
for `kodim02.png` in the Kodak file. Image names are unique within each dataset;
use `(dataset, image_name, quality_index)` to identify a record.

- `quality_index`: zero-based operating-point index, from 0 to 4.
- `nominal_bpp`: nominal operating-point target. Actual bitrates vary with image content.
- `bpp`: actual bits per pixel for each image in per-image files; the unweighted image mean in summary files.
- `image_name`, `width`, `height`: source image identifier and dimensions.
- `num_images`: number of images contributing to each dataset mean.
- `PSNR_dB`, `SNR_dB`: PSNR and SNR in decibels; higher is better.
- `SSIM`, `MS_SSIM`: single-scale SSIM and MS-SSIM; higher is better.
- `LPIPS`, `DISTS`, `NIQE`: lower is better.
- `FID`: dataset-level FID under the paper's patch protocol; lower is better.

FID is not a per-image metric. Kodak FID is blank because it is not reported
in the paper. No blank entry should be interpreted as zero.

## Evaluation records and verification

Per-image PSNR, SSIM, and SNR come from the saved RGB reconstruction evaluation.
Per-image LPIPS was recovered from the same saved RGB reconstructions using
the original evaluation procedure: LPIPS v0.1 with AlexNet, RGB inputs scaled
to [-1, 1], and full-image evaluation with batch size one.
MS-SSIM, DISTS, and NIQE come from the archived per-image evaluation records;
their means agree with the paper-source results within numerical tolerance.
No values were shifted or rescaled to enforce agreement. PSNR mean agreement
is checked within 1e-5 dB; the other exported per-image quantities within 1e-6.
All summary quantities also match the corresponding printed table values
at the precision used in the paper.

The recovered LPIPS means are checked against the original paper-source means
within 1e-6 and must also match every printed LPIPS value to five decimal places.
The operating-point summaries retain the original paper values; numerical
differences below the reported precision can arise across evaluation environments.

The CSV files contain metric records, not reconstruction images or datasets.
Use the paper and Supplementary Material for the metric definitions and
evaluation protocols. These files report AFP-GIC only, not baseline results.

## Example

```python
import pandas as pd

points = pd.read_csv("metrics/afp_gic_operating_points.csv")
kodak = pd.read_csv("metrics/afp_gic_per_image_kodak.csv")
print(points[points.dataset == "kodak"])
print(kodak.groupby("quality_index")[["bpp", "PSNR_dB", "SSIM", "NIQE"]].mean())
```

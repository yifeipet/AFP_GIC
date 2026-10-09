# AFP-GIC Training

The training implementation is isolated in [`training/`](training/) and does not
replace the existing inference runtime. Run commands from the repository root.

## Three Stages, Seven Steps

| Stage | Step | Purpose | Optimizer iterations |
| --- | --- | --- | ---: |
| I | 1 | High-rate warmup, without dual conditioning | 500,000 |
| I | 2 | Dual-control rate-distortion training, without GAN | 500,000 |
| I | 3 | Dual-control adversarial training | 500,000 |
| II | 1 | Prepare 2,000 validation crops (256 x 256) | No training |
| II | 2 | Binary-search rate controls for each prior control | No training |
| II | 3 | Evaluate candidates and select five control pairs | No training |
| III | 1 | Fine-tune on the five selected pairs | 500,000 |

Stage I has three steps; Stage II has three steps; Stage III has one step.
Each training step has its own iteration counter. Total optimizer iterations:
2,000,000. Stage II performs evaluation and selection, not gradient updates.

**Prior cosine is disabled from iteration zero in every supplied configuration:**

```yaml
prior_cosine_loss:
  enabled: false
  loss_weight: 0.0
```

The prior-consistency MSE remains enabled (weights 0.1, 0.006, 1.0, and 1.0
for the four training segments). Use the published pretrained model when
comparing directly with the paper's reported numbers; retraining results can
vary with data, initialization and the execution environment.

## Installation

Use Linux or WSL and a CUDA-capable GPU. The original experiments used an
RTX 4090 (24 GB). Memory requirements depend on batch size and training step.

```bash
conda create -n afp-gic-training python=3.9 -y
conda activate afp-gic-training
python -m pip install -r training/requirements.txt
```

The pinned stack uses PyTorch 2.1.0, torchvision 0.16.0 and CompressAI 1.2.4.
Install a matching CUDA build for your driver. Weights for LPIPS and FID may be
downloaded on first use. Training does not require a Weights & Biases account;
the logging integration is off by default.

## Data and Pretrained Prior

Prepare the OpenImages training subsets `train_0` through `train_9`, its separate
`validation` directory, and Kodak originals. Obtain datasets from their original
providers; images and model weights are not included in this source package.

Download the **AdaCode Stage II checkpoint `AdaCode_S2_model_g.pth`** from
the [official AdaCode pretrained-model release](https://github.com/kechunl/AdaCode/releases/tag/v0-pretrain_models).
Use the original filename when downloading a mirrored copy from our GitHub Releases.
AdaCode weights remain subject to their upstream license and attribution requirements.
This is the frozen prior used to initialize training, not the released AFP-GIC
inference checkpoint. Do not substitute a random model or a VQGAN checkpoint.

Edit **`training/settings.yaml`** with your data and checkpoint locations:

```yaml
openimage_root: /path/to/openimage
kodak_root: /path/to/kodak
adacode_checkpoint: /path/to/AdaCode_S2_model_g.pth
output_root: outputs
subsets: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
batch_size: 6
num_workers: 8
selection_batch_size: 8
selection_images: 2000
selection_seed: 0
```

Relative paths in settings are resolved from `training/`, not the shell's working
directory. Reduce workers or batch size when needed; changing data or batch size
changes the training protocol. Keep an even batch size of at least 2 for GAN steps.

```text
openimage/
  train_0/*.jpg
  ...
  train_9/*.jpg
  validation/*.jpg
kodak/
  kodim01.png
  ...
  kodim24.png
```

## Stage I: Three Training Steps

Run sequentially, waiting for each step to finish:

```bash
python training/run.py stage1-step1 --device cuda:0
python training/run.py stage1-step2 --device cuda:0
python training/run.py stage1-step3 --device cuda:0
```

Steps 2 and 3 initialize from the previous step's 500K model. Step 1 starts a new
compression model with the pretrained frozen AdaCode component. Training-state
checkpoints are saved for resumption.

## Stage II: Three Selection Steps

```bash
python training/run.py stage2-step1
python training/run.py stage2-step2 --device cuda:0
python training/run.py stage2-step3 --device cuda:0
```

Step 1 takes random 256 x 256 crops from the **validation subset**, using seed 0.
It uses images only, without VQGAN token files. Step 2 searches `beta_rate` in
[0, 3] at `beta_prior = 0.25, 0.50, ..., 3.50`, for target mean rates
0.050, 0.075, 0.100, 0.125, 0.150 bpp. The source code calls `beta_prior`
`beta_vq` for compatibility with the existing implementation.

Step 3 selects the highest `2 * PSNR - FID` score among rate-matched candidates
for each target. FID is part of the normal selection, not silently skipped.
The search uses forward-estimated rates on the validation crops; final codec
bitstreams should be evaluated separately. These are dataset-average operating
points, not guaranteed per-image rates.

Selection output:

```text
training/outputs/beta_selection/beta_selection_results.csv
```

If any target has no eligible candidate, the runner reports incomplete selection.
Do not start Stage III with fewer than five pairs. Inspect the search CSVs and
the candidate tolerance before changing the search protocol.

## Stage III: Selected-Pair Fine-Tuning

```bash
python training/run.py stage3 --device cuda:0
```

The runner reads and validates all five rows of the new selection CSV, then
writes `training/outputs/configs/stage3.yaml`. It loads the Stage I Step 3 model,
discriminator and optimizer states and resets the learning-rate schedule for
500K further iterations. The released model's fixed pairs are not substituted
for this search. Keep the generated config with the new model when evaluating it.
Training checkpoints are not automatically drop-in replacements for the existing
packaged inference release; the new model must use its own selected-pair config.

## Checks and Resume

Generate a command/config without starting training:

```bash
python training/run.py stage1-step1 --dry-run
python -m unittest discover -s training/tests -v
```

Resume the same step from an actually saved iteration:

```bash
python training/run.py stage1-step2 --resume 100000 --device cuda:0
```

Keep `comp_model`, `training_state`, and (for GAN steps) `discriminator` files
from the same iteration together. The runner refuses to start a fresh training
step in a nonempty experiment directory. For a separate experiment, set a new
`output_root`. Configuration changes during resume must be deliberate; retain
the original settings and generated configs with each run.

## Source and Attribution

The runtime retains the local AFP-GIC training implementation and its upstream
DC-VIC, AdaCode, latent-diffusion and taming-transformers components. See
[`training/THIRD_PARTY_NOTICES.md`](training/THIRD_PARTY_NOTICES.md). Existing
component licenses continue to apply. No datasets, pretrained weights, tokens,
private logs or trained checkpoints are part of this source release.

"""Run the seven ordered steps of AFP-GIC training from any working directory."""
import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parent
STEPS = ("stage1-step1", "stage1-step2", "stage1-step3",
         "stage2-step1", "stage2-step2", "stage2-step3", "stage3")
TARGETS = [0.05, 0.075, 0.1, 0.125, 0.15]
BETAS = [i / 4 for i in range(1, 15)]


def resolve(value):
    return str((ROOT / Path(value).expanduser()).resolve())


def selected_pairs(path):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 5:
        raise ValueError("Stage II must produce exactly five selected pairs.")
    rows.sort(key=lambda row: float(row["target_rate"]))
    rates, priors = [], []
    for target, row in zip(TARGETS, rows):
        if not math.isclose(float(row["target_rate"]), target, abs_tol=1e-8):
            raise ValueError("Unexpected or duplicate target rate in selection CSV.")
        rate, prior = float(row["selected_beta_rate"]), float(row["selected_beta_vq"])
        if not (math.isfinite(rate) and 0 <= rate <= 3 and math.isfinite(prior) and 0 <= prior <= 3.5):
            raise ValueError("Invalid beta pair in selection CSV.")
        rates.append(rate)
        priors.append(prior)
    return rates, priors


def write_config(name, settings, outputs):
    template = "stage3_template" if name == "stage3" else name
    config = yaml.safe_load((ROOT / "configs" / (template + ".yaml")).read_text())
    config["ckpt_root"] = str(outputs / "checkpoint")
    config["debug_dir"] = str(outputs / "debug")
    config["wandb_root"] = str(outputs)
    config["dataset"]["train_dataset"].update(root_dir=resolve(settings["openimage_root"]), subset_list=settings["subsets"])
    config["dataset"]["eval_dataset"]["root_dir"] = resolve(settings["kodak_root"])
    config["dataset"]["batch_size"] = settings["batch_size"]
    config["subnet"]["adacode_prior"]["ckpt_path"] = resolve(settings["adacode_checkpoint"])
    if name == "stage3":
        rates, priors = selected_pairs(outputs / "beta_selection/beta_selection_results.csv")
        config["model"].update(selected_beta_rate=rates, selected_beta_vq=priors)
    cosine = config["loss"]["prior_cosine_loss"]
    if cosine["enabled"] or cosine["loss_weight"] != 0:
        raise ValueError("This training recipe requires prior cosine disabled from iteration zero.")
    if not config["loss"]["prior_mse_loss"].get("enabled", True):
        raise ValueError("The main model must retain prior MSE.")
    path = outputs / "configs" / (name + ".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def require(path, label):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--settings", default=str(ROOT / "settings.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=int, default=0, help="Iteration of a saved checkpoint in this same step")
    parser.add_argument("--dry-run", action="store_true", help="Generate config and print command without running")
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.settings).read_text())
    if settings["batch_size"] < 2 or settings["batch_size"] % 2:
        parser.error("Use an even training batch_size of at least 2 for the GAN workflow.")
    if settings["selection_images"] < 2 or settings["selection_batch_size"] < 1 or settings["num_workers"] < 0:
        parser.error("Invalid image count, selection batch size or worker count.")
    if not settings["subsets"]:
        parser.error("At least one training subset is required.")
    outputs = Path(resolve(settings["output_root"]))
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "checkpoint").mkdir(exist_ok=True)
    runtime = ROOT / "runtime"
    selection = outputs / f"validation/img_only/crop_256_{settings['selection_images']}_seed_{settings['selection_seed']}"
    stage1_model = outputs / "checkpoint/stage1_step3/model/comp_model_iter500K.pth.tar"
    if args.resume < 0 or (args.resume and args.step.startswith("stage2")):
        parser.error("--resume must be nonnegative and is only for training steps.")
    if args.step.startswith("stage1") or args.step == "stage3":
        name = args.step.replace("-", "_")
        config_path = write_config(name, settings, outputs)
        config = yaml.safe_load(config_path.read_text())
        job = outputs / "checkpoint" / name
        if not args.dry_run:
            require(resolve(settings["adacode_checkpoint"]), "pretrained AdaCode checkpoint")
            require(resolve(settings["kodak_root"]), "Kodak directory")
            for subset in settings["subsets"]:
                require(Path(resolve(settings["openimage_root"])) / f"train_{subset}", "training subset")
            if job.exists() and any(job.iterdir()) and not args.resume:
                raise FileExistsError(f"Refusing to overwrite {job}; use --resume or another output_root.")
            if args.resume:
                if args.resume >= config["total_iter"]:
                    raise ValueError("Resume iteration must precede the configured final iteration.")
                itr = f"{args.resume // 1000}K" if args.resume % 1000 == 0 else str(args.resume)
                labels = ["comp_model", "training_state"]
                if name in ("stage1_step3", "stage3"):
                    labels.append("discriminator")
                for label in labels:
                    require(job / "model" / f"{label}_iter{itr}.pth.tar", "resume checkpoint")
            elif "load_checkpoint" in config:
                previous = outputs / "checkpoint" / config["load_checkpoint"]["exp"] / "model"
                require(previous / "comp_model_iter500K.pth.tar", "previous training step")
                if args.step == "stage3":
                    require(previous / "training_state_iter500K.pth.tar", "Stage I optimizer state")
                    require(previous / "discriminator_iter500K.pth.tar", "Stage I discriminator")
        command = [sys.executable, str(runtime / "scripts/train.py"), str(config_path), "-d", args.device,
                   "-nw", str(settings["num_workers"])]
        if args.resume:
            command += ["--start_iter", str(args.resume)]
    elif args.step == "stage2-step1":
        # Pass validation itself: the upstream resolver prefers train_0 if given the parent.
        validation = Path(resolve(settings["openimage_root"])) / "validation"
        if not args.dry_run:
            require(validation, "OpenImages validation directory")
            if selection.exists() and any(selection.iterdir()):
                raise FileExistsError(f"Validation destination is not empty: {selection}")
        command = [sys.executable, str(runtime / "scripts/build_openimage_val_dataset.py"),
                   "--openimage_root", str(validation), "--save_root", str(outputs / "validation"),
                   "--seed", str(settings["selection_seed"]), "--num_img", str(settings["selection_images"])]
    else:
        config_path = write_config("stage1_step3", settings, outputs)
        if not args.dry_run:
            require(stage1_model, "Stage I Step 3 model")
            require(selection, "selection images")
            if len(list(selection.glob("*.png"))) != settings["selection_images"]:
                raise ValueError("Selection image count does not match settings.")
            if args.step == "stage2-step3":
                for beta in BETAS:
                    for target in TARGETS:
                        require(outputs / "bin_search" / f"result_beta_vq_{beta:.2f}_target_rate_{target:.3f}.csv", "binary search result")
        script = "binary_rate_search.py" if args.step == "stage2-step2" else "beta_selection.py"
        command = [sys.executable, str(runtime / "scripts" / script), str(config_path),
                   "--model_path", str(stage1_model), "--dataset_root", str(selection),
                   "--batch_size", str(settings["selection_batch_size"]), "--num_workers", str(settings["num_workers"]),
                   "-d", args.device, "--beta_vq", *map(str, BETAS), "--target_rate", *map(str, TARGETS)]
        if args.step == "stage2-step2":
            command += ["--save_dir", str(outputs / "bin_search"), "--max_beta_rate", "3.0"]
        else:
            command += ["--save_dir", str(outputs / "beta_selection"), "--search_dir", str(outputs / "bin_search"), "--alpha", "2.0"]
    print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(runtime)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.step == "stage2-step1" and len(list(selection.glob("*.png"))) != settings["selection_images"]:
        raise ValueError("Not enough eligible images: Stage II dataset is incomplete.")
    if args.step == "stage2-step3":
        selected_pairs(outputs / "beta_selection/beta_selection_results.csv")
        print("All five selected beta pairs validated. Stage III is ready.")


if __name__ == "__main__":
    main()

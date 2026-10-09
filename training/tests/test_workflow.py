import csv
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("workflow", ROOT / "run.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def test_all_seven_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp) / "outputs"
            settings = yaml.safe_load((ROOT / "settings.yaml").read_text())
            settings["output_root"] = str(outputs)
            settings_path = Path(temp) / "settings.yaml"
            settings_path.write_text(yaml.safe_dump(settings))
            csv_path = outputs / "beta_selection/beta_selection_results.csv"
            csv_path.parent.mkdir(parents=True)
            with csv_path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["target_rate", "selected_beta_rate", "selected_beta_vq"])
                for target in workflow.TARGETS:
                    writer.writerow([target, 1.0, 2.0])
            for step in workflow.STEPS:
                result = subprocess.run([sys.executable, str(ROOT / "run.py"), step, "--settings", str(settings_path), "--dry-run"], cwd=temp, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                if step == "stage2-step1":
                    self.assertIn("openimage/validation", result.stdout)
                    self.assertNotIn("--vqgan_type", result.stdout)
                if step == "stage2-step3":
                    self.assertIn("--alpha 2.0", result.stdout)
                    self.assertNotIn("--skip_fid", result.stdout)

    def test_all_configs_disable_cosine_and_keep_mse(self):
        for path in (ROOT / "configs").glob("*.yaml"):
            cfg = yaml.safe_load(path.read_text())
            self.assertEqual(cfg["loss"]["prior_cosine_loss"], {"enabled": False, "loss_weight": 0.0})
            self.assertTrue(cfg["loss"]["prior_mse_loss"].get("enabled", True))
            self.assertGreater(cfg["loss"]["prior_mse_loss"]["loss_weight"], 0)
            self.assertEqual(cfg["total_iter"], 500000)
            self.assertNotIn("_base_", cfg)
            self.assertTrue(cfg["keep_training_state"])

    def test_stage_links(self):
        for name, previous in [("stage1_step2", "stage1_step1"), ("stage1_step3", "stage1_step2"), ("stage3_template", "stage1_step3")]:
            cfg = yaml.safe_load((ROOT / "configs" / (name + ".yaml")).read_text())
            self.assertEqual(cfg["load_checkpoint"]["exp"], previous)
            self.assertEqual(cfg["load_checkpoint"]["iter"], 500000)

    def test_selection_and_config(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp)
            path = outputs / "beta_selection/beta_selection_results.csv"
            path.parent.mkdir()
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["target_rate", "selected_beta_rate", "selected_beta_vq"])
                for target in reversed(workflow.TARGETS):
                    writer.writerow([target, 1.0, 2.0])
            rates, priors = workflow.selected_pairs(path)
            self.assertEqual(rates, [1.0] * 5)
            settings = yaml.safe_load((ROOT / "settings.yaml").read_text())
            cfg = yaml.safe_load(workflow.write_config("stage3", settings, outputs).read_text())
            self.assertEqual(cfg["model"]["selected_beta_vq"], priors)
            path.write_text("target_rate,selected_beta_rate,selected_beta_vq\n0.05,1,2\n")
            with self.assertRaises(ValueError):
                workflow.selected_pairs(path)
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["target_rate", "selected_beta_rate", "selected_beta_vq"])
                for target in workflow.TARGETS:
                    writer.writerow([target, "nan", 2])
            with self.assertRaises(ValueError):
                workflow.selected_pairs(path)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for rl/evaluate.py evaluation pipeline and integrity checks."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.dqn import DoubleDQNAgent
from rl.evaluate import _verify_holdout_separation, evaluate_held_out


class TestEvaluatePipeline(unittest.TestCase):
    def test_holdout_separation_rejects_overlap_and_duplicates(self):
        with self.assertRaises(RuntimeError):
            _verify_holdout_separation([(10.0, 50.0)], [(10.0, 50.0)])
        with self.assertRaises(RuntimeError):
            _verify_holdout_separation([(10.0, 50.0)], [(12.0, 55.0), (12.0, 55.0)])

    def test_evaluate_held_out_reproducible_no_overwrite_and_identical_starts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            model_path = tmp / "model.pt"
            agent = DoubleDQNAgent(state_dim=4, action_dim=5, seed=11, device="cpu")
            agent.save(model_path, overwrite=False, metadata={"test": True})

            logs_dir = tmp / "logs"
            figures_dir = tmp / "figures"

            summary1 = evaluate_held_out(
                model_path=model_path,
                max_steps=12,
                eval_seed=100,
                logs_dir=logs_dir,
                figures_dir=figures_dir,
                training_metrics_csv=None,
                generate_plots=True,
                max_loads=6,
            )

            self.assertEqual(summary1["status"], "completed")
            self.assertTrue(summary1["integrity"]["weights_unchanged"])
            self.assertEqual(summary1["integrity"]["holdout_size"], 350)
            self.assertEqual(summary1["integrity"]["train_holdout_overlap"], 0)
            self.assertEqual(summary1["integrity"]["evaluated_loads_count"], 6)
            self.assertEqual(len(summary1["plots"]), 5)
            for p in summary1["plots"]:
                self.assertTrue(Path(p).exists())
            self.assertTrue(Path(summary1["csv_path"]).exists())
            self.assertTrue(Path(summary1["report_path"]).exists())
            self.assertTrue(Path(summary1["summary_json_path"]).exists())

            with Path(summary1["csv_path"]).open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 18)  # 6 loads x 3 approaches

            by_app = {"double_dqn": [], "random": [], "grid_search": []}
            for row in rows:
                by_app[row["approach"]].append(row)
            for i in range(6):
                key_dqn = (
                    by_app["double_dqn"][i]["RL"],
                    by_app["double_dqn"][i]["XL"],
                    by_app["double_dqn"][i]["initial_V1"],
                    by_app["double_dqn"][i]["initial_V2"],
                )
                key_rnd = (
                    by_app["random"][i]["RL"],
                    by_app["random"][i]["XL"],
                    by_app["random"][i]["initial_V1"],
                    by_app["random"][i]["initial_V2"],
                )
                key_grd = (
                    by_app["grid_search"][i]["RL"],
                    by_app["grid_search"][i]["XL"],
                    by_app["grid_search"][i]["initial_V1"],
                    by_app["grid_search"][i]["initial_V2"],
                )
                self.assertEqual(key_dqn, key_rnd)
                self.assertEqual(key_dqn, key_grd)

            # Second run must not overwrite first outputs and must reproduce metrics identically
            summary2 = evaluate_held_out(
                model_path=model_path,
                max_steps=12,
                eval_seed=100,
                logs_dir=logs_dir,
                figures_dir=figures_dir,
                training_metrics_csv=None,
                generate_plots=False,
                max_loads=6,
            )
            self.assertNotEqual(summary1["csv_path"], summary2["csv_path"])
            self.assertNotEqual(summary1["summary_json_path"], summary2["summary_json_path"])
            timing_keys = {"exec_time_ms_mean", "exec_time_ms_total", "inference_time_ms_mean"}
            for app_name in ("double_dqn", "random"):
                d1 = {k: v for k, v in summary1["approaches"][app_name].items() if k not in timing_keys}
                d2 = {k: v for k, v in summary2["approaches"][app_name].items() if k not in timing_keys}
                self.assertEqual(d1, d2)
                self.assertGreater(summary1["approaches"][app_name]["exec_time_ms_mean"], 0.0)
                self.assertGreater(summary2["approaches"][app_name]["exec_time_ms_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()


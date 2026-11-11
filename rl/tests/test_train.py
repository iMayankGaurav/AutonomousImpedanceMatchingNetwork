"""Unit tests for rl/train.py pipeline (configuration, logging, interrupt safety, plots)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.dqn import DoubleDQNAgent
from rl.train import (
    build_load_catalog,
    get_non_overwriting_path,
    train_double_dqn,
)


class TestTrainPipeline(unittest.TestCase):
    def test_load_catalog_splits_matchable_and_tracks_unmatchable(self):
        catalog = build_load_catalog(
            vswr_threshold=1.5,
            rl_step=1.0,
            xl_step=1.0,
            full_scan_step=10.0,
            holdout_fraction=0.2,
            split_seed=42,
        )
        train_loads = catalog["train_loads"]
        holdout_loads = catalog["holdout_loads"]
        self.assertGreater(len(train_loads), 100)
        self.assertGreater(len(holdout_loads), 20)
        self.assertEqual(len(set(train_loads) & set(holdout_loads)), 0)
        self.assertGreater(catalog["unmatchable_envelope_count"], 0)
        self.assertGreater(catalog["full_range_scan"]["unmatchable_vswr_ge_2_count"], 0)

    def test_non_overwriting_path_increments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "model.pt"
            self.assertEqual(get_non_overwriting_path(base), base)
            base.write_text("existing", encoding="utf-8")
            v2 = get_non_overwriting_path(base)
            self.assertEqual(v2, Path(tmpdir) / "model_v2.pt")
            v2.write_text("existing2", encoding="utf-8")
            self.assertEqual(get_non_overwriting_path(base), Path(tmpdir) / "model_v3.pt")

    def test_training_pipeline_periodic_checkpoints_plots_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            existing_model = tmp / "model.pt"
            existing_model.write_bytes(b"existing-model-sentinel")

            summary = train_double_dqn(
                num_episodes=12,
                max_steps=15,
                lr=0.001,
                gamma=0.99,
                buffer_capacity=1000,
                batch_size=8,
                eps_decay_steps=60,
                warmup_steps=10,
                target_update_freq=5,
                checkpoint_interval=5,
                seed=7,
                final_model_path=existing_model,
                checkpoints_dir=tmp / "checkpoints",
                logs_dir=tmp / "logs",
                figures_dir=tmp / "figures",
                run_name="test_run",
                log_interval=0,
                generate_plots=True,
            )
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["completed_episodes"], 12)
            self.assertEqual(summary["nan_or_inf_events"], 0)

            # Existing model sentinel must be untouched; new model saved under _v2
            self.assertEqual(existing_model.read_bytes(), b"existing-model-sentinel")
            saved_model = Path(summary["final_model_path"])
            self.assertEqual(saved_model.name, "model_v2.pt")
            self.assertTrue(saved_model.exists())

            # Periodic checkpoints at ep 5 and ep 10
            self.assertEqual(len(summary["periodic_checkpoints"]), 2)
            for ckpt_p in summary["periodic_checkpoints"]:
                self.assertTrue(Path(ckpt_p).exists())

            # Logs, unmatchable report, and 7 plots (6 individual + overview)
            self.assertTrue(Path(summary["metrics_csv_path"]).exists())
            self.assertTrue(Path(summary["unmatchable_report_path"]).exists())
            self.assertTrue(Path(summary["summary_json_path"]).exists())
            self.assertEqual(len(summary["plots"]), 7)
            for plot_p in summary["plots"]:
                self.assertTrue(Path(plot_p).exists())

            loaded = DoubleDQNAgent.load(saved_model)
            self.assertIn(loaded.act([0.1, -0.2, 1.65, 1.65]), {0, 1, 2, 3, 4})

    def test_keyboard_interrupt_saves_partial_progress_safely(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            call_count = 0
            orig_select = DoubleDQNAgent.select_action

            def interrupt_after_25_steps(self_agent, state, epsilon=0.0):
                nonlocal call_count
                call_count += 1
                if call_count > 25:
                    raise KeyboardInterrupt()
                return orig_select(self_agent, state, epsilon=epsilon)

            with mock.patch.object(
                DoubleDQNAgent, "select_action", autospec=True, side_effect=interrupt_after_25_steps
            ):
                summary = train_double_dqn(
                    num_episodes=10,
                    max_steps=10,
                    batch_size=4,
                    warmup_steps=4,
                    checkpoint_interval=5,
                    seed=11,
                    final_model_path=tmp / "model.pt",
                    checkpoints_dir=tmp / "checkpoints",
                    logs_dir=tmp / "logs",
                    figures_dir=tmp / "figures",
                    run_name="int_run",
                    log_interval=0,
                    generate_plots=True,
                )

            self.assertEqual(summary["status"], "interrupted")
            self.assertGreaterEqual(summary["completed_episodes"], 2)
            self.assertFalse((tmp / "model.pt").exists())  # Final model target left untouched
            self.assertTrue(Path(summary["final_model_path"]).exists())  # Interrupted ckpt saved
            self.assertTrue(Path(summary["metrics_csv_path"]).exists())
            with open(summary["summary_json_path"], encoding="utf-8") as f:
                saved_summary = json.load(f)
            self.assertEqual(saved_summary["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()

import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reactoccflow.simulation.branch_collection import (
    SCENARIO_FAMILIES,
    BranchCollectionConfig,
    BranchCollector,
    BranchResult,
    ResetState,
    ScenarioOrigin,
    deterministic_origin_split,
)


class FakeBackend:
    def __init__(self, *, mismatch_action=None, failure_action=None):
        self.reset_calls = []
        self.execute_calls = []
        self.mismatch_action = mismatch_action
        self.failure_action = failure_action

    def reset(self, scenario, npc_seed):
        self.reset_calls.append((scenario.origin_id, npc_seed))
        history = {"timestamps": np.array([0.0, 0.5]), "actor_history": np.zeros((1, 2, 4))}
        fingerprint = f"{scenario.origin_id}:{npc_seed}"
        return ResetState(
            initial_state_fingerprint=fingerprint + (":mismatch" if self.mismatch_action == scenario.action_id else ""),
            common_history_fingerprint=fingerprint,
            public_history=history,
            metadata={"coordinate_frame": "sim_ego", "semantic_taxonomy": "carla", "npc_trajectory_mode": "responsive"},
        )

    def execute(self, candidate_reference, *, responsive_npc):
        self.execute_calls.append((candidate_reference["action_id"], responsive_npc))
        if candidate_reference["action_id"] == self.failure_action:
            raise TimeoutError("fake timeout")
        return BranchResult(
            ego_trajectory=np.zeros((2, 4), dtype=np.float32),
            actor_trajectories=np.zeros((1, 2, 4), dtype=np.float32),
            occupancy_labels=np.zeros((2, 2, 2), dtype=np.uint8),
            occupancy_masks=np.ones((2, 2, 2), dtype=bool),
            collision_events=np.zeros(2, dtype=np.uint8),
            tracking_error=np.zeros(2, dtype=np.float32),
            timestamps=np.array([0.0, 0.5]),
            execution_status="success",
            metadata={"npc_control_mode": "responsive"},
        )


class BranchCollectionTests(unittest.TestCase):
    def _config(self, root, *, actions=("a0", "a1"), seeds=(3, 7)):
        origins = tuple(ScenarioOrigin(f"o{i}", SCENARIO_FAMILIES[i], i) for i in range(2))
        return BranchCollectionConfig(
            output_root=root,
            origins=origins,
            action_ids=actions,
            npc_seeds=seeds,
            settings={"duration_s": 1.0, "step_s": 0.5, "candidate_references": {a: {"points": [[0.0, 0.0], [1.0, 0.0]]} for a in actions}},
        )

    def test_every_branch_resets_and_common_history_is_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeBackend()
            result = BranchCollector(backend, self._config(Path(tmp))).collect()
            self.assertEqual(result.expected_branches, 8)
            self.assertEqual(len(backend.reset_calls), 8)
            self.assertEqual(len(backend.execute_calls), 8)
            self.assertEqual(result.successful_branches, 8)
            self.assertTrue(result.complete)
            statuses = [json.loads(p.read_text()) for p in (p for p in Path(tmp).glob("*.json") if not p.name.startswith("_"))]
            self.assertTrue(all(s["common_history_fingerprint"] for s in statuses))

    def test_origin_grouping_and_deterministic_split(self):
        a = deterministic_origin_split("same-origin")
        self.assertEqual(a, deterministic_origin_split("same-origin"))
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), actions=("a0",), seeds=(1,))
            report = BranchCollector(FakeBackend(), cfg).collect()
            statuses = [json.loads(p.read_text()) for p in (p for p in Path(tmp).glob("*.json") if not p.name.startswith("_"))]
            by_origin = {}
            for status in statuses:
                by_origin.setdefault(status["origin_id"], set()).add(status["split"])
            self.assertTrue(all(len(v) == 1 for v in by_origin.values()))
            self.assertEqual(report.split_counts["train"] + report.split_counts["tune"] + report.split_counts["calibration"] + report.split_counts["evaluation"], 2)

    def test_future_observation_and_nonfinite_are_rejected(self):
        class BadBackend(FakeBackend):
            def reset(self, scenario, npc_seed):
                state = super().reset(scenario, npc_seed)
                state.public_history["future_trajectory"] = np.zeros(1)
                return state
        with tempfile.TemporaryDirectory() as tmp:
            report = BranchCollector(BadBackend(), self._config(Path(tmp), actions=("a0",), seeds=(1,))).collect()
            self.assertEqual(report.successful_branches, 0)
            status = next(json.loads(p.read_text()) for p in (p for p in Path(tmp).glob("*.json") if not p.name.startswith("_")))
            self.assertEqual(status["status"], "failed")
            self.assertIn("future", status["reason"])

    def test_failure_is_retained_and_resume_rejects_changed_settings_or_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self._config(root, actions=("a0", "a1"), seeds=(1,))
            first = BranchCollector(FakeBackend(failure_action="a1"), cfg).collect()
            self.assertEqual((first.successful_branches, first.failed_branches), (2, 2))
            self.assertTrue(first.complete)
            changed = self._config(root, actions=("a0", "different"), seeds=(1,))
            with self.assertRaises(ValueError):
                BranchCollector(FakeBackend(), changed).collect(resume=True)
            npz = next(root.glob("*.npz"))
            npz.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                BranchCollector(FakeBackend(), cfg).collect(resume=True)

    def test_declared_prerecorded_npc_is_rejected(self):
        class FrozenBackend(FakeBackend):
            def execute(self, candidate_reference, *, responsive_npc):
                return BranchResult(
                    ego_trajectory=np.zeros((1, 2)), actor_trajectories=np.zeros((1, 1, 2)),
                    occupancy_labels=np.zeros((1, 1, 1)), occupancy_masks=np.ones((1, 1, 1), bool),
                    collision_events=np.zeros(1), tracking_error=np.zeros(1), timestamps=np.zeros(1),
                    execution_status="success", metadata={"npc_trajectory_mode": "frozen"},
                )
        with tempfile.TemporaryDirectory() as tmp:
            report = BranchCollector(FrozenBackend(), self._config(Path(tmp), actions=("a0",), seeds=(1,))).collect()
            self.assertEqual(report.successful_branches, 0)
            status = next(json.loads(p.read_text()) for p in (p for p in Path(tmp).glob("*.json") if not p.name.startswith("_")))
            self.assertIn("frozen", status["reason"])


if __name__ == "__main__":
    unittest.main()

# T8 blocking-review regressions

class T8ProtocolRegressionTests(unittest.TestCase):
    def _config(self, root, *, actions=("a0", "a1"), seeds=(1,)):
        origins = (ScenarioOrigin("origin", SCENARIO_FAMILIES[0], 0),)
        return BranchCollectionConfig(
            output_root=root,
            origins=origins,
            action_ids=actions,
            npc_seeds=seeds,
            settings={
                "duration_s": 1.0,
                "step_s": 0.5,
                "candidate_references": {a: {"points": [[0.0, 0.0], [1.0, 0.0]]} for a in actions},
            },
        )

    def test_resume_rejects_tampered_branch_identity_and_structural_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self._config(root, actions=("a0",), seeds=(1,))
            BranchCollector(FakeBackend(), cfg).collect()
            status_path = next(p for p in root.glob("*.json") if not p.name.startswith("_"))
            status = json.loads(status_path.read_text())
            status["action_id"] = "tampered"
            status_path.write_text(json.dumps(status))
            with self.assertRaises(ValueError):
                BranchCollector(FakeBackend(), cfg).collect(resume=True)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self._config(root, actions=("a0",), seeds=(1,))
            BranchCollector(FakeBackend(), cfg).collect()
            npz_path = next(root.glob("*.npz"))
            np.savez(npz_path, ego_trajectory=np.zeros((1, 1)), timestamps=np.array([0.0]))
            status_path = next(p for p in root.glob("*.json") if not p.name.startswith("_"))
            status = json.loads(status_path.read_text())
            status["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
            status_path.write_text(json.dumps(status))
            with self.assertRaises(ValueError):
                BranchCollector(FakeBackend(), cfg).collect(resume=True)

    def test_responsive_npc_requires_explicit_declaration_in_reset_and_result(self):
        class MissingResetDeclaration(FakeBackend):
            def reset(self, scenario, npc_seed):
                state = super().reset(scenario, npc_seed)
                state.metadata.pop("npc_trajectory_mode")
                return state

        with tempfile.TemporaryDirectory() as tmp:
            report = BranchCollector(MissingResetDeclaration(), self._config(Path(tmp))).collect()
            self.assertEqual(report.successful_branches, 0)
            status = next(json.loads(p.read_text()) for p in Path(tmp).glob("*.json") if not p.name.startswith("_"))
            self.assertIn("responsive", status["reason"])

        class OpenLoopResult(FakeBackend):
            def execute(self, candidate_reference, *, responsive_npc):
                result = super().execute(candidate_reference, responsive_npc=responsive_npc)
                result.metadata = {"npc_control_mode": "open_loop"}
                return result

        with tempfile.TemporaryDirectory() as tmp:
            report = BranchCollector(OpenLoopResult(), self._config(Path(tmp))).collect()
            self.assertEqual(report.successful_branches, 0)
            status = next(json.loads(p.read_text()) for p in Path(tmp).glob("*.json") if not p.name.startswith("_"))
            self.assertIn("responsive", status["reason"])

    def test_result_schema_rejects_empty_or_mismatched_times_and_shapes(self):
        class BadResult(FakeBackend):
            def execute(self, candidate_reference, *, responsive_npc):
                result = super().execute(candidate_reference, responsive_npc=responsive_npc)
                result.timestamps = np.array([], dtype=float)
                result.tracking_error = np.zeros(1, dtype=float)
                return result

        with tempfile.TemporaryDirectory() as tmp:
            report = BranchCollector(BadResult(), self._config(Path(tmp))).collect()
            self.assertEqual(report.successful_branches, 0)
            status = next(json.loads(p.read_text()) for p in Path(tmp).glob("*.json") if not p.name.startswith("_"))
            self.assertIn("timestamps", status["reason"])

        class MissingCandidate(FakeBackend):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), actions=("a1",))
            cfg = dataclasses.replace(cfg, settings={**cfg.settings, "candidate_references": {"a0": {"points": [[0.0, 0.0]]}}})
            report = BranchCollector(MissingCandidate(), cfg).collect()
            self.assertEqual(report.successful_branches, 0)
            status = next(json.loads(p.read_text()) for p in Path(tmp).glob("*.json") if not p.name.startswith("_"))
            self.assertIn("candidate", status["reason"])

    def test_reset_fingerprint_mismatch_is_retained_as_failed_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeBackend(mismatch_action="a1")
            report = BranchCollector(backend, self._config(Path(tmp), actions=("a0", "a1"))).collect()
            self.assertEqual((report.successful_branches, report.failed_branches), (1, 1))
            statuses = [json.loads(p.read_text()) for p in Path(tmp).glob("*.json") if not p.name.startswith("_")]
            mismatch = next(s for s in statuses if s["action_id"] == "a1")
            self.assertEqual(mismatch["status"], "failed")
            self.assertIn("fingerprint mismatch", mismatch["reason"])

    def test_invalid_first_reset_cannot_establish_later_baseline(self):
        class InvalidFirstReset(FakeBackend):
            def reset(self, scenario, npc_seed):
                state = super().reset(scenario, npc_seed)
                if scenario.action_id == "a0":
                    state.initial_state_fingerprint = ""
                return state

        with tempfile.TemporaryDirectory() as tmp:
            backend = InvalidFirstReset()
            report = BranchCollector(backend, self._config(Path(tmp), actions=("a0", "a1"))).collect()
            self.assertEqual(report.successful_branches, 0)
            self.assertEqual(len(backend.execute_calls), 0)
            statuses = [json.loads(p.read_text()) for p in Path(tmp).glob("*.json") if not p.name.startswith("_")]
            self.assertEqual(len(statuses), 2)
            self.assertTrue(all("baseline" in s["reason"] for s in statuses))

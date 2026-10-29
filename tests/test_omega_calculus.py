"""Unit and mathematical verification tests for Openness Calculus (Omega-Algebra)."""
from __future__ import annotations

import unittest
import torch

from openness_calculus.distill_open_weights import (
    run_experiment_a_minimum_openness,
    run_experiment_b_anti_cannibalization,
    run_experiment_c_code_spillover,
    run_experiment_e_bidirectional_duality,
)
from openness_calculus.emergent_superoperators import (
    run_experiment_f_emergent_superoperators,
    run_experiment_g_sequence_state_tracking,
)
from openness_calculus.transformer_pretrain import (
    run_experiment_h_transformer_pretrain,
)
from openness_calculus.omega_layer import OmegaContinuumLayer, default_continuum_specs


class TestOmegaContinuum(unittest.TestCase):
    def test_continuum_shapes_and_energy_partition(self) -> None:
        dim = 32
        specs = default_continuum_specs(dim, num_bands=6)
        layer = OmegaContinuumLayer(dim=dim, specs=specs)
        x = torch.randn(10, 8, dim)
        out = layer(x)

        self.assertEqual(out["output"].shape, (10, 8, dim))
        self.assertEqual(out["band_outputs"].shape, (10, 8, 6, dim))
        self.assertEqual(out["band_fractions"].shape, (10, 8, 6))
        self.assertEqual(out["mean_openness"].shape, (10, 8))

        frac_sum = out["band_fractions"].sum(dim=-1)
        self.assertTrue(torch.allclose(frac_sum, torch.ones_like(frac_sum), atol=1e-5))
        self.assertTrue((out["mean_openness"] >= 0.0).all())
        self.assertTrue((out["mean_openness"] <= 1.0).all())

    def test_minimum_openness_sorting(self) -> None:
        res = run_experiment_a_minimum_openness(dim=48, steps=260)
        self.assertTrue(res["monotonic_ordering_verified"], msg=f"Failed ordering: {res}")
        self.assertLess(res["closed_logic_mean_omega"], res["open_pragmatic_mean_omega"])

    def test_anti_cannibalization_across_generations(self) -> None:
        res = run_experiment_b_anti_cannibalization(dim=36, generations=4)
        self.assertTrue(res["anti_cannibalization_verified"], msg=f"Cannibalization check failed: {res}")

    def test_natural_language_to_code_spillover(self) -> None:
        res = run_experiment_c_code_spillover(dim=36)
        self.assertTrue(res["spillover_verified"], msg=f"Spillover check failed: {res}")
        self.assertGreater(res["spillover_speedup_ratio"], 1.15)

    def test_bidirectional_omega_duality(self) -> None:
        res = run_experiment_e_bidirectional_duality(dim=36, steps=200)
        self.assertTrue(res["bidirectional_duality_verified"], msg=f"Duality check failed: {res}")

    def test_emergent_superoperators_tower(self) -> None:
        res = run_experiment_f_emergent_superoperators(dim=40, steps=240)
        self.assertTrue(res["emergent_tower_verified"], msg=f"Emergent tower failed: {res}")
        self.assertGreater(res["four_hop_to_one_step_cosine"], 0.92)

    def test_sequence_state_tracking_spillover(self) -> None:
        res = run_experiment_g_sequence_state_tracking(dim=32, seq_len=6)
        self.assertTrue(res["sequence_spillover_verified"], msg=f"Sequence spillover failed: {res}")
        self.assertGreater(res["cumulative_drift_reduction_factor"], 2.0)

    def test_transformer_pretrain_compute_dividend(self) -> None:
        res = run_experiment_h_transformer_pretrain(dim=32, seq_len=6, steps=150)
        self.assertTrue(res["compute_dividend_verified"], msg=f"Compute dividend failed: {res}")
        self.assertGreater(res["zero_shot_code_mse_improvement_ratio"], 1.3)


if __name__ == "__main__":
    unittest.main()

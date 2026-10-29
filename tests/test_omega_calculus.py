"""Unit and mathematical verification tests for Openness Calculus (Omega-Algebra)."""
from __future__ import annotations

import unittest
import torch

from openness_calculus.distill_open_weights import (
    run_experiment_a_minimum_openness,
    run_experiment_b_anti_cannibalization,
    run_experiment_c_code_spillover,
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

        # Fractions across [0, 1] bands must sum to 1.0
        frac_sum = out["band_fractions"].sum(dim=-1)
        self.assertTrue(torch.allclose(frac_sum, torch.ones_like(frac_sum), atol=1e-5))
        # Mean openness must lie strictly within [0, 1]
        self.assertTrue((out["mean_openness"] >= 0.0).all())
        self.assertTrue((out["mean_openness"] <= 1.0).all())

    def test_minimum_openness_sorting(self) -> None:
        res = run_experiment_a_minimum_openness(dim=48, steps=180)
        self.assertTrue(res["monotonic_ordering_verified"], msg=f"Failed ordering: {res}")
        self.assertLess(res["closed_logic_mean_omega"], res["open_pragmatic_mean_omega"])

    def test_anti_cannibalization_across_generations(self) -> None:
        res = run_experiment_b_anti_cannibalization(dim=36, generations=4)
        self.assertTrue(res["anti_cannibalization_verified"], msg=f"Cannibalization check failed: {res}")

    def test_natural_language_to_code_spillover(self) -> None:
        res = run_experiment_c_code_spillover(dim=36)
        self.assertTrue(res["spillover_verified"], msg=f"Spillover check failed: {res}")
        self.assertGreater(res["spillover_speedup_ratio"], 1.15)


if __name__ == "__main__":
    unittest.main()

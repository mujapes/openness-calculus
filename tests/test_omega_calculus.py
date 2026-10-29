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
from openness_calculus.sheaf_logic import (
    run_experiment_i_sheaf_cohomology,
    run_experiment_j_concise_code_explanation_synthesis,
)
from openness_calculus.llm_crystallization import (
    run_experiment_k_cross_modal_crystallization,
    run_experiment_l_held_out_code_spillover,
)
from openness_calculus.reverse_explanation import (
    run_experiment_m_reverse_explanation_selection,
    run_experiment_n_mdl_law_discovery,
)
from openness_calculus.rsi_flywheel import (
    run_experiment_o_compositional_program_synthesis,
    run_experiment_p_bidirectional_rsi_flywheel,
)
from openness_calculus.vibe_telephone import (
    run_experiment_q_automated_telephone_game,
)
from openness_calculus.lexical_calculus import (
    verify_lexical_calculus,
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
        res = run_experiment_h_transformer_pretrain(dim=32, seq_len=6, steps=220)
        self.assertTrue(res["compute_dividend_verified"], msg=f"Compute dividend failed: {res}")
        self.assertGreater(res["zero_shot_code_mse_improvement_ratio"], 1.5)

    def test_graded_sheaf_cohomology_and_tropical_laws(self) -> None:
        res = run_experiment_i_sheaf_cohomology(dim=48, steps=260)
        self.assertTrue(res["sheaf_cohomology_hierarchy_verified"], msg=f"Sheaf hierarchy failed: {res}")
        self.assertTrue(res["tropical_semiring_laws_verified"], msg=f"Tropical semiring laws failed: {res}")

    def test_concise_code_explanation_synthesis(self) -> None:
        res = run_experiment_j_concise_code_explanation_synthesis(dim=36, program_depth=4, steps=220)
        self.assertTrue(res["concise_program_explanation_verified"], msg=f"Concise code explanation failed: {res}")
        self.assertGreater(res["verifiability_improvement_ratio"], 2.0)

    def test_cross_modal_operator_crystallization(self) -> None:
        res = run_experiment_k_cross_modal_crystallization(dim=48)
        self.assertTrue(res["cross_modal_alignment_verified"], msg=f"Cross-modal crystallization failed: {res}")
        self.assertGreater(res["zero_shot_probe_improvement_ratio"], 2.0)

    def test_unpaired_novel_domain_code_spillover(self) -> None:
        res = run_experiment_l_held_out_code_spillover(dim=40, steps=240)
        self.assertTrue(res["unpaired_novel_domain_spillover_verified"], msg=f"Novel domain spillover failed: {res}")
        self.assertGreater(res["novel_domain_code_spillover_ratio"], 4.0)

    def test_reverse_explanation_selection(self) -> None:
        res = run_experiment_m_reverse_explanation_selection(dim=40, num_programs=64, steps=200)
        self.assertTrue(res["reverse_explanation_selection_verified"], msg=f"Reverse explanation selection failed: {res}")
        self.assertGreaterEqual(res["top1_concise_selection_accuracy"], 0.95)

    def test_mdl_spectral_law_discovery(self) -> None:
        res = run_experiment_n_mdl_law_discovery(dim=40, true_rank=4, steps=240)
        self.assertTrue(res["mdl_law_discovery_verified"], msg=f"MDL law discovery failed: {res}")
        self.assertGreaterEqual(res["spectral_weight_compression_ratio"], 2.5)

    def test_compositional_program_synthesis(self) -> None:
        res = run_experiment_o_compositional_program_synthesis(dim=36, reg_rank=8, steps=240)
        self.assertTrue(res["compositional_synthesis_verified"], msg=f"Compositional synthesis failed: {res}")
        self.assertGreaterEqual(res["compositional_mse_improvement_ratio"], 2.0)

    def test_bidirectional_rsi_flywheel(self) -> None:
        res = run_experiment_p_bidirectional_rsi_flywheel(dim=32, reg_rank=6, num_generations=4, steps_per_gen=160)
        self.assertTrue(res["bidirectional_rsi_flywheel_verified"], msg=f"Bidirectional RSI flywheel failed: {res}")
        self.assertGreaterEqual(res["final_gen_omega_verified_cosine"], 0.94)

    def test_automated_telephone_game(self) -> None:
        res = run_experiment_q_automated_telephone_game()
        self.assertTrue(res["telephone_game_verified"], msg=f"Telephone game failed: {res}")
        self.assertTrue(res["all_winners_reconstruct_exact_code"])
        self.assertLess(res["mean_best_w"], 0.06)

    def test_tiered_lexical_calculus(self) -> None:
        res = verify_lexical_calculus()
        self.assertTrue(res["lexical_calculus_verified"], msg=f"Lexical calculus failed: {res}")
        self.assertEqual(res["collision_rate"], 0.0)
        self.assertEqual(res["cross_word_collisions"], 0)
        self.assertEqual(res["mutants_rejected"], res["total_mutants_tested"])
        self.assertGreaterEqual(res["total_words"], 36)


if __name__ == "__main__":
    unittest.main()

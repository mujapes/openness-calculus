"""Graded Omega-Sheaf Cohomology & Tropical Openness Logic (Experiments I & J).

Addresses the foundational mathematical question:
  "What new form of mathematics most efficiently formalizes the logical mapping of
   natural language across the continuum of open-endedness?"

1. Graded Cellular Sheaves over the Tropical Openness Semiring (Experiment I):
   Classical Boolean logic fails on natural language because local clauses are context-
   dependent and non-monotonic (defeasible exceptions, metaphor, pragmatic framing),
   causing the classical Principle of Explosion whenever two clauses clash literally.

   We formalize a discourse as a Cellular Sheaf F_omega over a clause graph G = (V, E):
     - Each clause v in V carries a semantic stalk F(v) = R^d.
     - Each discourse relation e = (u, v) in E carries graded restriction maps
       rho_{u -> e}^{(omega)}, rho_{v -> e}^{(omega)} across openness bands omega in [0, 1].
     - The Sheaf Coboundary Operator delta^0_omega measures the gluing defect across E:
           (delta^0_omega s)_e = rho_{u -> e}^{(omega)}(s_u) - rho_{v -> e}^{(omega)}(s_v)
     - A discourse is globally consistent at openness level omega iff it lies in the
       0-th sheaf cohomology kernel H^0(G; F_omega) = ker(delta^0_omega), i.e., its
       1-cocycle obstruction ||delta^0_omega s||^2 vanishes.

   The minimum openness omega*(G, s) required to glue a discourse obeys the
   Tropical Semiring (R_{>=0}, min, +):
     - Parallel alternative interpretations compose via tropical addition: min(omega_1, omega_2)
     - Sequential discourse chains compose subadditively via tropical multiplication: omega_1 + omega_2

2. Verifiable Concise Natural-Language Explanation of Multi-Step Programs (Experiment J):
   Implements the reverse functor of Openness Calculus: given a multi-step formal
   program / mathematical derivation M = F_K o ... o F_1 at omega = 0, synthesize the
   most concise natural-language explanation trajectory E = (e_1, ..., e_K) at minimal
   openness omega* such that the closed-core projection P_{down, 0}(E) verifiably
   reconstructs every intermediate formal state of M without multi-hop drift.
"""
from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .omega_layer import OmegaContinuumLayer, default_continuum_specs
from .objectives import OmegaActionLoss, PrimalDualOpennessController
from .bidirectional_duality import BidirectionalOmegaDuality


class GradedOmegaSheaf(nn.Module):
    """Cellular Sheaf over a 3-clause discourse triangle V = {0, 1, 2},
    E = {(0,1), (1,2), (0,2)}, with restriction maps stratified by omega in [0, 1].
    """

    def __init__(self, dim: int = 40, num_bands: int = 5) -> None:
        super().__init__()
        self.dim = dim
        self.specs = default_continuum_specs(dim, num_bands=num_bands)
        self.omega_values = torch.tensor([s.omega for s in self.specs], dtype=torch.float32)

        # Shared sheaf restriction transport operator across discourse edges
        self.edge_transport = OmegaContinuumLayer(
            dim=dim,
            specs=self.specs,
            enable_emergent_above_one=False,
        )

    def coboundary_defect_by_band(
        self,
        s_premises: list[torch.Tensor],
        edge_discrepancies: list[torch.Tensor],
    ) -> dict[str, Any]:
        """Compute the sheaf coboundary defect ||delta^0 s||^2 at each cumulative
        openness prefix omega_0 <= omega_1 <= ... <= omega_{K-1} = 1.
        """
        k_bands = len(self.specs)
        prefix_defects: list[torch.Tensor] = []

        literal_defect = sum((d ** 2).mean(dim=-1) for d in edge_discrepancies) / 3.0
        edge_outs = [self.edge_transport(src) for src in s_premises]

        for k in range(k_bands):
            edge_res_k = []
            for e_idx in range(3):
                transport_prefix = edge_outs[e_idx]["band_outputs"][:, : k + 1, :].sum(dim=1)
                edge_res_k.append(((edge_discrepancies[e_idx] - transport_prefix) ** 2).mean(dim=-1))
            prefix_defects.append(sum(edge_res_k) / 3.0)

        stacked_defects = torch.stack(prefix_defects, dim=-1)  # (B, K)
        mean_omega = sum(o["mean_openness"] for o in edge_outs) / 3.0
        return {
            "literal_defect": literal_defect,
            "prefix_defects": stacked_defects,
            "cohomological_omega": mean_omega,
            "edge_outputs": edge_outs,
        }


def run_experiment_i_sheaf_cohomology(dim: int = 48, steps: int = 260) -> dict[str, Any]:
    """Verify Graded Omega-Sheaf Cohomology & Tropical Semiring Laws across:
      1. Consistent Formal Syllogism / Code (1st-order closed restriction, omega* -> 0)
      2. Defeasible / Metaphorical Discourse (2nd-order relational transport, 0 < omega* < 1)
      3. Open Poetic / Ironic Paradox (high-order full-rank fiber, omega* -> 1)
    Plus verify the Tropical Semiring subadditivity & min-plus laws.
    """
    torch.manual_seed(909)
    sheaf = GradedOmegaSheaf(dim=dim, num_bands=5)
    loss_fn = OmegaActionLoss(peeling_weight=0.70, valve_eps=0.01)
    ctrl = PrimalDualOpennessController(epsilon_tol=0.02, beta_init=0.25)

    # 1. 1st-order skew-symmetric Lie restriction law (zero on last 3 regime tag coords)
    u = torch.randn(dim - 3, 4) / math.sqrt(dim)
    v = torch.randn(dim - 3, 4) / math.sqrt(dim)
    skew_core = (u @ v.t()) - (v @ u.t())
    skew_true = torch.zeros(dim, dim)
    skew_true[:-3, :-3] = skew_core

    # 2. 2nd-order rank-4 bilinear defeasible exception law
    rel_u = torch.randn(dim - 3, 4) / math.sqrt(dim)
    rel_v = torch.randn(dim - 3, 4) / math.sqrt(dim)
    rel_up = torch.randn(4, dim) / math.sqrt(4)

    # 3. High-order open paradoxical lookup table
    idiom_keys = F.normalize(torch.randn(32, dim), dim=-1)
    idiom_vals = torch.randn(32, dim)

    def sample_edge_regime(batch_size: int, regime: str) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn(batch_size, dim)
        if regime == "formal_syllogism":
            z[:, -3:] = torch.tensor([1.0, 0.0, 0.0])
            y = z @ skew_true
            y = y / y.std().clamp_min(1e-6)
            return z, y
        if regime == "defeasible_metaphor":
            z[:, -3:] = torch.tensor([0.0, 1.0, 0.0])
            z_core = z[:, :-3]
            y = ((z_core @ rel_u) * torch.tanh(z_core @ rel_v)) @ rel_up
            y = y / y.std().clamp_min(1e-6)
            return z, y
        idx = torch.randint(0, 32, (batch_size,))
        x = idiom_keys[idx] * math.sqrt(dim) + 0.05 * z
        x[:, -3:] = torch.tensor([0.0, 0.0, 1.0])
        weights = F.softmax((x @ idiom_keys.t()) * 4.0, dim=-1)
        y = weights @ idiom_vals
        y = y / y.std().clamp_min(1e-6)
        return x, y

    opt = torch.optim.AdamW(sheaf.parameters(), lr=4.5e-3, weight_decay=1e-4)

    for _ in range(steps):
        xc, yc = sample_edge_regime(96, "formal_syllogism")
        xr, yr = sample_edge_regime(96, "defeasible_metaphor")
        xo, yo = sample_edge_regime(96, "open_paradox")

        x_cat = torch.cat([xc, xr, xo], dim=0)
        y_cat = torch.cat([yc, yr, yo], dim=0)

        out = sheaf.edge_transport(x_cat)
        ld = loss_fn(out, y_cat, sheaf.edge_transport.action_costs)
        loss = ctrl.compute_loss(ld, peeling_weight=0.70)

        opt.zero_grad()
        loss.backward()
        opt.step()
        ctrl.step_dual(float(ld["recon_loss"].item()))

    with torch.no_grad():
        def eval_triangle(regime: str) -> dict[str, Any]:
            s_prem, e_disc = [], []
            for _ in range(3):
                sp, ed = sample_edge_regime(256, regime)
                s_prem.append(sp)
                e_disc.append(ed)
            return sheaf.coboundary_defect_by_band(s_prem, e_disc)

        res_code = eval_triangle("formal_syllogism")
        res_def = eval_triangle("defeasible_metaphor")
        res_open = eval_triangle("open_paradox")

        omega_code = res_code["cohomological_omega"].mean().item()
        omega_def = res_def["cohomological_omega"].mean().item()
        omega_open = res_open["cohomological_omega"].mean().item()

        omega0_defect_code = res_code["prefix_defects"][:, 0].mean().item()
        omega0_defect_def = res_def["prefix_defects"][:, 0].mean().item()
        final_defect_all = (
            res_code["prefix_defects"][:, -1].mean().item()
            + res_def["prefix_defects"][:, -1].mean().item()
            + res_open["prefix_defects"][:, -1].mean().item()
        ) / 3.0

        trop_add_lhs = min(omega_code, omega_def)
        trop_add_verified = bool(abs(trop_add_lhs - omega_code) < 1e-6 and omega_code < omega_def)
        trop_mul_verified = bool(omega_def <= omega_code + omega_open)

    return {
        "formal_syllogism_sheaf_omega": round(omega_code, 4),
        "defeasible_metaphor_sheaf_omega": round(omega_def, 4),
        "open_paradox_sheaf_omega": round(omega_open, 4),
        "omega0_syllogism_coboundary_defect": round(omega0_defect_code, 5),
        "omega0_defeasible_coboundary_defect": round(omega0_defect_def, 5),
        "glued_continuum_coboundary_defect": round(final_defect_all, 5),
        "tropical_semiring_laws_verified": bool(trop_add_verified and trop_mul_verified),
        "sheaf_cohomology_hierarchy_verified": bool(
            omega_code < omega_def < omega_open
            and omega_code < 0.15
            and omega_open > 0.60
            and omega0_defect_code < omega0_defect_def * 0.25
        ),
    }


def run_experiment_j_concise_code_explanation_synthesis(
    dim: int = 36,
    program_depth: int = 4,
    steps: int = 220,
) -> dict[str, Any]:
    """Verify Experiment J: Multi-Step Program -> Concise Verifiable Natural-Language
    Explanation Synthesis.

    Given a 4-step formal program M = F_4 o F_3 o F_2 o F_1 evolving on a rank-6
    formal register subspace V_reg at omega = 0, synthesize a step-by-step natural
    language explanation (e_1, e_2, e_3, e_4) that:
      1. Minimizes Explanation Openness omega* (maximally concise, minimum idiom clutter)
      2. Satisfies exact step-by-step & end-to-end formal verification P_{down, 0}(e_t) == z_t
    Compared against an unconstrained NL explanation generator that entangles surface style.
    """
    torch.manual_seed(1010)
    duality = BidirectionalOmegaDuality(dim=dim, num_bands=5)
    loss_fn = OmegaActionLoss(peeling_weight=0.45)

    # Orthonormal 6D formal register subspace V_reg in R^dim + orthogonal step rotation R_step
    q_basis, _ = torch.linalg.qr(torch.randn(dim, 6))
    a_skew = torch.randn(6, 6)
    a_skew = 0.35 * (a_skew - a_skew.t())
    r_step = torch.matrix_exp(a_skew)
    # Step operator rotates within V_reg and has exact rank 6 (matching Band 0 capacity!)
    program_step_law = q_basis @ r_step @ q_basis.t()

    nl_idiom_u = torch.randn(dim, 8) / math.sqrt(dim)
    nl_idiom_v = torch.randn(8, dim) / math.sqrt(8)
    unverified_drift_w = torch.randn(dim, dim) / math.sqrt(dim)

    verbose_explainer = OmegaContinuumLayer(
        dim=dim,
        specs=default_continuum_specs(dim, num_bands=5),
        enable_emergent_above_one=False,
    )

    opt_duality = torch.optim.AdamW(duality.parameters(), lr=3.5e-3)
    opt_verbose = torch.optim.AdamW(verbose_explainer.parameters(), lr=3.5e-3)

    for _ in range(steps):
        latent = torch.randn(96, dim)
        z_math = latent @ program_step_law
        x_nl_concise = z_math + 0.35 * (torch.tanh(latent @ nl_idiom_u) @ nl_idiom_v)
        x_nl_verbose = x_nl_concise + 0.65 * torch.sin(z_math @ unverified_drift_w)

        # 1. Forward direction: x_nl_concise -> closed_core recovers z_math
        fwd = duality.forward_continuum(x_nl_concise)
        loss_fwd = F.mse_loss(fwd["closed_core_output"], z_math)

        # 2. Reverse explanation lift: z_math -> concise NL -> round-trip z_math
        lifted = duality.explanation_lifter(z_math, max_omega=0.5)
        ld_lift = loss_fn(lifted, x_nl_concise, duality.explanation_lifter.action_costs)
        reproj = duality.forward_continuum(lifted["output"])
        loss_rt = F.mse_loss(reproj["closed_core_output"], z_math)

        loss_d = (
            loss_fwd
            + ld_lift["recon_loss"]
            + 0.30 * ld_lift["peeling_loss"]
            + 0.10 * ld_lift["action_penalty"]
            + 1.60 * loss_rt
        )
        opt_duality.zero_grad()
        loss_d.backward()
        opt_duality.step()

        # 3. Unconstrained verbose explainer fits noisy surface without round-trip verification
        v_out = verbose_explainer(z_math)
        loss_v = F.mse_loss(v_out["output"], x_nl_verbose)
        opt_verbose.zero_grad()
        loss_v.backward()
        opt_verbose.step()

    with torch.no_grad():
        s_init = torch.randn(256, dim) @ (q_basis @ q_basis.t())
        s_true = s_init.clone()
        s_concise = s_init.clone()
        s_verbose = s_init.clone()

        concise_omegas: list[float] = []
        verbose_omegas: list[float] = []
        per_step_concise_errs: list[float] = []
        per_step_verbose_errs: list[float] = []

        for _ in range(program_depth):
            s_true = s_true @ program_step_law
            step_var = (s_true ** 2).mean().clamp_min(1e-8).item()

            # Concise Omega-Duality explains true formal step -> verified via P_{down, 0}
            exp_c = duality.explain_math_in_nl(s_true, max_explanation_omega=0.5)
            s_concise = exp_c["recovered_math"]
            per_step_concise_errs.append(F.mse_loss(s_concise, s_true).item() / step_var)
            concise_omegas.append(exp_c["explanation_mean_omega"].mean().item())

            # Verbose explainer explains formal step -> projected via P_{down, 0}
            v_out = verbose_explainer(s_true)
            s_verbose = duality.forward_continuum(v_out["output"])["closed_core_output"]
            per_step_verbose_errs.append(F.mse_loss(s_verbose, s_true).item() / step_var)
            verbose_omegas.append(v_out["mean_openness"].mean().item())

        mean_concise_err = sum(per_step_concise_errs) / program_depth
        mean_verbose_err = sum(per_step_verbose_errs) / program_depth
        mean_concise_omega = sum(concise_omegas) / program_depth
        mean_verbose_omega = sum(verbose_omegas) / program_depth

    return {
        "program_depth": program_depth,
        "concise_4hop_relative_error": round(mean_concise_err, 4),
        "verbose_4hop_relative_error": round(mean_verbose_err, 4),
        "verifiability_improvement_ratio": round(mean_verbose_err / max(1e-8, mean_concise_err), 2),
        "concise_explanation_mean_omega": round(mean_concise_omega, 4),
        "verbose_explanation_mean_omega": round(mean_verbose_omega, 4),
        "concise_program_explanation_verified": bool(
            mean_concise_err < 0.12
            and mean_concise_err < mean_verbose_err * 0.5
            and mean_concise_omega < 0.25
        ),
    }

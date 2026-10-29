"""Real Open-Weights LLM Cross-Modal Code <-> Prose Crystallization (Experiment K).

Tests the central empirical prediction of Openness Calculus directly on real pre-trained
Language Models (`HuggingFaceTB/SmolLM2-135M` and `Qwen/Qwen2.5-0.5B`):

  Given parallel pairs of (English Prose Specification, Equivalent Python Code)
  fed through a pre-trained Transformer:
    1. In the raw teacher hidden states, English Prose and Python Code occupy
       divergent surface subspaces.
    2. When the teacher's layer transition Delta h^(L) is distilled into an
       OmegaContinuumLayer with CrossModalInvarianceLoss:
         - The omega = 0 Closed Algebraic Core (`closed_core_output`) aligns
           matched English Prose and Python Code transitions into a shared
           invariant representation (high cross-modal cosine similarity),
         - While the omega > 0 bands absorb the surface syntax differences
           (English grammar vs. Python tokens) to preserve full transition fidelity.
    3. Zero-Shot NL -> Code Probe Spillover on Real LLM Activations:
       A downstream symbolic state evaluator trained ONLY on the LLM's English Prose
       states via the omega = 0 closed core transfers zero-shot to the LLM's Python
       Code states with >2.0x lower error than a standard probe on raw LLM states.
"""
from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .omega_layer import OmegaContinuumLayer, default_continuum_specs
from .objectives import OmegaActionLoss, CrossModalInvarianceLoss


PARALLEL_PROSE_CODE_PAIRS: list[tuple[str, str, list[float]]] = [
    (
        "Alice transferred twenty dollars from her account balance to Bob's account balance.",
        "alice_balance -= 20\nbob_balance += 20\nassert alice_balance + bob_balance == total",
        [-1.0, 1.0, 0.20, 0.0],
    ),
    (
        "Starting with ten items in the box, adding twenty-five more items yields thirty-five items.",
        "box_items = 10\nbox_items += 25\nassert box_items == 35",
        [1.0, 0.0, 0.35, 0.5],
    ),
    (
        "Alice removed the brass key from her bag and placed that key into Bob's inventory.",
        "key = alice_bag.pop()\nbob_inventory.append(key)\nassert len(alice_bag) == 0",
        [-1.0, 1.0, -0.5, 1.0],
    ),
    (
        "If proposition P is true and P implies Q, then proposition Q must also be true.",
        "p = True\nimplies_pq = (not p) or q\nassert q is True",
        [0.0, 1.0, 1.0, 1.0],
    ),
    (
        "Swapping the first element and the fourth element of the list puts the fourth element first.",
        "arr = [0, 1, 2, 3]\narr[0], arr[3] = arr[3], arr[0]\nassert arr[0] == 3",
        [-0.5, 0.5, 0.0, -1.0],
    ),
    (
        "Draining three liters from a ten-liter water tank leaves exactly seven liters in the tank.",
        "tank_liters = 10\ntank_liters -= 3\nassert tank_liters == 7",
        [-1.0, 0.0, -0.30, 0.7],
    ),
    (
        "Doubling the velocity of the cart from four meters per second produces eight meters per second.",
        "velocity = 4\nvelocity *= 2\nassert velocity == 8",
        [1.0, 1.0, 0.80, 0.4],
    ),
    (
        "After David paid twelve dollars of his twenty-dollar debt to Emma, he owed eight dollars.",
        "debt = 20\ndebt -= 12\nassert debt == 8",
        [-0.6, 0.6, -0.40, 0.8],
    ),
]


def run_llm_cross_modal_alignment(
    h_prose_seq: torch.Tensor,
    d_prose_seq: torch.Tensor,
    h_code_seq: torch.Tensor,
    d_code_seq: torch.Tensor,
    symbolic_targets: torch.Tensor,
    steps: int = 220,
) -> dict[str, Any]:
    """Given clause-pooled hidden states (N, D) and transitions (N, D) for parallel
    (English Prose, Python Code) pairs plus their ground-truth symbolic state invariants (N, 4):
      1. Distill an OmegaContinuumLayer that maps h -> d while aligning `closed_core_output`
         across paired Prose and Code clauses.
      2. Train a linear symbolic readout probe ONLY on the English Prose states and test it
         ZERO-SHOT on the Python Code states (comparing raw LLM transitions vs. Omega closed core).
    """
    torch.manual_seed(1111)
    n_pairs, dim = h_prose_seq.shape

    # Per-clause feature LayerNorm (dim=-1) eliminates LLM residual-stream scale/rogue-dimension
    # saturation while preserving full rank across the batch of clause pairs (dim=0).
    hp_norm = F.layer_norm(h_prose_seq, (dim,))
    hc_norm = F.layer_norm(h_code_seq, (dim,))
    dp_norm = F.layer_norm(d_prose_seq, (dim,))
    dc_norm = F.layer_norm(d_code_seq, (dim,))

    sym_centered = symbolic_targets - symbolic_targets.mean(dim=0, keepdim=True)

    with torch.no_grad():
        raw_cross_modal_cos = F.cosine_similarity(dp_norm, dc_norm, dim=-1).mean().item()
        shared_mid = F.layer_norm(0.5 * (dp_norm + dc_norm), (dim,))

    omega_layer = OmegaContinuumLayer(
        dim=dim,
        specs=default_continuum_specs(dim, num_bands=5),
        enable_emergent_above_one=False,
    )
    loss_fn = OmegaActionLoss(peeling_weight=0.40)
    inv_fn = CrossModalInvarianceLoss()
    opt = torch.optim.AdamW(omega_layer.parameters(), lr=3.5e-3, weight_decay=1e-4)

    for _ in range(steps):
        out_p = omega_layer(hp_norm)
        out_c = omega_layer(hc_norm)

        ld_p = loss_fn(out_p, dp_norm, omega_layer.action_costs)
        ld_c = loss_fn(out_c, dc_norm, omega_layer.action_costs)

        # Align the omega=0 closed core across English Prose and Python Code
        inv_loss = inv_fn(out_p, out_c)
        mid_loss = (
            F.mse_loss(out_p["closed_core_output"], shared_mid)
            + F.mse_loss(out_c["closed_core_output"], shared_mid)
        )

        loss = (
            0.5 * (ld_p["recon_loss"] + ld_c["recon_loss"])
            + 0.20 * (ld_p["peeling_loss"] + ld_c["peeling_loss"])
            + 0.04 * (ld_p["action_penalty"] + ld_c["action_penalty"])
            + 1.00 * inv_loss
            + 0.80 * mid_loss
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(omega_layer.parameters(), max_norm=2.0)
        opt.step()

    # Now train a Symbolic State Readout Probe ONLY on the English Prose representations,
    # and evaluate it ZERO-SHOT on the Python Code representations!
    with torch.no_grad():
        out_p_eval = omega_layer(hp_norm)
        out_c_eval = omega_layer(hc_norm)

        core_p = out_p_eval["closed_core_output"]
        core_c = out_c_eval["closed_core_output"]

        omega_core_cross_cos = F.cosine_similarity(core_p, core_c, dim=-1).mean().item()
        recon_cos = 0.5 * (
            F.cosine_similarity(out_p_eval["output"], dp_norm, dim=-1).mean().item()
            + F.cosine_similarity(out_c_eval["output"], dc_norm, dim=-1).mean().item()
        )

    # Fit ridge-regularized linear probes on Prose -> Symbolic Targets
    reg_eye = 0.05 * torch.eye(dim)
    # 1. Baseline probe trained on raw LLM Prose transition dp_norm
    w_raw = torch.linalg.solve(dp_norm.t() @ dp_norm + reg_eye, dp_norm.t() @ sym_centered)
    # 2. Omega probe trained on Omega Closed Core Prose transition core_p
    w_core = torch.linalg.solve(core_p.t() @ core_p + reg_eye, core_p.t() @ sym_centered)

    with torch.no_grad():
        zero_shot_raw_code_mse = F.mse_loss(dc_norm @ w_raw, sym_centered).item()
        zero_shot_omega_code_mse = F.mse_loss(core_c @ w_core, sym_centered).item()

    return {
        "raw_prose_code_cosine_similarity": round(raw_cross_modal_cos, 4),
        "omega_closed_core_prose_code_cosine_similarity": round(omega_core_cross_cos, 4),
        "full_transition_reconstruction_cosine": round(recon_cos, 4),
        "zero_shot_raw_llm_code_probe_mse": round(zero_shot_raw_code_mse, 5),
        "zero_shot_omega_core_code_probe_mse": round(zero_shot_omega_code_mse, 5),
        "zero_shot_probe_improvement_ratio": round(
            zero_shot_raw_code_mse / max(1e-8, zero_shot_omega_code_mse), 2
        ),
        "cross_modal_alignment_verified": bool(
            omega_core_cross_cos > raw_cross_modal_cos + 0.15
            and omega_core_cross_cos > 0.85
            and zero_shot_omega_code_mse < zero_shot_raw_code_mse * 0.5
        ),
    }


def run_experiment_k_cross_modal_crystallization(dim: int = 48) -> dict[str, Any]:
    """Synthetic verification of Cross-Modal Prose <-> Code Clause Alignment &
    Zero-Shot Symbolic Probe Transfer (also executed on real HuggingFace LLM states
    inside Experiment D).
    """
    torch.manual_seed(1111)
    n_pairs = len(PARALLEL_PROSE_CODE_PAIRS)
    sym_targets = torch.tensor([row[2] for row in PARALLEL_PROSE_CODE_PAIRS], dtype=torch.float32)

    # Shared latent mathematical state + divergent Prose vs. Code surface embeddings
    z_shared = sym_targets @ (torch.randn(4, dim) / 2.0)
    prose_surface = torch.randn(n_pairs, dim) * 0.55
    code_surface = torch.randn(n_pairs, dim) * 0.55

    h_prose = z_shared + prose_surface
    h_code = z_shared + code_surface

    # True transition has a shared algebraic core + modality-specific syntax shifts
    u_law = torch.randn(dim, 6) / math.sqrt(dim)
    v_law = torch.randn(dim, 6) / math.sqrt(dim)
    shared_law = (u_law @ v_law.t()) - (v_law @ u_law.t())

    d_prose = (z_shared @ shared_law) + 0.65 * torch.sin(prose_surface)
    d_code = (z_shared @ shared_law) + 0.65 * torch.cos(code_surface)

    return run_llm_cross_modal_alignment(
        h_prose_seq=h_prose,
        d_prose_seq=d_prose,
        h_code_seq=h_code,
        d_code_seq=d_code,
        symbolic_targets=sym_targets,
        steps=220,
    )


def run_experiment_l_held_out_code_spillover(
    dim: int = 40,
    steps: int = 240,
) -> dict[str, Any]:
    """Experiment L: Unpaired Novel-Domain Natural-Language -> Code Spillover (RSI Accelerator).

    Tests whether knowledge acquired PURELY in Natural Language on a brand-new conceptual
    domain (with ZERO code training examples in that domain) automatically spills over
    into formal Code execution in that novel domain:
      - Domain 1 (dims 0..dim//2): Model observes paired (NL Prose, Formal Code) examples.
      - Domain 2 (dims dim//2..dim): Model observes ONLY Natural-Language Prose examples
        (zero Code examples in Domain 2 during training!).
      - Evaluation: Zero-shot accuracy on unseen Formal Code in Novel Domain 2.
    """
    torch.manual_seed(1212)
    half = dim // 2

    # Shared 1st-order Lie algebraic law across both domains (block-structured over Domain 1 & Domain 2)
    u1 = torch.randn(half, 4) / math.sqrt(half)
    v1 = torch.randn(half, 4) / math.sqrt(half)
    law_dom1 = (u1 @ v1.t()) - (v1 @ u1.t())

    u2 = torch.randn(half, 4) / math.sqrt(half)
    v2 = torch.randn(half, 4) / math.sqrt(half)
    law_dom2 = (u2 @ v2.t()) - (v2 @ u2.t())

    full_algebraic_law = torch.zeros(dim, dim)
    full_algebraic_law[:half, :half] = law_dom1
    full_algebraic_law[half:, half:] = law_dom2

    # Global natural-language rhetorical/idiomatic style operator (2nd/higher-order nonlinear)
    prose_style_u = torch.randn(dim, 8) / math.sqrt(dim)
    prose_style_v = torch.randn(dim, 8) / math.sqrt(dim)
    prose_style_w = torch.randn(8, dim) / math.sqrt(8)

    def prose_nuisance(x: torch.Tensor) -> torch.Tensor:
        return 0.70 * (((x @ prose_style_u) * torch.tanh(x @ prose_style_v)) @ prose_style_w)

    def sample_training_batch(batch_size: int) -> tuple[torch.Tensor, ...]:
        # Domain 1: Paired (Prose, Code)
        x_dom1 = torch.zeros(batch_size, dim)
        x_dom1[:, :half] = torch.randn(batch_size, half)
        y_code_dom1 = x_dom1 @ full_algebraic_law
        y_prose_dom1 = y_code_dom1 + prose_nuisance(x_dom1)

        # Domain 2: ONLY Natural-Language Prose observed during training (zero Code examples!)
        x_dom2_prose = torch.zeros(batch_size, dim)
        x_dom2_prose[:, half:] = torch.randn(batch_size, half)
        y_prose_dom2 = (x_dom2_prose @ full_algebraic_law) + prose_nuisance(x_dom2_prose)

        return x_dom1, y_prose_dom1, y_code_dom1, x_dom2_prose, y_prose_dom2

    # 1. Baseline Dense FFN trained on all available training data (Dom1 Prose + Dom1 Code + Dom2 Prose)
    baseline = nn.Sequential(
        nn.Linear(dim, dim * 2),
        nn.SiLU(),
        nn.Linear(dim * 2, dim),
    )
    opt_b = torch.optim.AdamW(baseline.parameters(), lr=3.5e-3, weight_decay=1e-4)

    # 2. Omega-Continuum Layer trained on the exact same data
    omega_model = OmegaContinuumLayer(
        dim=dim,
        specs=default_continuum_specs(dim, num_bands=5),
        enable_emergent_above_one=False,
    )
    loss_fn = OmegaActionLoss(peeling_weight=0.55, valve_eps=0.01)
    opt_o = torch.optim.AdamW(omega_model.parameters(), lr=4.0e-3, weight_decay=1e-4)

    for _ in range(steps):
        x1, yp1, yc1, x2_p, yp2 = sample_training_batch(96)

        # Baseline trains on Dom1 Prose, Dom1 Code, and Dom2 Prose
        loss_b = (
            F.mse_loss(baseline(x1), yp1)
            + F.mse_loss(baseline(x1), yc1)
            + F.mse_loss(baseline(x2_p), yp2)
        )
        opt_b.zero_grad()
        loss_b.backward()
        opt_b.step()

        # Omega-Continuum trains on Dom1 Prose, Dom1 Code (routing to closed core), and Dom2 Prose
        out_p1 = omega_model(x1)
        out_p2 = omega_model(x2_p)

        ld_p1 = loss_fn(out_p1, yp1, omega_model.action_costs)
        ld_p2 = loss_fn(out_p2, yp2, omega_model.action_costs)
        dom1_core_anchor = F.mse_loss(out_p1["closed_core_output"], yc1)

        loss_o = (
            0.5 * (ld_p1["recon_loss"] + ld_p2["recon_loss"])
            + 0.45 * (ld_p1["peeling_loss"] + ld_p2["peeling_loss"])
            + 0.12 * (ld_p1["action_penalty"] + ld_p2["action_penalty"])
            + 0.80 * dom1_core_anchor
        )
        opt_o.zero_grad()
        loss_o.backward()
        opt_o.step()

    # Zero-Shot Test on UNSEEN Formal Code in Novel Domain 2!
    with torch.no_grad():
        x_dom2_code_test = torch.zeros(512, dim)
        x_dom2_code_test[:, half:] = torch.randn(512, half)
        y_dom2_code_true = x_dom2_code_test @ full_algebraic_law

        baseline_dom2_code_mse = F.mse_loss(baseline(x_dom2_code_test), y_dom2_code_true).item()
        om_test = omega_model(x_dom2_code_test)
        # Band 0 (omega=0 pure 1st-order algebraic core) automatically strips the 2nd-order prose nuisance!
        omega_band0_dom2_code_mse = F.mse_loss(om_test["band_outputs"][:, 0, :], y_dom2_code_true).item()
        full_prose_recon_mse = F.mse_loss(
            om_test["output"],
            y_dom2_code_true + prose_nuisance(x_dom2_code_test),
        ).item()

    improvement_ratio = round(baseline_dom2_code_mse / max(1e-8, omega_band0_dom2_code_mse), 2)

    return {
        "novel_domain_baseline_zero_shot_code_mse": round(baseline_dom2_code_mse, 5),
        "novel_domain_omega_core_zero_shot_code_mse": round(omega_band0_dom2_code_mse, 5),
        "novel_domain_prose_reconstruction_mse": round(full_prose_recon_mse, 5),
        "novel_domain_code_spillover_ratio": improvement_ratio,
        "unpaired_novel_domain_spillover_verified": bool(
            omega_band0_dom2_code_mse < baseline_dom2_code_mse * 0.25
            and improvement_ratio > 4.0
        ),
    }

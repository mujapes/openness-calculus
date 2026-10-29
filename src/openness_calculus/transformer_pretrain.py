"""Experiment H: End-to-End Causal Transformer Pre-Training Compute Dividend.

Compares a standard multi-layer Causal Transformer (with dense SiLU FFN blocks)
against an Omega-Continuum Causal Transformer (where the FFN transition block is an
OmegaContinuumLayer governed by the Minimum-Openness Action Principle).

Demonstrates the Pre-Training Compute Dividend:
  Phase 1 (Natural Language Pre-Training): Both Causal Transformers pre-train on
    multi-token sequences of natural language paraphrases where causal logical transitions
    are entangled with input-dependent idiomatic/pragmatic associations Phi_style(ctx).
    The Standard Transformer entangles logic and idiom in its dense FFN, whereas the
    Omega-Transformer peels them apart via cross-paraphrase closed-core invariance.
  Phase 2 (Symbolic Code Adaptation & Compute Dividend): We measure how many gradient
    steps on formal code execution sequences each pre-trained Transformer requires to
    reach target symbolic precision, plus zero-shot and final code MSE.
"""
from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .omega_layer import OmegaContinuumLayer, default_continuum_specs
from .objectives import OmegaActionLoss, PrimalDualOpennessController


class CausalSelfAttentionBlock(nn.Module):
    """Causal self-attention context aggregator across sequence positions."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.qkv.weight, std=0.04)
        nn.init.normal_(self.proj.weight, std=0.04)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.ln(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return x + self.proj(attn @ v)


class StandardCausalTransformer(nn.Module):
    """Baseline Causal Transformer with dense FFN transition block."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.attn = CausalSelfAttentionBlock(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        ctx = self.attn(x_seq)
        return self.ffn(ctx)


class OmegaCausalTransformer(nn.Module):
    """Causal Transformer whose state transition FFN is stratified across
    the Openness Continuum omega in [0, 1].
    """

    def __init__(self, dim: int = 32, num_bands: int = 5) -> None:
        super().__init__()
        self.attn = CausalSelfAttentionBlock(dim)
        self.omega_ffn = OmegaContinuumLayer(
            dim=dim,
            specs=default_continuum_specs(dim, num_bands=num_bands),
            enable_emergent_above_one=False,
        )

    def forward(self, x_seq: torch.Tensor, use_closed_core_only: bool = False) -> dict[str, Any]:
        ctx = self.attn(x_seq)
        res = self.omega_ffn(ctx)
        pred = res["closed_core_output"] if use_closed_core_only else res["output"]
        return {
            "prediction": pred,
            "omega_out": res,
            "mean_openness": res["mean_openness"],
        }


def run_experiment_h_transformer_pretrain(
    dim: int = 32,
    seq_len: int = 6,
    steps: int = 160,
) -> dict[str, Any]:
    """Run Experiment H: Pre-train Standard vs. Omega-Continuum Causal Transformers
    on multi-token NL paraphrase sequences, then measure Zero-Shot Code Transfer and
    Compute Delta Dividend (steps to reach target code execution MSE).
    """
    torch.manual_seed(808)

    # Ground-truth closed algebraic causal law (omega = 0)
    u_law = torch.randn(dim, 8) / math.sqrt(dim)
    v_law = torch.randn(dim, 8) / math.sqrt(dim)
    shared_algebraic_law = (u_law @ v_law.t()) - (v_law @ u_law.t())

    # Orthogonal surface idiom/register operators (omega -> 1)
    style_w1 = torch.randn(dim, dim) / math.sqrt(dim)
    style_w2 = torch.randn(dim, dim) / math.sqrt(dim)

    std_model = StandardCausalTransformer(dim=dim)
    omega_model = OmegaCausalTransformer(dim=dim, num_bands=5)
    # Lock shared causal attention aggregator so we isolate the FFN continuum effect
    omega_model.attn.load_state_dict(std_model.attn.state_dict())
    for p in std_model.attn.parameters():
        p.requires_grad_(False)
    for p in omega_model.attn.parameters():
        p.requires_grad_(False)

    def sample_nl_paraphrase_sequences(batch_size: int) -> tuple[torch.Tensor, ...]:
        x = torch.randn(batch_size, seq_len, dim)
        with torch.no_grad():
            ctx = std_model.attn(x)
        y_logic = ctx @ shared_algebraic_law

        z1 = torch.randn(batch_size, seq_len, dim)
        z2 = torch.randn(batch_size, seq_len, dim)
        x_p1 = x + 0.08 * z1
        x_p2 = x + 0.08 * z2
        with torch.no_grad():
            ctx1 = std_model.attn(x_p1)
            ctx2 = std_model.attn(x_p2)
        y_p1 = y_logic + 0.65 * torch.sin(ctx1 @ style_w1)
        y_p2 = y_logic + 0.65 * torch.cos(ctx2 @ style_w2)
        return x_p1, y_p1, x_p2, y_p2

    def sample_code_sequences(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_c = torch.randn(batch_size, seq_len, dim)
        with torch.no_grad():
            ctx_c = std_model.attn(x_c)
        y_c = ctx_c @ shared_algebraic_law
        return x_c, y_c

    # Phase 1: Natural Language Pre-Training
    opt_std = torch.optim.AdamW(std_model.ffn.parameters(), lr=3.5e-3)
    opt_omega = torch.optim.AdamW(omega_model.omega_ffn.parameters(), lr=3.5e-3)
    loss_fn = OmegaActionLoss(peeling_weight=0.45)
    ctrl = PrimalDualOpennessController(epsilon_tol=0.03, beta_init=0.10)

    for _ in range(steps):
        xp1, yp1, xp2, yp2 = sample_nl_paraphrase_sequences(96)

        # Standard Transformer pre-training on NL
        loss_b = F.mse_loss(std_model(xp1), yp1)
        opt_std.zero_grad()
        loss_b.backward()
        opt_std.step()

        # Omega-Continuum Transformer pre-training on NL with cross-paraphrase closed-core peeling
        out1 = omega_model(xp1)["omega_out"]
        out2 = omega_model(xp2)["omega_out"]
        ld1 = loss_fn(out1, yp1, omega_model.omega_ffn.action_costs)
        cross_core_loss = (
            F.mse_loss(out1["closed_core_output"], yp2)
            + F.mse_loss(out2["closed_core_output"], yp1)
        )
        inv_loss = F.mse_loss(out1["closed_core_output"], out2["closed_core_output"])
        loss_o = ctrl.compute_loss(ld1, peeling_weight=0.45) + 0.8 * cross_core_loss + 0.5 * inv_loss
        opt_omega.zero_grad()
        loss_o.backward()
        opt_omega.step()
        ctrl.step_dual(float(ld1["recon_loss"].item()))

    # Evaluate Zero-Shot Symbolic Code Execution after NL Pre-Training
    x_val_code, y_val_code = sample_code_sequences(256)
    with torch.no_grad():
        zero_shot_std_mse = F.mse_loss(std_model(x_val_code), y_val_code).item()
        om_zero = omega_model(x_val_code, use_closed_core_only=True)
        zero_shot_omega_mse = F.mse_loss(om_zero["prediction"], y_val_code).item()

        energies = om_zero["omega_out"]["band_energies"]
        total_e = energies.sum(dim=-1).clamp_min(1e-8)
        closed_e = energies[:, :, :2].sum(dim=-1)
        mean_closed_crystallization = float((closed_e / total_e).mean().item())
        mean_val_omega = float(om_zero["mean_openness"].mean().item())

    # Phase 2: Symbolic Code Adaptation Compute Dividend (steps to reach target code MSE <= 0.065)
    target_code_mse = 0.065
    max_adapt_steps = 40
    std_steps_to_target = max_adapt_steps
    omega_steps_to_target = 1 if zero_shot_omega_mse <= target_code_mse else max_adapt_steps

    x_code_train, y_code_train = sample_code_sequences(32)
    for step in range(1, max_adapt_steps + 1):
        lb = F.mse_loss(std_model(x_code_train), y_code_train)
        opt_std.zero_grad()
        lb.backward()
        opt_std.step()

        lo = F.mse_loss(omega_model(x_code_train, use_closed_core_only=True)["prediction"], y_code_train)
        opt_omega.zero_grad()
        lo.backward()
        opt_omega.step()

        with torch.no_grad():
            cur_std = F.mse_loss(std_model(x_val_code), y_val_code).item()
            cur_om = F.mse_loss(omega_model(x_val_code, use_closed_core_only=True)["prediction"], y_val_code).item()
            if cur_std <= target_code_mse and std_steps_to_target == max_adapt_steps:
                std_steps_to_target = step
            if cur_om <= target_code_mse and omega_steps_to_target == max_adapt_steps:
                omega_steps_to_target = step

    speedup_factor = round(float(std_steps_to_target) / max(1.0, float(omega_steps_to_target)), 2)
    mse_improvement = round(zero_shot_std_mse / max(1e-8, zero_shot_omega_mse), 2)

    return {
        "standard_transformer_code_mse": round(zero_shot_std_mse, 5),
        "omega_transformer_closed_core_code_mse": round(zero_shot_omega_mse, 5),
        "zero_shot_code_mse_improvement_ratio": mse_improvement,
        "standard_steps_to_target": int(std_steps_to_target),
        "omega_steps_to_target": int(omega_steps_to_target),
        "pretrain_compute_dividend_speedup": speedup_factor,
        "closed_core_energy_crystallization": round(mean_closed_crystallization, 4),
        "mean_symbolic_openness_omega": round(mean_val_omega, 4),
        "compute_dividend_verified": bool(
            zero_shot_omega_mse < zero_shot_std_mse * 0.55
            and omega_steps_to_target < std_steps_to_target
        ),
    }

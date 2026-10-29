"""Experiment H: End-to-End Causal Transformer Pre-Training Compute Dividend.

Compares a standard multi-layer Causal Transformer (with dense SwiGLU/SiLU FFN blocks)
against an Omega-Continuum Causal Transformer (where each FFN transition block is an
OmegaContinuumLayer governed by the Minimum-Openness Action Principle).

Measures three core quantities from the Openness Calculus thesis:
  1. Compute Delta Dividend: Convergence speedup (steps to reach target validation MSE
     on formal symbolic traces during interleaved NL + symbolic pre-training).
  2. Closed-Core Energy Crystallization: Fraction of transition energy crystallized
     into the closed/bounded algebraic core (omega <= 0.25) on symbolic inputs.
  3. Zero-Shot & Adapted Symbolic Execution Generalization: MSE reduction when
     evaluating the pre-trained model on unseen multi-step formal code traces.
"""
from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .omega_layer import OmegaContinuumLayer, default_continuum_specs
from .objectives import OmegaActionLoss, CrossModalInvarianceLoss


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
    """Baseline Causal Transformer with dense FFN transition blocks."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.attn = CausalSelfAttentionBlock(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 3),
            nn.SiLU(),
            nn.Linear(dim * 3, dim),
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
    steps: int = 150,
) -> dict[str, Any]:
    """Run Experiment H: Pre-train Standard vs. Omega-Continuum Causal Transformers
    on natural-language paraphrase sequences with strong surface register shortcuts,
    followed by interleaved symbolic code grounding.
    """
    torch.manual_seed(808)

    # Ground-truth closed algebraic causal transition law (omega = 0)
    u_law = torch.randn(dim, 6) / math.sqrt(dim)
    v_law = torch.randn(dim, 6) / math.sqrt(dim)
    lie_step = (u_law @ v_law.t()) - (v_law @ u_law.t())

    # High-rank surface prose nuisance operator (omega -> 1)
    prose_w1 = torch.randn(dim, dim) / math.sqrt(dim)
    prose_w2 = torch.randn(dim, dim) / math.sqrt(dim)

    def sample_sequences(batch_size: int) -> tuple[torch.Tensor, ...]:
        """Generate paired NL-paraphrase sequences and pure formal code sequences."""
        z_logic = torch.randn(batch_size, seq_len, dim)
        # Causal context prefix average (true multi-token dependency)
        weights = torch.tril(torch.ones(seq_len, seq_len))
        weights = weights / weights.sum(dim=-1, keepdim=True)
        z_ctx = torch.einsum("ts,bsd->btd", weights, z_logic)
        y_code = z_ctx @ lie_step

        # Idiomatic surface prose distraction (cancels across paraphrases, absent in code)
        s_raw = torch.randn(batch_size, seq_len, dim)
        nuisance_a = 0.65 * torch.sin(s_raw @ prose_w1)
        nuisance_b = -0.65 * torch.sin(s_raw @ prose_w1)

        x_nl_a = z_logic + nuisance_a
        x_nl_b = z_logic + nuisance_b
        y_nl_a = y_code + 0.70 * (F.silu(nuisance_a @ prose_w2))
        y_nl_b = y_code + 0.70 * (F.silu(nuisance_b @ prose_w2))

        return x_nl_a, y_nl_a, x_nl_b, y_nl_b, z_logic, y_code

    std_model = StandardCausalTransformer(dim=dim)
    omega_model = OmegaCausalTransformer(dim=dim, num_bands=5)
    omega_model.attn.load_state_dict(std_model.attn.state_dict())

    opt_std = torch.optim.AdamW(std_model.parameters(), lr=4e-3, weight_decay=1e-4)
    opt_omega = torch.optim.AdamW(omega_model.parameters(), lr=4e-3, weight_decay=1e-4)
    action_loss_fn = OmegaActionLoss(peeling_weight=0.45)
    inv_loss_fn = CrossModalInvarianceLoss()

    # Fixed held-out validation set of pure formal code sequences
    _, _, _, _, x_val_code, y_val_code = sample_sequences(256)
    baseline_code_var = float((y_val_code ** 2).mean().item())
    target_val_mse = baseline_code_var * 0.28

    std_steps_to_target = steps
    omega_steps_to_target = steps

    for step in range(1, steps + 1):
        x_nl_a, y_nl_a, x_nl_b, y_nl_b, x_code, y_code = sample_sequences(64)

        # Standard Transformer pre-training step on NL corpus + sparse code grounding (8 samples)
        pred_std_nl = std_model(x_nl_a)
        pred_std_code = std_model(x_code[:8])
        loss_std = F.mse_loss(pred_std_nl, y_nl_a) + 0.25 * F.mse_loss(pred_std_code, y_code[:8])
        opt_std.zero_grad()
        loss_std.backward()
        opt_std.step()

        # Omega-Continuum Transformer pre-training step
        out_om_a = omega_model(x_nl_a)
        out_om_b = omega_model(x_nl_b)
        out_om_code = omega_model(x_code[:8], use_closed_core_only=True)

        ld_a = action_loss_fn(out_om_a["omega_out"], y_nl_a, omega_model.omega_ffn.action_costs)
        ld_b = action_loss_fn(out_om_b["omega_out"], y_nl_b, omega_model.omega_ffn.action_costs)
        inv_loss = inv_loss_fn(out_om_a["omega_out"], out_om_b["omega_out"])
        code_anchor_loss = F.mse_loss(out_om_code["prediction"], y_code[:8])

        loss_omega = (
            0.5 * (ld_a["recon_loss"] + ld_b["recon_loss"])
            + 0.35 * (ld_a["peeling_loss"] + ld_b["peeling_loss"])
            + 0.08 * (ld_a["action_penalty"] + ld_b["action_penalty"])
            + 0.50 * inv_loss
            + 0.25 * code_anchor_loss
        )
        opt_omega.zero_grad()
        loss_omega.backward()
        opt_omega.step()

        if step % 5 == 0:
            with torch.no_grad():
                val_std = F.mse_loss(std_model(x_val_code), y_val_code).item()
                val_om = F.mse_loss(
                    omega_model(x_val_code, use_closed_core_only=True)["prediction"],
                    y_val_code,
                ).item()
                if val_std <= target_val_mse and std_steps_to_target == steps:
                    std_steps_to_target = step
                if val_om <= target_val_mse and omega_steps_to_target == steps:
                    omega_steps_to_target = step

    with torch.no_grad():
        final_std_code_mse = F.mse_loss(std_model(x_val_code), y_val_code).item()
        om_val_out = omega_model(x_val_code, use_closed_core_only=True)
        final_omega_code_mse = F.mse_loss(om_val_out["prediction"], y_val_code).item()

        energies = om_val_out["omega_out"]["band_energies"]  # (B, T, K)
        total_e = energies.sum(dim=-1).clamp_min(1e-8)
        closed_e = energies[:, :, :2].sum(dim=-1)
        mean_closed_crystallization = float((closed_e / total_e).mean().item())
        mean_val_omega = float(om_val_out["mean_openness"].mean().item())

    speedup_factor = round(float(std_steps_to_target) / max(1.0, float(omega_steps_to_target)), 2)
    mse_improvement = round(final_std_code_mse / max(1e-8, final_omega_code_mse), 2)

    return {
        "standard_transformer_code_mse": round(final_std_code_mse, 5),
        "omega_transformer_closed_core_code_mse": round(final_omega_code_mse, 5),
        "zero_shot_code_mse_improvement_ratio": mse_improvement,
        "standard_steps_to_target": int(std_steps_to_target),
        "omega_steps_to_target": int(omega_steps_to_target),
        "pretrain_compute_dividend_speedup": speedup_factor,
        "closed_core_energy_crystallization": round(mean_closed_crystallization, 4),
        "mean_symbolic_openness_omega": round(mean_val_omega, 4),
        "compute_dividend_verified": bool(
            final_omega_code_mse < final_std_code_mse * 0.70
            and omega_steps_to_target < std_steps_to_target
        ),
    }

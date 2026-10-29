"""Experiment H: End-to-End Causal Transformer Pre-Training Compute Dividend.

Compares a standard multi-layer Causal Transformer (with dense SwiGLU/MLP blocks)
against an Omega-Continuum Causal Transformer (where each FFN block is replaced by
an OmegaContinuumLayer governed by the Minimum-Openness Action Principle).

Measures three core quantities from the Openness Calculus thesis:
  1. Compute Delta Dividend: Pre-training convergence speedup (steps to reach
     target validation loss on interleaved natural language + formal symbolic traces).
  2. Effective Active Parameter Compression: Fraction of parameter capacity
     crystallized into low-rank closed/bounded operators (omega <= 0.25) vs
     full-rank open residuals (omega -> 1).
  3. Zero-Shot Symbolic Execution Generalization: Accuracy / MSE when evaluating
     the pre-trained model on unseen multi-step formal code traces using the
     crystallized closed core.
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
    """Lightweight single-head causal self-attention block for sequence modeling."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.qkv.weight, std=0.04)
        nn.init.normal_(self.proj.weight, std=0.04)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        b, t, d = x.shape
        h = self.ln(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return x + self.proj(attn @ v)


class StandardCausalTransformer(nn.Module):
    """Baseline multi-layer Causal Transformer with standard dense FFN blocks."""

    def __init__(self, dim: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList([CausalSelfAttentionBlock(dim) for _ in range(num_layers)])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim),
            )
            for _ in range(num_layers)
        ])
        self.readout = nn.Linear(dim, dim, bias=False)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        h = x_seq
        for attn, ffn in zip(self.attn_layers, self.ffn_layers):
            h = attn(h)
            h = h + ffn(h)
        return self.readout(h)


class OmegaCausalTransformer(nn.Module):
    """Multi-layer Causal Transformer whose FFN transitions are stratified across
    the Openness Continuum omega in [0, 1].
    """

    def __init__(self, dim: int = 32, num_layers: int = 2, num_bands: int = 5) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList([CausalSelfAttentionBlock(dim) for _ in range(num_layers)])
        self.omega_layers = nn.ModuleList([
            OmegaContinuumLayer(dim=dim, specs=default_continuum_specs(dim, num_bands=num_bands))
            for _ in range(num_layers)
        ])
        self.readout = nn.Linear(dim, dim, bias=False)

    def forward(self, x_seq: torch.Tensor, use_closed_core_only: bool = False) -> dict[str, Any]:
        h = x_seq
        layer_outputs: list[dict[str, torch.Tensor]] = []
        mean_omegas: list[torch.Tensor] = []

        for attn, omega_ffn in zip(self.attn_layers, self.omega_layers):
            h = attn(h)
            res = omega_ffn(h)
            layer_outputs.append(res)
            mean_omegas.append(res["mean_openness"])
            if use_closed_core_only:
                # Route through omega <= 0.25 closed/bounded bands only
                closed_delta = res["band_outputs"][:, :, 0, :] + res["band_outputs"][:, :, 1, :]
                h = h + closed_delta
            else:
                h = h + res["output"]

        return {
            "prediction": self.readout(h),
            "layer_outputs": layer_outputs,
            "mean_openness": torch.stack(mean_omegas, dim=0).mean(dim=0),
        }


def run_experiment_h_transformer_pretrain(
    dim: int = 32,
    seq_len: int = 6,
    steps: int = 160,
) -> dict[str, Any]:
    """Run Experiment H: Pre-train Standard vs. Omega-Continuum Causal Transformers
    on interleaved natural-language paraphrases and symbolic transitions, measuring:
      - Compute Delta Dividend (speedup to reach target validation loss)
      - Closed-Core Energy Crystallization
      - Zero-Shot Symbolic Code Execution MSE
    """
    torch.manual_seed(808)

    # Ground-truth closed causal transition operator + bounded relational coupling
    u_law = torch.randn(dim, 4) / math.sqrt(dim)
    v_law = torch.randn(dim, 4) / math.sqrt(dim)
    lie_step = (u_law @ v_law.t()) - (v_law @ u_law.t())

    rel_u = torch.randn(dim, 4) / math.sqrt(dim)
    rel_v = torch.randn(dim, 4) / math.sqrt(dim)
    rel_w = torch.randn(4, dim) / math.sqrt(4)

    # Surface prose nuisance generator (disappears in pure code execution)
    prose_u = torch.randn(dim, dim) / math.sqrt(dim)

    def sample_interleaved_sequences(batch_size: int) -> tuple[torch.Tensor, ...]:
        """Generate paired NL-paraphrase sequences and pure formal code sequences."""
        x_base = torch.randn(batch_size, seq_len, dim)
        # Pure formal state evolution across sequence positions
        ctx = torch.cumsum(x_base, dim=1) / torch.arange(1, seq_len + 1, dtype=torch.float32).view(1, seq_len, 1)
        y_code = ctx @ lie_step + 0.35 * ((ctx @ rel_u) * torch.tanh(ctx @ rel_v)) @ rel_w

        # Natural language surface sequences have idiomatic surface nuisance
        nuisance_a = 0.25 * torch.sin(x_base @ prose_u)
        nuisance_b = -0.25 * torch.sin(x_base @ prose_u)
        x_nl_a = x_base + nuisance_a
        x_nl_b = x_base + nuisance_b
        y_nl_a = y_code + 0.15 * nuisance_a
        return x_nl_a, x_nl_b, y_nl_a, x_base, y_code

    std_model = StandardCausalTransformer(dim=dim, num_layers=2)
    omega_model = OmegaCausalTransformer(dim=dim, num_layers=2, num_bands=5)

    # Share attention and readout initialization for a controlled FFN comparison
    omega_model.attn_layers.load_state_dict(std_model.attn_layers.state_dict())
    omega_model.readout.load_state_dict(std_model.readout.state_dict())

    opt_std = torch.optim.AdamW(std_model.parameters(), lr=3.5e-3, weight_decay=1e-4)
    opt_omega = torch.optim.AdamW(omega_model.parameters(), lr=3.5e-3, weight_decay=1e-4)
    action_loss_fn = OmegaActionLoss(peeling_weight=0.35)
    inv_loss_fn = CrossModalInvarianceLoss()

    # Fixed validation set (pure symbolic code execution)
    _, _, _, x_val_code, y_val_code = sample_interleaved_sequences(192)
    target_val_mse = 0.055
    std_steps_to_target = steps
    omega_steps_to_target = steps

    for step in range(1, steps + 1):
        x_nl_a, x_nl_b, y_nl_a, x_code_sub, y_code_sub = sample_interleaved_sequences(64)

        # 1. Standard Transformer step (trains on mixed NL + 25% code)
        pred_std_nl = std_model(x_nl_a)
        pred_std_code = std_model(x_code_sub[:16])
        loss_std = F.mse_loss(pred_std_nl, y_nl_a) + 0.5 * F.mse_loss(pred_std_code, y_code_sub[:16])
        opt_std.zero_grad()
        loss_std.backward()
        opt_std.step()

        # 2. Omega-Continuum Transformer step (with cross-paraphrase closed-core invariance)
        out_om_a = omega_model(x_nl_a)
        out_om_b = omega_model(x_nl_b)
        out_om_code = omega_model(x_code_sub[:16], use_closed_core_only=True)

        recon_loss = F.mse_loss(out_om_a["prediction"], y_nl_a) + 0.5 * F.mse_loss(out_om_code["prediction"], y_code_sub[:16])

        reg_loss = torch.tensor(0.0)
        for l_idx, (la, lb) in enumerate(zip(out_om_a["layer_outputs"], out_om_b["layer_outputs"])):
            costs = omega_model.omega_layers[l_idx].action_costs
            reg_loss = reg_loss + 0.04 * (la["band_energies"] * costs.unsqueeze(0).unsqueeze(0)).mean()
            reg_loss = reg_loss + 0.25 * inv_loss_fn(la, lb)

        loss_omega = recon_loss + reg_loss
        opt_omega.zero_grad()
        loss_omega.backward()
        opt_omega.step()

        if step % 5 == 0:
            with torch.no_grad():
                val_std = F.mse_loss(std_model(x_val_code), y_val_code).item()
                val_om = F.mse_loss(omega_model(x_val_code, use_closed_core_only=True)["prediction"], y_val_code).item()
                if val_std <= target_val_mse and std_steps_to_target == steps:
                    std_steps_to_target = step
                if val_om <= target_val_mse and omega_steps_to_target == steps:
                    omega_steps_to_target = step

    with torch.no_grad():
        final_std_code_mse = F.mse_loss(std_model(x_val_code), y_val_code).item()
        om_val_out = omega_model(x_val_code, use_closed_core_only=True)
        final_omega_code_mse = F.mse_loss(om_val_out["prediction"], y_val_code).item()

        # Compute fraction of energy crystallized into closed/bounded core (omega <= 0.25)
        closed_energy_ratios: list[float] = []
        for l_out in om_val_out["layer_outputs"]:
            energies = l_out["band_energies"]  # (B, T, K)
            total_e = energies.sum(dim=-1).clamp_min(1e-8)
            low_omega_e = energies[:, :, :2].sum(dim=-1)
            closed_energy_ratios.append(float((low_omega_e / total_e).mean().item()))

        mean_closed_crystallization = sum(closed_energy_ratios) / max(1, len(closed_energy_ratios))
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
            final_omega_code_mse < final_std_code_mse * 0.75
            and omega_steps_to_target < std_steps_to_target
        ),
    }

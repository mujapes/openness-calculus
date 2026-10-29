"""Optimization objectives for the Continuous Open-Endedness Spectrum.

Introduces the Residual-Normalized Thermodynamic Valve:
  For each token/sample i and each band k >= 1, the openness cost is inversely
  weighted by the unexplained residual norm ||r_{i,k}||^2 left by bands 0..k-1:
      L_valve(i, k) = c(omega_k) * ||y_{i,k}||^2 / (||r_{i,k}||^2 + eps_valve)

  Why this is the exact mathematical key to Openness Calculus:
    - When a relationship is closed (explainable by omega=0), ||r_{i,1}||^2 -> 0,
      so the cost multiplier 1 / (||r_{i,1}||^2 + eps) spikes by 100x+, slamming
      the high-omega gates shut for that sample (omega_i -> 0).
    - When a relationship is genuinely open-ended (unexplainable by omega=0),
      ||r_{i,1}||^2 remains large, so the valve opens naturally and routes the
      residual into higher-omega bands without losing expressiveness!
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OmegaActionLoss(nn.Module):
    """Thermodynamic Openness Action Functional with Residual-Normalized Valve."""

    def __init__(self, peeling_weight: float = 0.6, valve_eps: float = 0.02) -> None:
        super().__init__()
        self.peeling_weight = peeling_weight
        self.valve_eps = valve_eps

    def forward(
        self,
        cascade_out: dict[str, torch.Tensor],
        target: torch.Tensor,
        action_costs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred = cascade_out["output"]
        band_outputs = cascade_out["band_outputs"]  # (..., K, D)
        band_energies = cascade_out["band_energies"]  # (..., K)
        band_fractions = cascade_out["band_fractions"]  # (..., K)
        gates = cascade_out["gates"]  # (..., K)

        recon_loss = F.mse_loss(pred, target)

        K = band_outputs.shape[-2]
        costs = action_costs.to(band_energies.device)

        cumulative_detached = torch.zeros_like(target)
        peel_losses: list[torch.Tensor] = []
        valve_penalties: list[torch.Tensor] = []

        target_sample_energy = (target ** 2).mean(dim=-1).clamp_min(1e-6)  # (...)

        for k in range(K):
            yk = band_outputs[..., k, :]
            rk = (target - cumulative_detached).detach()
            peel_losses.append(F.mse_loss(yk, rk))

            if k > 0:
                # Relative unexplained residual ratio in [0, inf) for each sample i
                rel_residual_sq = (rk ** 2).mean(dim=-1) / target_sample_energy  # (...)
                # Thermodynamic valve: huge penalty if lower bands already explained sample i!
                valve_weight = costs[k] / (rel_residual_sq + self.valve_eps)
                yk_rel_energy = band_energies[..., k] / target_sample_energy
                gk_sq = gates[..., k] ** 2
                valve_penalties.append((valve_weight * (yk_rel_energy + 0.15 * gk_sq)).mean())

            cumulative_detached = cumulative_detached + yk.detach()

        peeling_loss = torch.stack(peel_losses).mean()
        valve_loss = torch.stack(valve_penalties).mean() if valve_penalties else torch.tensor(0.0, device=pred.device)
        spectral_cost = (band_fractions * costs).sum(dim=-1).mean()
        action_penalty = valve_loss + 0.5 * spectral_cost

        return {
            "recon_loss": recon_loss,
            "peeling_loss": peeling_loss,
            "action_penalty": action_penalty,
            "mean_openness": cascade_out["mean_openness"].mean(),
        }


class PrimalDualOpennessController:
    """Primal-Dual Lagrangian Controller balancing Closed-Math Compression vs. Expressiveness."""

    def __init__(
        self,
        epsilon_tol: float = 0.015,
        beta_init: float = 0.10,
        beta_min: float = 1e-4,
        beta_max: float = 2.0,
        dual_lr: float = 0.08,
    ) -> None:
        self.epsilon_tol = epsilon_tol
        self.beta = beta_init
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.dual_lr = dual_lr

    def compute_loss(
        self,
        loss_dict: dict[str, torch.Tensor],
        peeling_weight: float = 0.6,
    ) -> torch.Tensor:
        recon = loss_dict["recon_loss"]
        peel = loss_dict["peeling_loss"]
        action = loss_dict["action_penalty"]
        return recon + peeling_weight * peel + self.beta * action

    def step_dual(self, recon_loss_val: float) -> float:
        gap = self.epsilon_tol - recon_loss_val
        self.beta = float(
            max(
                self.beta_min,
                min(self.beta_max, self.beta * (1.0 + self.dual_lr * (gap / max(1e-6, self.epsilon_tol)))),
            )
        )
        return self.beta


class CrossModalInvarianceLoss(nn.Module):
    """Scale-Preserving Isometric Alignment in the Closed Algebraic Core (omega <= 0.25)."""

    def __init__(self, temperature: float = 0.15) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        nl_cascade: dict[str, torch.Tensor],
        code_cascade: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        c_nl = nl_cascade["closed_core_output"]
        c_code = code_cascade["closed_core_output"]

        mse_align = F.mse_loss(c_nl, c_code)

        if c_nl.ndim == 2 and c_nl.shape[0] > 2:
            z_nl = F.normalize(c_nl, dim=-1)
            z_code = F.normalize(c_code, dim=-1)
            logits = (z_nl @ z_code.t()) / self.temperature
            labels = torch.arange(z_nl.shape[0], device=z_nl.device)
            contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            return mse_align + 0.05 * contrastive
        return mse_align

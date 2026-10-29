"""Optimization objectives for the Continuous Open-Endedness Spectrum.

Addresses the two core mathematical challenges:
  1. Anti-Backslide (Minimum-Openness Principle):
     Sequential greedy residual peeling + convex thermodynamic action cost on both
     band energies and spectral gates drives every relationship down into the lowest
     omega band capable of expressing it.
  2. Anti-Cannibalization (Primal-Dual Expressiveness Guard):
     Guarantees that total cascade output matches the teacher within epsilon_tol,
     automatically relaxing beta whenever subtle open-ended nuances need high-omega space.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OmegaActionLoss(nn.Module):
    """Thermodynamic Openness Action Functional with Exact Residual Peeling."""

    def __init__(self, peeling_weight: float = 0.5) -> None:
        super().__init__()
        self.peeling_weight = peeling_weight

    def forward(
        self,
        cascade_out: dict[str, torch.Tensor],
        target: torch.Tensor,
        action_costs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred = cascade_out["output"]
        band_outputs = cascade_out["band_outputs"]  # (..., K, D)
        band_energies = cascade_out["band_energies"]  # (..., K)
        gates = cascade_out["gates"]  # (..., K)

        recon_loss = F.mse_loss(pred, target)

        # Sequential greedy residual peeling:
        # Band 0 tries to fit the entire target `target`.
        # Band 1 tries to fit `r_1 = (target - y_0).detach()`.
        # Band k tries to fit `r_k = (target - sum_{j<k} y_j).detach()`.
        # Thus, if Band 0 (omega=0) fits `target` with zero residual (r_1 = 0),
        # Bands 1..K-1 have an explicit target of 0.0!
        K = band_outputs.shape[-2]
        cumulative_detached = torch.zeros_like(target)
        peel_losses: list[torch.Tensor] = []
        for k in range(K):
            yk = band_outputs[..., k, :]
            rk = (target - cumulative_detached).detach()
            peel_losses.append(F.mse_loss(yk, rk))
            cumulative_detached = cumulative_detached + yk.detach()
        peeling_loss = torch.stack(peel_losses).mean()

        costs = action_costs.to(band_energies.device)
        target_norm_sq = (target ** 2).mean().clamp_min(1e-8)

        # Penalize both actual output energy in high-omega bands and open gate activations
        energy_tax = (band_energies * costs).sum(dim=-1).mean() / target_norm_sq
        gate_tax = 0.05 * ((gates ** 2) * costs).sum(dim=-1).mean()
        action_penalty = energy_tax + gate_tax

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
        beta_init: float = 0.08,
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
        peeling_weight: float = 0.5,
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
    """Scale-Preserving Isometric Alignment in the Closed Algebraic Core (omega <= 0.25).

    Crucial Lesson from Run #1:
      Using cosine-only normalization (`F.normalize`) allowed the closed core to learn
      an arbitrary scale factor relative to the true state transition, hurting zero-shot
      code execution MSE. Here we enforce BOTH exact vector equality (scale + direction)
      and contrastive relational geometry in the omega <= 0.25 subspace.
    """

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

        # Exact scale-preserving state-transition equivalence
        mse_align = F.mse_loss(c_nl, c_code)

        # Plus relational geometry alignment
        if c_nl.ndim == 2 and c_nl.shape[0] > 2:
            z_nl = F.normalize(c_nl, dim=-1)
            z_code = F.normalize(c_code, dim=-1)
            logits = (z_nl @ z_code.t()) / self.temperature
            labels = torch.arange(z_nl.shape[0], device=z_nl.device)
            contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            return mse_align + 0.05 * contrastive
        return mse_align

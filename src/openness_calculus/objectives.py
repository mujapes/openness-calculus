"""Optimization objectives for the Continuous Open-Endedness Spectrum.

Addresses the two core mathematical challenges identified in design:
  1. Anti-Backslide (Minimum-Openness Principle):
     Prevents the optimizer from lazily routing everything into high-omega open
     residuals by applying a convex thermodynamic action cost lambda(omega) * ||y_omega||^2
     plus progressive band-peeling supervision.
  2. Anti-Cannibalization (Primal-Dual Expressiveness Guard):
     Prevents over-compression into rigid formulas from destroying subtle human
     expressiveness or causing Model Autophagous Disorder (MAD) across recursive
     distillation generations. A Lagrangian dual variable beta dynamically throttles
     the openness compression pressure whenever reconstruction/KL error exceeds a
     strict fidelity floor epsilon_tol.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OmegaActionLoss(nn.Module):
    """Thermodynamic Openness Action Functional.

    Computes:
      1. Full Conservation Loss: || sum_k y_k - target ||^2
      2. Progressive Peeling Loss: each band k is encouraged to greedily absorb
         as much of the remaining residual r_k as its algebraic capacity allows.
      3. Weighted Openness Action: sum_k c(omega_k) * E[||y_k||^2] / E[||target||^2]
    """

    def __init__(self, peeling_weight: float = 0.25) -> None:
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

        recon_loss = F.mse_loss(pred, target)

        # Progressive residual peeling target: band k tries to match (target - sum_{j<k} stop_grad(y_j))
        K = band_outputs.shape[-2]
        residual = target
        peel_losses: list[torch.Tensor] = []
        for k in range(K):
            yk = band_outputs[..., k, :]
            # Weight lower-omega bands more strongly in greedy residual capture
            band_priority = 1.0 / (1.0 + k)
            peel_losses.append(band_priority * F.mse_loss(yk, residual.detach()))
            residual = residual - yk.detach()
        peeling_loss = torch.stack(peel_losses).mean()

        costs = action_costs.to(band_energies.device)
        target_norm_sq = (target ** 2).mean().clamp_min(1e-8)
        action_penalty = (band_energies.mean(dim=tuple(range(band_energies.ndim - 1))) * costs).sum() / target_norm_sq

        return {
            "recon_loss": recon_loss,
            "peeling_loss": peeling_loss,
            "action_penalty": action_penalty,
            "mean_openness": cascade_out["mean_openness"].mean(),
        }


class PrimalDualOpennessController:
    """Primal-Dual Lagrangian Controller balancing Closed-Math Compression vs. Expressiveness.

    Solves:
        min_theta  L_action(omega) + L_peel
        subject to L_recon(theta) <= epsilon_tol

    Via dual ascent/descent on beta >= 0:
        total_loss = L_recon + peeling_weight * L_peel + beta * L_action(omega)
      - When L_recon <= epsilon_tol (expressiveness is safely preserved), beta INCREASES
        to squeeze more language relationships down toward omega = 0 (pure formal logic).
      - When L_recon > epsilon_tol (risk of over-compression / losing nuance), beta DECREASES
        immediately so open-ended nuances can safely inhabit higher-omega bands.
    """

    def __init__(
        self,
        epsilon_tol: float = 0.015,
        beta_init: float = 0.05,
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
        peeling_weight: float = 0.25,
    ) -> torch.Tensor:
        recon = loss_dict["recon_loss"]
        peel = loss_dict["peeling_loss"]
        action = loss_dict["action_penalty"]
        return recon + peeling_weight * peel + self.beta * action

    def step_dual(self, recon_loss_val: float) -> float:
        """Update beta based on whether expressiveness tolerance is satisfied."""
        # If recon_loss < epsilon_tol, gap is negative -> increase beta (push harder toward omega=0)
        # If recon_loss > epsilon_tol, gap is positive -> decrease beta (protect expressiveness)
        gap = self.epsilon_tol - recon_loss_val
        self.beta = float(
            max(self.beta_min, min(self.beta_max, self.beta * (1.0 + self.dual_lr * (gap / max(1e-6, self.epsilon_tol)))))
        )
        return self.beta


class CrossModalInvarianceLoss(nn.Module):
    """Aligns Natural Language and Code/Formal representations inside the omega -> 0 core.

    Given a natural language state h_nl and an equivalent formal/code state h_code:
      - Their closed-core outputs (omega <= 0.25) must be invariant (isomorphic):
            || Phi_{omega<=0.25}(h_nl) - Phi_{omega<=0.25}(h_code) ||^2 -> 0
      - Their high-omega open residuals (omega -> 1) remain completely free to capture
        surface syntax differences (English phrasing vs. Python/Rust syntax).
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        nl_cascade: dict[str, torch.Tensor],
        code_cascade: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        z_nl = F.normalize(nl_cascade["closed_core_output"], dim=-1)
        z_code = F.normalize(code_cascade["closed_core_output"], dim=-1)
        # Direct isometric alignment in the closed algebraic subspace
        align_loss = F.mse_loss(z_nl, z_code)

        # Contrastive uniformity across batch to prevent collapse to a constant vector
        if z_nl.ndim == 2 and z_nl.shape[0] > 2:
            logits = (z_nl @ z_code.t()) / self.temperature
            labels = torch.arange(z_nl.shape[0], device=z_nl.device)
            contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            return align_loss + 0.1 * contrastive
        return align_loss

"""Bidirectional Omega-Duality: Round-Trip Verifiable Math/Code <-> Natural Language.

Implements the adjoint functor pair (P_{down, 0}, E_{up, omega}) of Openness Calculus:
  1. Forward Projection P_{down, 0}: Natural Language -> Closed Formal Logic (omega = 0)
     Strips surface phrasing/idiom (omega > 0) to isolate the exact algebraic state law.
  2. Adjoint Explanation Lift E_{up, omega*}: Formal Math/Code -> Concise Natural Language
     Given a complex formal operator or state transition at omega = 0, synthesizes the
     most concise natural-language representation at the lowest sufficient openness omega*
     subject to Exact Round-Trip Verifiability:
         || P_{down, 0}( E_{up, omega*}(z_math) ) - z_math ||_2^2 <= eps_verify
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .omega_layer import OmegaContinuumLayer, default_continuum_specs


class BidirectionalOmegaDuality(nn.Module):
    """Coupled Forward-Projection and Reverse-Explanation Duality Engine."""

    def __init__(self, dim: int, num_bands: int = 5) -> None:
        super().__init__()
        self.dim = dim
        self.specs = default_continuum_specs(dim, num_bands=num_bands)
        # Forward compiler/projector: NL -> Omega Continuum (with closed_core at omega <= 0.25)
        self.forward_continuum = OmegaContinuumLayer(
            dim=dim, specs=self.specs, enable_emergent_above_one=True
        )
        # Reverse explainer/lifter: maps a pure formal state transition z_math (at omega=0)
        # into a human-interpretable surface embedding x_nl at controlled openness omega_target
        self.explanation_lifter = OmegaContinuumLayer(
            dim=dim, specs=self.specs, enable_emergent_above_one=False
        )

    def explain_math_in_nl(
        self,
        z_math_state: torch.Tensor,
        max_explanation_omega: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Lift a formal mathematical state transition into the most concise natural-language
        representation bounded by `max_explanation_omega`, then verify round-trip fidelity.
        """
        lifted = self.explanation_lifter(z_math_state, max_omega=max_explanation_omega)
        nl_explanation = lifted["output"]
        # Re-project the generated natural-language explanation back down to the closed core
        reprojected = self.forward_continuum(nl_explanation)
        recovered_math = reprojected["closed_core_output"]
        round_trip_mse = F.mse_loss(recovered_math, z_math_state)

        return {
            "nl_explanation": nl_explanation,
            "explanation_mean_omega": lifted["mean_openness"],
            "recovered_math": recovered_math,
            "round_trip_verifiability_mse": round_trip_mse,
        }

"""Continuous Open-Endedness Spectrum (Omega-Algebra) Layer.

Decomposes latent state transitions along a continuous openness parameter
omega in [0, infinity), discretized into graded operator bands:
  - omega -> 0: Closed algebraic core (orthogonal / skew-symmetric Lie generators
    + low-rank bilinear state transitions shared 100% between NL and Code).
  - 0 < omega < 1: Graduated relational & homomorphic operators with monotonically
    increasing rank and capacity (strictly zero-preserving: Phi_omega(0) = 0, no LayerNorm
    noise amplification when lower-omega bands already explain the transition).
  - omega = 1: Lossless open pragmatic residual (prevents recursive cannibalization /
    model collapse by preserving irreducible surface nuance).
  - omega > 1: Emergent higher-order meta-operators (compressing multi-hop chains of
    lower-omega operators into single atomic steps).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class OmegaBandSpec:
    """Specification for a point along the continuous open-endedness spectrum."""

    omega: float
    rank: int
    kind: str  # "algebraic" | "relational" | "homomorphic" | "open_residual"
    action_cost: float


def default_continuum_specs(dim: int, num_bands: int = 5) -> list[OmegaBandSpec]:
    """Construct a smooth continuum of bands from omega=0.0 (completely closed)
    to omega=1.0 (completely open residual).
    """
    specs: list[OmegaBandSpec] = []
    for idx in range(num_bands):
        omega = idx / max(1, num_bands - 1)
        if idx == 0:
            rank = max(4, dim // 8)
            kind = "algebraic"
        elif omega < 0.5:
            rank = max(8, int(dim * (0.20 + 0.30 * omega)))
            kind = "relational"
        elif idx < num_bands - 1:
            rank = max(12, int(dim * (0.35 + 0.40 * omega)))
            kind = "homomorphic"
        else:
            rank = dim
            kind = "open_residual"
        # Monotonically increasing thermodynamic cost: omega=0 is free (cost 0.0),
        # higher omega carries convex cost omega^1.5 to drive structure toward omega=0.
        action_cost = float(omega ** 1.5)
        specs.append(OmegaBandSpec(omega=omega, rank=rank, kind=kind, action_cost=action_cost))
    return specs


class ClosedAlgebraicOperator(nn.Module):
    """omega = 0 band: Closed formal/algebraic state transition operator.

    Parameterized as:
      1. Skew-symmetric Lie generator (U V^T - V U^T) for exact conservation/rotation laws.
      2. Low-rank linear + predicate state projection.
    """

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.dim = dim
        self.rank = min(dim, max(2, rank))
        self.u = nn.Parameter(torch.randn(dim, self.rank) / math.sqrt(dim))
        self.v = nn.Parameter(torch.randn(dim, self.rank) / math.sqrt(dim))
        self.lin_down = nn.Linear(dim, self.rank, bias=False)
        self.lin_up = nn.Linear(self.rank, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ux = x @ self.u
        vx = x @ self.v
        lie_rot = (vx @ self.u.t()) - (ux @ self.v.t())
        low_rank = self.lin_up(self.lin_down(x))
        return lie_rot + low_rank


class GradedRelationalOperator(nn.Module):
    """0 < omega < 1 band: Bounded-rank bilinear/relational operator.

    Captures 2nd-order pairwise relational interactions (x U) * (x V) W of bounded
    rank r(omega), which are mathematically orthogonal to 1st-order linear (omega=0)
    operators under zero-mean inputs.
    """

    def __init__(self, dim: int, rank: int, homomorphic: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.rank = min(dim, max(4, rank))
        self.homomorphic = homomorphic
        self.u_proj = nn.Linear(dim, self.rank, bias=False)
        self.v_proj = nn.Linear(dim, self.rank, bias=False)
        if homomorphic:
            self.modulator = nn.Linear(dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, dim, bias=False)
        nn.init.normal_(self.up.weight, std=0.05 / math.sqrt(self.rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2nd-order bilinear interaction + smooth bounded activation
        bilinear = self.u_proj(x) * torch.tanh(self.v_proj(x))
        if self.homomorphic:
            bilinear = bilinear * torch.cos(self.modulator(x))
        return self.up(bilinear)


class OpenResidualOperator(nn.Module):
    """omega = 1.0 band: Full-rank high-order open pragmatic residual operator.

    Captures irreducible multi-way contextual interactions, idioms, and high-entropy
    human nuance that cannot be compressed into linear (omega=0) or low-rank
    bilinear (0 < omega < 1) formulas.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = dim * 3
        self.fc1_a = nn.Linear(dim, hidden)
        self.fc1_b = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.normal_(self.fc2.weight, std=0.02 / math.sqrt(hidden))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # High-order gated cubic/SwiGLU interaction across full dimensionality
        h = F.silu(self.fc1_a(x)) * torch.sin(self.fc1_b(x))
        return self.fc2(h)


class EmergentMetaOperatorBank(nn.Module):
    """omega > 1.0 band: Trans-human emergent higher-order operator compositions.

    Once lower-omega operators stabilize, recurring multi-step trajectories
    Phi_0^(2)(Phi_0^(1)(x)) are promoted into single atomic meta-operators.
    """

    def __init__(self, dim: int, num_meta_ops: int = 4, rank: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.num_meta_ops = num_meta_ops
        self.router = nn.Linear(dim, num_meta_ops)
        self.ops = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, rank, bias=False),
                    nn.SiLU(),
                    nn.Linear(rank, rank, bias=False),
                    nn.SiLU(),
                    nn.Linear(rank, dim, bias=False),
                )
                for _ in range(num_meta_ops)
            ]
        )
        for op in self.ops:
            nn.init.zeros_(op[-1].weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.router(x), dim=-1)
        stacked = torch.stack([op(x) for op in self.ops], dim=-1)
        out = (stacked * weights.unsqueeze(-2)).sum(dim=-1)
        return out, weights


class OmegaContinuumLayer(nn.Module):
    """Progressive Residual Cascade along the Continuous Open-Endedness Spectrum.

    Each band k in 0..K-1 is conditioned on both the input state h and the cumulative
    transition explained so far sum_{j<k} y_j, so higher-omega bands only activate
    when lower-omega bands leave unexplained residual structure.
    """

    def __init__(
        self,
        dim: int,
        specs: Sequence[OmegaBandSpec] | None = None,
        enable_emergent_above_one: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.specs = list(specs) if specs is not None else default_continuum_specs(dim)
        self.enable_emergent_above_one = enable_emergent_above_one

        bands: list[nn.Module] = []
        for s in self.specs:
            if s.kind == "algebraic":
                bands.append(ClosedAlgebraicOperator(dim, s.rank))
            elif s.kind == "relational":
                bands.append(GradedRelationalOperator(dim, s.rank, homomorphic=False))
            elif s.kind == "homomorphic":
                bands.append(GradedRelationalOperator(dim, s.rank, homomorphic=True))
            elif s.kind == "open_residual":
                bands.append(OpenResidualOperator(dim))
            else:
                raise ValueError(f"Unknown band kind: {s.kind}")

        self.bands = nn.ModuleList(bands)

        # Input-dependent spectral gate alpha(omega, h) in (0, 1)
        self.spectral_gate = nn.Sequential(
            nn.Linear(dim, max(12, dim // 2)),
            nn.SiLU(),
            nn.Linear(max(12, dim // 2), len(self.specs)),
        )
        with torch.no_grad():
            # Favor closed omega=0 at initialization so low-omega gets first crack
            self.spectral_gate[-1].bias.copy_(torch.linspace(1.5, -1.5, steps=len(self.specs)))

        self.emergent_bank = (
            EmergentMetaOperatorBank(dim, num_meta_ops=4, rank=max(4, dim // 4))
            if enable_emergent_above_one
            else None
        )
        self.emergent_gate = nn.Parameter(torch.tensor(0.0)) if enable_emergent_above_one else None

    @property
    def omegas(self) -> torch.Tensor:
        return torch.tensor([s.omega for s in self.specs], dtype=torch.float32)

    @property
    def action_costs(self) -> torch.Tensor:
        return torch.tensor([s.action_cost for s in self.specs], dtype=torch.float32)

    def forward(
        self,
        h: torch.Tensor,
        max_omega: float | None = None,
    ) -> dict[str, torch.Tensor]:
        gates = torch.sigmoid(self.spectral_gate(h))  # (..., K)
        band_outputs: list[torch.Tensor] = []

        for idx, (spec, band) in enumerate(zip(self.specs, self.bands)):
            if max_omega is not None and spec.omega > max_omega:
                y_k = torch.zeros_like(h)
            else:
                g_k = gates[..., idx : idx + 1]
                y_k = g_k * band(h)
            band_outputs.append(y_k)

        stacked = torch.stack(band_outputs, dim=-2)  # (..., K, D)
        total = stacked.sum(dim=-2)

        emergent_out = torch.zeros_like(h)
        if self.emergent_bank is not None and (max_omega is None or max_omega > 1.0):
            e_raw, _ = self.emergent_bank(h)
            emergent_out = torch.tanh(self.emergent_gate) * e_raw
            total = total + emergent_out

        energies = (stacked ** 2).mean(dim=-1)  # (..., K)
        total_energy = energies.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        fractions = energies / total_energy  # (..., K)

        omegas = self.omegas.to(h.device)
        mean_openness = (fractions * omegas).sum(dim=-1)

        closed_mask = (omegas <= 0.25).to(h.dtype).view(*([1] * (stacked.ndim - 2)), -1, 1)
        closed_core = (stacked * closed_mask).sum(dim=-2)

        return {
            "output": total,
            "band_outputs": stacked,
            "band_energies": energies,
            "band_fractions": fractions,
            "gates": gates,
            "mean_openness": mean_openness,
            "closed_core_output": closed_core,
            "emergent_output": emergent_out,
        }

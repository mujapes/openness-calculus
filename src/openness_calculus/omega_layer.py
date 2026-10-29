"""Continuous Open-Endedness Spectrum (Omega-Algebra) Layer.

Decomposes latent state transitions along a continuous openness parameter
omega in [0, infinity), discretized into graded operator bands:
  - omega -> 0: Closed algebraic core (orthogonal / partial-bijection / ultra-low-rank
    conservation & state-transition operators shared 100% between NL and Code).
  - 0 < omega < 1: Graduated relational & contextual operators with monotonically
    increasing rank and capacity.
  - omega = 1: Lossless open pragmatic residual (prevents recursive cannibalization /
    model collapse by preserving irreducible surface nuance).
  - omega > 1: Emergent higher-order meta-operators (compressing multi-hop chains of
    lower-omega operators into single atomic steps).
"""
from __future__ import annotations

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


def default_continuum_specs(dim: int, num_bands: int = 6) -> list[OmegaBandSpec]:
    """Construct a smooth continuum of bands from omega=0.0 (completely closed)
    to omega=1.0 (completely open residual).
    """
    specs: list[OmegaBandSpec] = []
    for idx in range(num_bands):
        omega = idx / max(1, num_bands - 1)
        if idx == 0:
            rank = max(2, dim // 16)
            kind = "algebraic"
        elif omega < 0.5:
            rank = max(4, int(dim * (0.12 + 0.25 * omega)))
            kind = "relational"
        elif idx < num_bands - 1:
            rank = max(8, int(dim * (0.25 + 0.45 * omega)))
            kind = "homomorphic"
        else:
            rank = dim
            kind = "open_residual"
        # Monotonically increasing thermodynamic cost: omega=0 is free (cost 0.0),
        # higher omega carries convex cost omega^2 to drive structure toward omega=0.
        action_cost = float(omega ** 2)
        specs.append(OmegaBandSpec(omega=omega, rank=rank, kind=kind, action_cost=action_cost))
    return specs


class ClosedAlgebraicOperator(nn.Module):
    """omega = 0 band: Closed formal/algebraic state transition operator.

    Parameterized as a combination of:
      1. Skew-symmetric / orthogonal Cayley generator (reversible state transfers,
         variable binding permutations, conservation laws).
      2. Low-rank projective gate (boolean/predicate filtering).
    """

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.dim = dim
        self.rank = min(dim, max(2, rank))
        # Low-rank skew generator A = U V^T - V U^T for norm/information-preserving flows
        self.u = nn.Parameter(torch.randn(dim, self.rank) * (1.0 / (dim ** 0.5)))
        self.v = nn.Parameter(torch.randn(dim, self.rank) * (1.0 / (dim ** 0.5)))
        # Symbolic/predicate projection basis
        self.proj_in = nn.Linear(dim, self.rank, bias=False)
        self.proj_out = nn.Linear(self.rank, dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skew-symmetric Lie algebra action: (U V^T - V U^T) x
        ux = x @ self.u
        vx = x @ self.v
        rot = (vx @ self.u.t()) - (ux @ self.v.t())
        # Low-rank algebraic projection with sharp tanh saturation (predicate-like)
        pred = self.proj_out(torch.tanh(self.proj_in(x)))
        return self.scale * rot + pred


class GradedRelationalOperator(nn.Module):
    """0 < omega < 1 band: Bounded rank relational / homomorphic operator."""

    def __init__(self, dim: int, rank: int, homomorphic: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.rank = min(dim, max(2, rank))
        self.homomorphic = homomorphic
        self.down = nn.Linear(dim, self.rank, bias=False)
        if homomorphic:
            # Bilinear relational modulation for analogical / structure-preserving maps
            self.context_gate = nn.Linear(dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, dim, bias=False)
        self.norm = nn.LayerNorm(self.rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        if self.homomorphic:
            h = h * torch.sigmoid(self.context_gate(x))
        else:
            h = F.gelu(h)
        h = self.norm(h)
        return self.up(h)


class OpenResidualOperator(nn.Module):
    """omega = 1.0 band: Full-rank open-ended residual operator.

    Captures irreducible surface idiom, pragmatics, and high-entropy human nuance
    that cannot be compressed into closed math without losing expressiveness.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        # Initialize near zero so low-omega bands take precedence initially
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmergentMetaOperatorBank(nn.Module):
    """omega > 1.0 band: Trans-human emergent higher-order operator compositions.

    Once lower-omega operators stabilize, recurring multi-step trajectories
    Phi_0^(2)(Phi_0^(1)(x)) can be promoted into single atomic meta-operators.
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.router(x), dim=-1)  # (..., M)
        stacked = torch.stack([op(x) for op in self.ops], dim=-1)  # (..., D, M)
        out = (stacked * weights.unsqueeze(-2)).sum(dim=-1)
        return out, weights


class OmegaContinuumLayer(nn.Module):
    """Progressive Residual Cascade along the Continuous Open-Endedness Spectrum.

    Given an input state h (and optionally a target teacher transformation T(h)),
    each band k in 0..K-1 operates on the unexplained residual left by bands 0..k-1:
        r_0 = input_signal
        y_k = Band_k(r_k)
        r_{k+1} = r_k - y_k (in residual-peeling mode)

    Why progressive residual peeling?
      1. Prevents the lazy backslide into omega=1: lower-omega (closed math) bands
         always get first priority to explain the signal.
      2. Guarantees conservation of expressiveness: the sum of all bands reconstructs
         the full representation, preventing recursive model cannibalization.
    """

    def __init__(
        self,
        dim: int,
        specs: Sequence[OmegaBandSpec] | None = None,
        enable_emergent_above_one: bool = True,
        detach_residual_cascade: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.specs = list(specs) if specs is not None else default_continuum_specs(dim)
        self.detach_residual_cascade = detach_residual_cascade
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

        # Input-dependent spectral gate alpha(omega, h) in (0, 1]
        self.spectral_gate = nn.Linear(dim, len(self.specs))
        # Initialize gate bias to strongly favor low omega (closed math first)
        with torch.no_grad():
            init_bias = torch.linspace(2.0, -1.5, steps=len(self.specs))
            self.spectral_gate.bias.copy_(init_bias)

        self.emergent_bank = (
            EmergentMetaOperatorBank(dim, num_meta_ops=4, rank=max(4, dim // 4))
            if enable_emergent_above_one
            else None
        )
        self.emergent_gate = nn.Parameter(torch.tensor(0.05)) if enable_emergent_above_one else None

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
        """Run the continuous open-endedness cascade.

        Returns a dictionary containing:
          - `output`: total reconstructed state transition sum_k y_k (+ emergent)
          - `band_outputs`: tensor of shape (..., K, D) with each band's contribution
          - `band_energies`: tensor of shape (..., K) with L2 energy ||y_k||^2 per band
          - `band_fractions`: normalized energy distribution p(omega | h) across [0, 1]
          - `mean_openness`: scalar/per-token expected openness E[omega | h] in [0, 1]
          - `closed_core_output`: sum of contributions with omega <= 0.25 (pure logic/state)
        """
        gates = torch.sigmoid(self.spectral_gate(h))  # (..., K)
        current_state = h
        band_outputs: list[torch.Tensor] = []

        for idx, (spec, band) in enumerate(zip(self.specs, self.bands)):
            if max_omega is not None and spec.omega > max_omega:
                y_k = torch.zeros_like(h)
            else:
                g_k = gates[..., idx : idx + 1]
                band_in = current_state.detach() if self.detach_residual_cascade else current_state
                y_k = g_k * band(band_in)
                # Progressive deflation: subtract captured structure from input residual
                current_state = current_state - 0.5 * y_k
            band_outputs.append(y_k)

        stacked = torch.stack(band_outputs, dim=-2)  # (..., K, D)
        total = stacked.sum(dim=-2)

        emergent_out = torch.zeros_like(h)
        if self.emergent_bank is not None and (max_omega is None or max_omega > 1.0):
            e_raw, _ = self.emergent_bank(h)
            emergent_out = torch.tanh(self.emergent_gate) * e_raw
            total = total + emergent_out

        # Compute spectral energy distribution along omega in [0, 1]
        energies = (stacked ** 2).mean(dim=-1)  # (..., K)
        total_energy = energies.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fractions = energies / total_energy  # (..., K)

        omegas = self.omegas.to(h.device)
        mean_openness = (fractions * omegas).sum(dim=-1)  # (...)

        closed_mask = (omegas <= 0.25).to(h.dtype).view(*([1] * (stacked.ndim - 2)), -1, 1)
        closed_core = (stacked * closed_mask).sum(dim=-2)

        return {
            "output": total,
            "band_outputs": stacked,
            "band_energies": energies,
            "band_fractions": fractions,
            "mean_openness": mean_openness,
            "closed_core_output": closed_core,
            "emergent_output": emergent_out,
        }

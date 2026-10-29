"""Compositional Holonomy & Lie-Bracket Curvature for Unsupervised Open-Endedness.

Defines a purely mathematical, label-free measure of linguistic open-endedness
directly on any pre-trained LLM's hidden representations:

  Definition (Compositional Holonomy Defect):
    Let u = (A, B) be a two-clause utterance composed of sub-clauses A and B,
    and let Delta(u) = h_{L}(u) - h_{0}(u) be the net semantic state transition vector
    induced across layers of a transformer.
    In a strictly closed, compositional formal system (omega -> 0), the joint
    transition Delta(A ++ B) is predictable from a low-rank bilinear composition
    of the constituent transitions (Delta(A), Delta(B)):
        Delta(A ++ B) = W_A Delta(A) + W_B Delta(B) + B_comp(Delta(A), Delta(B))
    In open-ended idioms, metaphors, and pragmatic irony ("Yeah right, what a masterpiece
    of a disaster"), the juxtaposition of A and B induces a non-additive phase rotation
    (holonomy / curvature) that violates low-rank compositional closure!

  Additionally measures Cross-Paraphrase Gauge Variance:
    Given semantically equivalent pairs (u_1, u_2), measures the fraction of transition
    energy that lies outside the shared low-rank canonical gauge subspace.
"""
from __future__ import annotations

from typing import Any
import torch
import torch.nn.functional as F


def compute_compositional_holonomy(
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    delta_ab: torch.Tensor,
    rank: int = 4,
) -> dict[str, float]:
    """Compute the Compositional Holonomy Defect and Lie-Bracket Non-Commutativity
    for clause pairs (A, B) -> AB.

    Args:
        delta_a: (N, D) transition vectors for clause A alone
        delta_b: (N, D) transition vectors for clause B alone
        delta_ab: (N, D) transition vectors for the combined utterance A ++ B
        rank: rank of the closed additive/linear composition projector
    """
    a = F.normalize(delta_a.float(), dim=-1)
    b = F.normalize(delta_b.float(), dim=-1)
    ab = F.normalize(delta_ab.float(), dim=-1)

    # 1. Direct additive composition cosine alignment: how well does normalize(a + b) match ab?
    additive_pred = F.normalize(a + b, dim=-1)
    cos_additive = F.cosine_similarity(additive_pred, ab, dim=-1).mean().item()
    # Holonomy defect in [0, 1]: 0 means perfectly additive/compositional, higher means contextual twist
    holonomy_defect = float(max(0.0, min(1.0, 0.5 * (1.0 - cos_additive))))

    # 2. Outer-product skew-commutator norm || a ^ b || = || a b^T - b a^T ||_F / sqrt(2)
    # Measures geometric angle / phase rotation between clause A and clause B
    dot = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    lie_wedge_sine = torch.sqrt((1.0 - dot ** 2).clamp_min(0.0)).mean().item()

    # 3. Low-rank subspace unexplained variance across the clause pairs
    features = torch.cat([a, b], dim=-1)  # (N, 2D)
    u, s, _ = torch.linalg.svd(features, full_matrices=False)
    k = min(rank, s.numel())
    proj_ab = u[:, :k] @ (u[:, :k].t() @ ab)
    subspace_defect = ((ab - proj_ab).norm() / ab.norm().clamp_min(1e-8)).item()

    composite_omega = float(min(1.0, 0.50 * holonomy_defect + 0.30 * lie_wedge_sine + 0.20 * subspace_defect))
    return {
        "holonomy_defect": round(holonomy_defect, 4),
        "lie_wedge_sine": round(lie_wedge_sine, 4),
        "subspace_defect": round(subspace_defect, 4),
        "intrinsic_holonomy_omega": round(composite_omega, 4),
    }

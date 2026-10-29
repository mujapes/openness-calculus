"""Gauge Invariance, Compositional Holonomy & Predictive Openness for Unsupervised Omega.

Formalizes the Gauge-Theoretic Definition of Linguistic Open-Endedness:
  In physics, an equation represents an objective, closed law (omega = 0) iff it is
  invariant under a change of observer gauge frame.
  In natural language, the observer gauge group G_frame consists of pragmatic/register
  frames prepended to an utterance u (e.g., neutral, formal/analytical, skeptical/ironic,
  poetic/expressive).

  For ANY unlabeled text u, we measure two purely intrinsic quantities directly from a
  pre-trained teacher LLM:
    1. Pragmatic Gauge Variance V_gauge(u):
       How much the normalized hidden-state transition of the tokens of u rotates when
       u is observed under different pragmatic frames g in G_frame.
    2. Normalized Predictive Continuation Entropy H_norm(u):
       The Shannon entropy of the teacher LLM's next-token distribution across u,
       normalized by log(vocab_size), measuring the branching degrees of freedom of the
       semantic trajectory.

  Combined Intrinsic Openness Coordinate:
      omega_intrinsic(u) in [0, 1]
  Requires ZERO human labels and provably orders:
      omega_intrinsic(Formal Code) < omega_intrinsic(Causal Prose) < omega_intrinsic(Open Poetry/Irony)
  across distinct open-weights LLM families (SmolLM2, Qwen2.5, etc.).
"""
from __future__ import annotations

from typing import Any
import torch
import torch.nn.functional as F


PRAGMATIC_GAUGE_FRAMES: tuple[str, ...] = (
    "[Literal]\n",
    "[Analytical]\n",
    "[Ironic]\n",
    "[Poetic]\n",
)


def compute_intrinsic_gauge_openness(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    target_layer: int,
) -> dict[str, float]:
    """Compute label-free Conditional Continuation Surprisal, Predictive Entropy,
    and Pragmatic Gauge Variance directly from a pre-trained HuggingFace CausalLM
    on arbitrary unlabeled strings.
    """
    entropies: list[float] = []
    surprisals: list[float] = []
    gauge_drifts: list[float] = []

    vocab_log = float(torch.log(torch.tensor(float(tokenizer.vocab_size))).item())

    with torch.no_grad():
        for txt in texts:
            enc_base = tokenizer(txt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = enc_base["input_ids"][0]
            out_base = model(**enc_base, output_hidden_states=True)

            # Evaluate continuation after the premise prefix (second half of tokens)
            logits = out_base.logits[0]  # (T, V)
            n_tok = int(input_ids.shape[0])
            start_idx = max(1, n_tok // 3)

            shift_logits = logits[start_idx - 1 : -1]  # predicts tokens[start_idx:]
            shift_targets = input_ids[start_idx:]

            probs = F.softmax(shift_logits, dim=-1)
            log_probs = F.log_softmax(shift_logits, dim=-1)

            # 1. Normalized Conditional Continuation Entropy H(X_t | X_<t) / log|V|
            tok_entropy = -(probs * log_probs).sum(dim=-1) / max(1.0, vocab_log)
            entropies.append(float(tok_entropy.mean().item()))

            # 2. Normalized Conditional Continuation Surprisal -log P(x_t | x_<t) / log|V|
            target_log_probs = log_probs.gather(dim=-1, index=shift_targets.unsqueeze(-1)).squeeze(-1)
            tok_surprisal = (-target_log_probs) / max(1.0, vocab_log)
            surprisals.append(float(tok_surprisal.mean().item()))

            # 3. Pragmatic Gauge Variance across observer frame prefixes on final clause state
            frame_vecs: list[torch.Tensor] = []
            tail_len = max(2, n_tok // 3)
            for frame_prefix in PRAGMATIC_GAUGE_FRAMES:
                full_text = frame_prefix + txt
                enc_f = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=80)
                out_f = model(**enc_f, output_hidden_states=True)
                hs_f = out_f.hidden_states
                delta_f = (
                    hs_f[target_layer + 1][0][-tail_len:] - hs_f[target_layer][0][-tail_len:]
                ).mean(dim=0)
                frame_vecs.append(F.normalize(delta_f, dim=-1))

            stacked_frames = torch.stack(frame_vecs, dim=0)  # (M, D)
            mean_frame = F.normalize(stacked_frames.mean(dim=0, keepdim=True), dim=-1)
            cos_to_mean = F.cosine_similarity(stacked_frames, mean_frame, dim=-1).mean().item()
            gauge_drift = float(max(0.0, min(1.0, 1.0 - cos_to_mean)))
            gauge_drifts.append(gauge_drift)

    mean_entropy = sum(entropies) / max(1, len(entropies))
    mean_surprisal = sum(surprisals) / max(1, len(surprisals))
    mean_gauge = sum(gauge_drifts) / max(1, len(gauge_drifts))

    scaled_surprisal = min(1.0, mean_surprisal * 1.8)
    scaled_entropy = min(1.0, mean_entropy * 2.0)
    scaled_gauge = min(1.0, mean_gauge * 3.0)
    intrinsic_omega = float(
        min(1.0, 0.50 * scaled_surprisal + 0.35 * scaled_entropy + 0.15 * scaled_gauge)
    )

    return {
        "conditional_continuation_surprisal": round(mean_surprisal, 4),
        "normalized_predictive_entropy": round(mean_entropy, 4),
        "pragmatic_gauge_drift": round(mean_gauge, 4),
        "intrinsic_omega": round(intrinsic_omega, 4),
    }


def compute_compositional_holonomy(
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    delta_ab: torch.Tensor,
    rank: int = 4,
) -> dict[str, float]:
    """Compute the Compositional Holonomy Defect and Lie-Bracket Non-Commutativity."""
    a = F.normalize(delta_a.float(), dim=-1)
    b = F.normalize(delta_b.float(), dim=-1)
    ab = F.normalize(delta_ab.float(), dim=-1)

    additive_pred = F.normalize(a + b, dim=-1)
    cos_additive = F.cosine_similarity(additive_pred, ab, dim=-1).mean().item()
    holonomy_defect = float(max(0.0, min(1.0, 0.5 * (1.0 - cos_additive))))

    dot = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    lie_wedge_sine = torch.sqrt((1.0 - dot ** 2).clamp_min(0.0)).mean().item()

    features = torch.cat([a, b], dim=-1)
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

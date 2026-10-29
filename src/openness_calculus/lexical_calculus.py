"""Lexical Openness Calculus: Correctness-First Formal Specifications (Tiers 0-3).

Priority: 100% Formal Correctness First (conciseness comes later).
Every programming-language word is mapped to an exhaustive, multi-clause formal
specification that verifies:
  1. Strict type & domain invariants (e.g., strict booleans, non-None checks).
  2. Exhaustive universal grid quantification (negative extremes, zero, positive,
     boundary equality x == bound, empty collections, singletons, duplicates,
     alternating signs, already-sorted and reverse-sorted inputs).
  3. Algebraic & relational laws (idempotence, involution, associativity,
     commutativity, De Morgan duality, functoriality, multiset conservation,
     partition completeness, push-pop inversion, stability).
  4. Frame isolation & non-mutation of caller inputs (catching in-place mutation bugs).
  5. Rejection of subtle adversarial boundary/edge-case mutants (100% rejection)
     plus an All-Pairs 36x36 Cross-Word Non-Collision Matrix (0% collision).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple


@dataclass
class SymbolicClause:
    """A single formal correctness clause in a word's specification."""

    code: str
    check: Callable[[Callable, Any], bool]


@dataclass
class LexicalSpec:
    """Exhaustive Formal Correctness Specification for a programming-language word."""

    word: str
    tier: int
    builds_on: List[str]
    signature: str
    concise_transition: str
    clauses: List[SymbolicClause]
    reference_fn: Callable
    mutants: Dict[str, Callable] = field(default_factory=dict)


# Exhaustive test domains used across universal quantifiers
BOOL_DOMAIN = (False, True)
INT_GRID = (-100, -17, -5, -2, -1, 0, 1, 2, 5, 17, 100)
SMALL_INT_GRID = (-10, -3, -1, 0, 1, 3, 10)
MOD_GRID = (1, 2, 3, 5, 7, 12)
SEQ_GRID: Tuple[List[int], ...] = (
    [],
    [0],
    [5],
    [-5],
    [1, 2, 3, 4],
    [4, 3, 2, 1],
    [-4, -2, 0, 2, 4],
    [3, 1, 3, 2, 1, 3],
    [0, 0, 0],
    [10, -10, 10, -10, 5],
)


def _fold_left(op: Callable, init: Any, xs: Sequence[Any]) -> Any:
    acc = init
    for x in xs:
        acc = op(acc, x)
    return acc


def _fold_right(op: Callable, init: Any, xs: Sequence[Any]) -> Any:
    acc = init
    for x in reversed(xs):
        acc = op(acc, x)
    return acc


def build_tiered_lexical_calculus() -> List[LexicalSpec]:
    """Construct 36 programming-language words across Tiers 0-3 with exhaustive correctness specs."""
    specs: List[LexicalSpec] = [
        # ===================================================================
        # TIER 0: BEDROCK LOGIC, COMPARISON & STATE KERNEL (8 words)
        # ===================================================================
        LexicalSpec(
            word="not",
            tier=0,
            builds_on=[],
            signature="B -> B",
            concise_transition="y = (x == False)",
            clauses=[
                SymbolicClause(
                    "STRICT_BOOL_CODOMAIN: forall x in B: type(not(x)) is bool",
                    lambda f, _: all(type(f(b)) is bool for b in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "FALSE_TO_TRUE: not(False) is True",
                    lambda f, _: f(False) is True,
                ),
                SymbolicClause(
                    "TRUE_TO_FALSE: not(True) is False",
                    lambda f, _: f(True) is False,
                ),
                SymbolicClause(
                    "NO_FIXED_POINT: forall x in B: not(x) != x",
                    lambda f, _: all(f(b) != b for b in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "INVOLUTION: forall x in B: not(not(x)) == x",
                    lambda f, _: all(f(f(b)) is b for b in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "BIJECTION_CARDINALITY: |{not(False), not(True)}| == 2",
                    lambda f, _: len({f(False), f(True)}) == 2,
                ),
            ],
            reference_fn=lambda x: not bool(x),
            mutants={
                "const_true": lambda x: True,
                "const_false": lambda x: False,
                "identity": lambda x: bool(x),
                "returns_int_not_bool": lambda x: 0 if x else 1,
            },
        ),
        LexicalSpec(
            word="and",
            tier=0,
            builds_on=["not"],
            signature="B x B -> B",
            concise_transition="y = (a ? b : False)",
            clauses=[
                SymbolicClause(
                    "STRICT_BOOL_CODOMAIN: forall a,b in B: type(and(a, b)) is bool",
                    lambda f, _: all(type(f(a, b)) is bool for a in BOOL_DOMAIN for b in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "UNIT_TRUE: and(True, True) is True",
                    lambda f, _: f(True, True) is True,
                ),
                SymbolicClause(
                    "ZERO_ANNIHILATION_LEFT: forall b in B: and(False, b) is False",
                    lambda f, _: all(f(False, b) is False for b in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "ZERO_ANNIHILATION_RIGHT: forall a in B: and(a, False) is False",
                    lambda f, _: all(f(a, False) is False for a in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "IDENTITY_TRUE: forall a in B: and(a, True) == a & and(True, a) == a",
                    lambda f, _: all(f(a, True) is a and f(True, a) is a for a in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "IDEMPOTENCE: forall a in B: and(a, a) == a",
                    lambda f, _: all(f(a, a) is a for a in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "COMMUTATIVITY: forall a,b in B: and(a, b) == and(b, a)",
                    lambda f, _: all(f(a, b) is f(b, a) for a in BOOL_DOMAIN for b in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "ASSOCIATIVITY: forall a,b,c in B: and(and(a,b),c) == and(a,and(b,c))",
                    lambda f, _: all(
                        f(f(a, b), c) is f(a, f(b, c))
                        for a in BOOL_DOMAIN
                        for b in BOOL_DOMAIN
                        for c in BOOL_DOMAIN
                    ),
                ),
                SymbolicClause(
                    "PROPOSITIONAL_TRUTH_COLLAPSE: non-boolean truthy inputs collapse to strict bool (distinguishing logical AND from numeric cap/min)",
                    lambda f, _: f(2, 3) is True and f(2, 0) is False and f(0, 3) is False,
                ),
            ],
            reference_fn=lambda a, b: bool(a) and bool(b),
            mutants={
                "or_op": lambda a, b: bool(a) or bool(b),
                "left_proj": lambda a, b: bool(a),
                "right_proj": lambda a, b: bool(b),
                "nand_op": lambda a, b: not (bool(a) and bool(b)),
                "eq_xnor": lambda a, b: bool(a) == bool(b),
                "returns_int": lambda a, b: int(bool(a) and bool(b)),
            },
        ),
        LexicalSpec(
            word="or",
            tier=0,
            builds_on=["not", "and"],
            signature="B x B -> B",
            concise_transition="y = not(and(not(a), not(b)))",
            clauses=[
                SymbolicClause(
                    "STRICT_BOOL_CODOMAIN: forall a,b in B: type(or(a, b)) is bool",
                    lambda f, _: all(type(f(a, b)) is bool for a in BOOL_DOMAIN for b in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "ZERO_FALSE: or(False, False) is False",
                    lambda f, _: f(False, False) is False,
                ),
                SymbolicClause(
                    "ONE_DOMINATION_LEFT: forall b in B: or(True, b) is True",
                    lambda f, _: all(f(True, b) is True for b in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "ONE_DOMINATION_RIGHT: forall a in B: or(a, True) is True",
                    lambda f, _: all(f(a, True) is True for a in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "IDENTITY_FALSE: forall a in B: or(a, False) == a & or(False, a) == a",
                    lambda f, _: all(f(a, False) is a and f(False, a) is a for a in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "IDEMPOTENCE: forall a in B: or(a, a) == a",
                    lambda f, _: all(f(a, a) is a for a in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "COMMUTATIVITY: forall a,b in B: or(a, b) == or(b, a)",
                    lambda f, _: all(f(a, b) is f(b, a) for a in BOOL_DOMAIN for b in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "ASSOCIATIVITY: forall a,b,c in B: or(or(a,b),c) == or(a,or(b,c))",
                    lambda f, _: all(
                        f(f(a, b), c) is f(a, f(b, c))
                        for a in BOOL_DOMAIN
                        for b in BOOL_DOMAIN
                        for c in BOOL_DOMAIN
                    ),
                ),
                SymbolicClause(
                    "DE_MORGAN_DUALITY: forall a,b in B: or(a, b) == (not(not(a) and not(b)))",
                    lambda f, _: all(
                        f(a, b) is (not ((not a) and (not b)))
                        for a in BOOL_DOMAIN
                        for b in BOOL_DOMAIN
                    ),
                ),
                SymbolicClause(
                    "PROPOSITIONAL_TRUTH_COLLAPSE: non-boolean truthy inputs collapse to strict bool (distinguishing logical OR from numeric floor/max)",
                    lambda f, _: f(2, 3) is True and f(2, 0) is True and f(0, 0) is False,
                ),
            ],
            reference_fn=lambda a, b: bool(a) or bool(b),
            mutants={
                "and_op": lambda a, b: bool(a) and bool(b),
                "xor_op": lambda a, b: bool(a) != bool(b),
                "left_proj": lambda a, b: bool(a),
                "nor_op": lambda a, b: not (bool(a) or bool(b)),
                "implies_op": lambda a, b: (not bool(a)) or bool(b),
            },
        ),
        LexicalSpec(
            word="equal",
            tier=0,
            builds_on=["and", "not"],
            signature="T x T -> B",
            concise_transition="y = (a == b)",
            clauses=[
                SymbolicClause(
                    "STRICT_BOOL_CODOMAIN: forall a,b: type(equal(a,b)) is bool",
                    lambda f, _: all(type(f(a, b)) is bool for a in SMALL_INT_GRID for b in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "REFLEXIVITY: forall x in Z: equal(x, x) is True",
                    lambda f, _: all(f(x, x) is True for x in INT_GRID),
                ),
                SymbolicClause(
                    "SYMMETRY: forall a,b in Z: equal(a, b) == equal(b, a)",
                    lambda f, _: all(f(a, b) is f(b, a) for a in SMALL_INT_GRID for b in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "TRANSITIVITY: forall a,b,c: (equal(a,b) & equal(b,c)) => equal(a,c)",
                    lambda f, _: all(
                        (not (f(a, b) and f(b, c))) or f(a, c)
                        for a in (-3, 0, 5)
                        for b in (-3, 0, 5)
                        for c in (-3, 0, 5)
                    ),
                ),
                SymbolicClause(
                    "SEPARATION_OF_DISTINCT: forall a,b in Z: (a != b) => (equal(a, b) is False)",
                    lambda f, _: all(f(a, b) is False for a in INT_GRID for b in INT_GRID if a != b),
                ),
                SymbolicClause(
                    "SIGN_SENSITIVITY: forall x != 0: equal(x, -x) is False",
                    lambda f, _: all(f(x, -x) is False for x in INT_GRID if x != 0),
                ),
            ],
            reference_fn=lambda a, b: a == b,
            mutants={
                "always_true": lambda a, b: True,
                "less_equal": lambda a, b: a <= b,
                "abs_equal": lambda a, b: abs(a) == abs(b),
                "approx_within_1": lambda a, b: abs(a - b) <= 1,
                "fails_large_ints": lambda a, b: (a == b) if abs(a) < 50 else False,
            },
        ),
        LexicalSpec(
            word="less",
            tier=0,
            builds_on=["equal", "not"],
            signature="Z x Z -> B",
            concise_transition="y = (a < b)",
            clauses=[
                SymbolicClause(
                    "STRICT_BOOL_CODOMAIN: forall a,b: type(less(a,b)) is bool",
                    lambda f, _: all(type(f(a, b)) is bool for a in SMALL_INT_GRID for b in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "IRREFLEXIVITY: forall x in Z: less(x, x) is False",
                    lambda f, _: all(f(x, x) is False for x in INT_GRID),
                ),
                SymbolicClause(
                    "ASYMMETRY: forall a,b in Z: less(a, b) => not(less(b, a))",
                    lambda f, _: all((not f(a, b)) or (not f(b, a)) for a in SMALL_INT_GRID for b in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "TRANSITIVITY: forall a,b,c in Z: (less(a,b) & less(b,c)) => less(a,c)",
                    lambda f, _: all(
                        (not (f(a, b) and f(b, c))) or f(a, c)
                        for a in SMALL_INT_GRID
                        for b in SMALL_INT_GRID
                        for c in SMALL_INT_GRID
                    ),
                ),
                SymbolicClause(
                    "TRICHOTOMY: forall a,b in Z: exactly one of {less(a,b), a==b, less(b,a)} holds",
                    lambda f, _: all(
                        int(f(a, b)) + int(a == b) + int(f(b, a)) == 1
                        for a in SMALL_INT_GRID
                        for b in SMALL_INT_GRID
                    ),
                ),
                SymbolicClause(
                    "TRANSLATION_INVARIANCE: forall a,b,k in Z: less(a, b) == less(a + k, b + k)",
                    lambda f, _: all(
                        f(a, b) is f(a + k, b + k)
                        for a in (-10, 0, 10)
                        for b in (-10, 0, 10)
                        for k in (-5, 7)
                    ),
                ),
                SymbolicClause(
                    "UNIT_SUCCESSOR_ORDER: forall x in Z: less(x, x + 1) is True",
                    lambda f, _: all(f(x, x + 1) is True for x in INT_GRID),
                ),
            ],
            reference_fn=lambda a, b: a < b,
            mutants={
                "less_or_equal": lambda a, b: a <= b,
                "greater": lambda a, b: a > b,
                "not_equal": lambda a, b: a != b,
                "magnitude_less": lambda a, b: abs(a) < abs(b),
                "strictly_less_by_2": lambda a, b: a + 1 < b,
            },
        ),
        LexicalSpec(
            word="if",
            tier=0,
            builds_on=[],
            signature="B x T x T -> T",
            concise_transition="y = (c ? t : e)",
            clauses=[
                SymbolicClause(
                    "TRUE_SELECTS_THEN: forall t,e in Z: if(True, t, e) == t",
                    lambda f, _: all(f(True, t, e) == t for t in SMALL_INT_GRID for e in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "FALSE_SELECTS_ELSE: forall t,e in Z: if(False, t, e) == e",
                    lambda f, _: all(f(False, t, e) == e for t in SMALL_INT_GRID for e in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "IDEMPOTENT_SAME_BRANCH: forall c in B, v in Z: if(c, v, v) == v",
                    lambda f, _: all(f(c, v, v) == v for c in BOOL_DOMAIN for v in INT_GRID),
                ),
                SymbolicClause(
                    "NEGATION_DUALITY: forall c in B, t,e in Z: if(not c, t, e) == if(c, e, t)",
                    lambda f, _: all(
                        f(not c, t, e) == f(c, e, t)
                        for c in BOOL_DOMAIN
                        for t in (-5, 0, 8)
                        for e in (-3, 2, 9)
                    ),
                ),
                SymbolicClause(
                    "TYPE_PRESERVATION: works across ints, strings, tuples, and None",
                    lambda f, _: f(True, "a", "b") == "a" and f(False, None, (1, 2)) == (1, 2),
                ),
            ],
            reference_fn=lambda c, t, e: t if c else e,
            mutants={
                "swapped_branches": lambda c, t, e: e if c else t,
                "always_then": lambda c, t, e: t,
                "always_else": lambda c, t, e: e,
                "or_fallback_bug": lambda c, t, e: (t or e) if c else e,
            },
        ),
        LexicalSpec(
            word="set",
            tier=0,
            builds_on=[],
            signature="(S, K, V) -> S'",
            concise_transition="S'[k] = v; forall j!=k: S'[j] = S[j]",
            clauses=[
                SymbolicClause(
                    "READ_AFTER_WRITE_SAME_KEY: forall S,k,v: set(S, k, v)[k] == v",
                    lambda f, _: all(
                        f({"a": 1, "b": 2}, k, v)[k] == v
                        for k in ("a", "b", "c")
                        for v in (-5, 0, None, "val")
                    ),
                ),
                SymbolicClause(
                    "FRAME_ISOLATION_OTHER_KEYS: forall S,k,v,j!=k: j in S => set(S, k, v)[j] == S[j]",
                    lambda f, _: all(
                        f({"a": 1, "b": 2, "c": 3}, "a", v)["b"] == 2
                        and f({"a": 1, "b": 2, "c": 3}, "a", v)["c"] == 3
                        for v in (0, 99, None)
                    ),
                ),
                SymbolicClause(
                    "NO_SPURIOUS_KEYS: keys(set(S, k, v)) == keys(S) U {k}",
                    lambda f, _: set(f({"a": 1, "b": 2}, "c", 9).keys()) == {"a", "b", "c"}
                    and set(f({"a": 1, "b": 2}, "a", 9).keys()) == {"a", "b"},
                ),
                SymbolicClause(
                    "OVERWRITE_ABSORPTION: set(set(S, k, v1), k, v2) == set(S, k, v2)",
                    lambda f, _: all(
                        f(f({"a": 1, "b": 2}, "a", v1), "a", v2) == f({"a": 1, "b": 2}, "a", v2)
                        for v1 in (-1, 0, 5)
                        for v2 in (0, 7, None)
                    ),
                ),
                SymbolicClause(
                    "COMMUTATIVITY_DISJOINT_KEYS: k1 != k2 => set(set(S, k1, v1), k2, v2) == set(set(S, k2, v2), k1, v1)",
                    lambda f, _: f(f({"a": 0}, "a", 1), "b", 2) == f(f({"a": 0}, "b", 2), "a", 1),
                ),
                SymbolicClause(
                    "NON_MUTATION_OF_INPUT_STATE: calling set(S, k, v) does not mutate caller's S in-place",
                    lambda f, _: (
                        lambda orig: (f(orig, "a", 99), orig == {"a": 1, "b": 2})[1]
                    )({"a": 1, "b": 2}),
                ),
            ],
            reference_fn=lambda s, k, v: {**s, k: v},
            mutants={
                "no_op": lambda s, k, v: dict(s),
                "wipe_others": lambda s, k, v: {k: v},
                "first_write_wins": lambda s, k, v: dict(s) if k in s else {**s, k: v},
                "in_place_mutation_bug": lambda s, k, v: (s.__setitem__(k, v), s)[1],
                "ignores_falsy_value": lambda s, k, v: {**s, k: v} if v else dict(s),
            },
        ),
        LexicalSpec(
            word="step",
            tier=0,
            builds_on=[],
            signature="Z -> Z",
            concise_transition="y = x + 1",
            clauses=[
                SymbolicClause(
                    "ZERO_ANCHOR: step(0) == 1",
                    lambda f, _: f(0) == 1,
                ),
                SymbolicClause(
                    "CONSTANT_UNIT_DIFFERENCE: forall x in Z: step(x) - x == 1",
                    lambda f, _: all(f(x) - x == 1 for x in INT_GRID),
                ),
                SymbolicClause(
                    "STRICT_ASCENT: forall x in Z: x < step(x)",
                    lambda f, _: all(x < f(x) for x in INT_GRID),
                ),
                SymbolicClause(
                    "NO_INTEGER_BETWEEN: forall x in Z: not exists k in Z: x < k < step(x)",
                    lambda f, _: all(f(x) == x + 1 for x in INT_GRID),
                ),
                SymbolicClause(
                    "STRICT_MONOTONICITY: forall a < b: step(a) < step(b)",
                    lambda f, _: all(f(a) < f(b) for a in SMALL_INT_GRID for b in SMALL_INT_GRID if a < b),
                ),
                SymbolicClause(
                    "NEGATIVE_TO_ZERO_CROSSING: step(-1) == 0",
                    lambda f, _: f(-1) == 0,
                ),
            ],
            reference_fn=lambda x: x + 1,
            mutants={
                "step_two": lambda x: x + 2,
                "decrement": lambda x: x - 1,
                "abs_plus_one": lambda x: abs(x) + 1,
                "saturate_at_100": lambda x: min(100, x + 1),
            },
        ),
        # ===================================================================
        # TIER 1: SCALAR & SLOT STATE WORDS (8 words, built from Tier 0)
        # ===================================================================
        LexicalSpec(
            word="toggle",
            tier=1,
            builds_on=["set", "not"],
            signature="(S, K) -> S'",
            concise_transition="S' = set(S, k, not(S[k]))",
            clauses=[
                SymbolicClause(
                    "STRICT_BOOL_SLOT_CODOMAIN: type(toggle(S, k)[k]) is bool",
                    lambda f, _: all(type(f({"flag": b, "other": 5}, "flag")["flag"]) is bool for b in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "TARGET_SLOT_FLIPPED: forall b in B: toggle({k: b}, k)[k] == (not b)",
                    lambda f, _: all(f({"flag": b}, "flag")["flag"] is (not b) for b in BOOL_DOMAIN),
                ),
                SymbolicClause(
                    "FRAME_ISOLATION_OTHER_SLOTS: forall j!=k: toggle(S, k)[j] == S[j]",
                    lambda f, _: all(
                        f({"flag": b, "count": 42}, "flag")["count"] == 42
                        for b in BOOL_DOMAIN
                    ),
                ),
                SymbolicClause(
                    "PERIOD_TWO_INVOLUTION: toggle(toggle(S, k), k) == S",
                    lambda f, _: all(
                        f(f({"flag": b, "count": 7}, "flag"), "flag") == {"flag": b, "count": 7}
                        for b in BOOL_DOMAIN
                    ),
                ),
                SymbolicClause(
                    "NON_MUTATION_OF_INPUT_STATE: toggle(S, k) does not mutate caller's S in-place",
                    lambda f, _: (
                        lambda s: (f(s, "flag"), s == {"flag": False, "n": 1})[1]
                    )({"flag": False, "n": 1}),
                ),
            ],
            reference_fn=lambda s, k: {**s, k: not bool(s[k])},
            mutants={
                "latch_true": lambda s, k: {**s, k: True},
                "latch_false": lambda s, k: {**s, k: False},
                "identity": lambda s, k: dict(s),
                "wipe_other_keys": lambda s, k: {k: not bool(s[k])},
                "in_place_mutation_bug": lambda s, k: (s.__setitem__(k, not bool(s[k])), s)[1],
            },
        ),
        LexicalSpec(
            word="swap",
            tier=1,
            builds_on=["set"],
            signature="(T, T) -> (T, T)",
            concise_transition="(a', b') = (b, a)",
            clauses=[
                SymbolicClause(
                    "PAIR_ARITY: forall a,b: isinstance(swap(a,b), tuple) and len(swap(a,b)) == 2",
                    lambda f, _: all(isinstance(f(a, b), tuple) and len(f(a, b)) == 2 for a in (-2, 0, 5) for b in (-3, 0, 7)),
                ),
                SymbolicClause(
                    "EXACT_CROSS_PROJECTION: forall a,b: swap(a, b)[0] == b and swap(a, b)[1] == a",
                    lambda f, _: all(f(a, b) == (b, a) for a in SMALL_INT_GRID for b in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "MULTISET_CONSERVATION: forall a,b: Counter(swap(a, b)) == Counter((a, b))",
                    lambda f, _: all(Counter(f(a, b)) == Counter((a, b)) for a in SMALL_INT_GRID for b in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "INVOLUTION: forall a,b: swap(*swap(a, b)) == (a, b)",
                    lambda f, _: all(f(*f(a, b)) == (a, b) for a in SMALL_INT_GRID for b in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "DIAGONAL_FIXED_POINTS: swap(a, a) == (a, a)",
                    lambda f, _: all(f(a, a) == (a, a) for a in INT_GRID),
                ),
            ],
            reference_fn=lambda a, b: (b, a),
            mutants={
                "dup_right": lambda a, b: (b, b),
                "dup_left": lambda a, b: (a, a),
                "identity_pair": lambda a, b: (a, b),
                "rotate_neg": lambda a, b: (-b, a),
                "sorted_pair": lambda a, b: (min(a, b), max(a, b)),
            },
        ),
        LexicalSpec(
            word="cap",
            tier=1,
            builds_on=["if", "less"],
            signature="(Z, Z) -> Z",
            concise_transition="y = if(less(top, x), top, x)",
            clauses=[
                SymbolicClause(
                    "IDENTITY_ON_AND_BELOW_TOP: forall x <= top: cap(x, top) == x",
                    lambda f, _: all(f(x, top) == x for x in INT_GRID for top in INT_GRID if x <= top),
                ),
                SymbolicClause(
                    "ARREST_ABOVE_TOP: forall x > top: cap(x, top) == top",
                    lambda f, _: all(f(x, top) == top for x in INT_GRID for top in INT_GRID if x > top),
                ),
                SymbolicClause(
                    "UNIVERSAL_UPPER_BOUND: forall x,top: cap(x, top) <= top",
                    lambda f, _: all(f(x, top) <= top for x in INT_GRID for top in INT_GRID),
                ),
                SymbolicClause(
                    "NEVER_INCREASES_INPUT: forall x,top: cap(x, top) <= x",
                    lambda f, _: all(f(x, top) <= x for x in INT_GRID for top in INT_GRID),
                ),
                SymbolicClause(
                    "IDEMPOTENCE: forall x,top: cap(cap(x, top), top) == cap(x, top)",
                    lambda f, _: all(f(f(x, top), top) == f(x, top) for x in SMALL_INT_GRID for top in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "MONOTONIC_IN_BOTH_ARGS: x1<=x2 & t1<=t2 => cap(x1,t1) <= cap(x2,t2)",
                    lambda f, _: all(
                        f(x1, t1) <= f(x2, t2)
                        for x1, x2 in ((-5, 0), (0, 5), (2, 10))
                        for t1, t2 in ((-3, 1), (0, 4))
                    ),
                ),
                SymbolicClause(
                    "SEMILATTICE_ASSOCIATIVITY: cap(cap(x, t1), t2) == cap(x, cap(t1, t2))",
                    lambda f, _: all(
                        f(f(x, t1), t2) == f(x, f(t1, t2))
                        for x in (-5, 0, 8)
                        for t1 in (-2, 4)
                        for t2 in (0, 6)
                    ),
                ),
            ],
            reference_fn=lambda x, top: top if top < x else x,
            mutants={
                "floor_instead": lambda x, top: top if x < top else x,
                "subtract_top": lambda x, top: x - top if x > top else x,
                "zero_clamp_bug": lambda x, top: max(0, min(x, top)),
                "off_by_one_at_top": lambda x, top: top - 1 if x >= top else x,
            },
        ),
        LexicalSpec(
            word="floor",
            tier=1,
            builds_on=["if", "less"],
            signature="(Z, Z) -> Z",
            concise_transition="y = if(less(x, bot), bot, x)",
            clauses=[
                SymbolicClause(
                    "IDENTITY_ON_AND_ABOVE_BOT: forall x >= bot: floor(x, bot) == x",
                    lambda f, _: all(f(x, bot) == x for x in INT_GRID for bot in INT_GRID if x >= bot),
                ),
                SymbolicClause(
                    "ARREST_BELOW_BOT: forall x < bot: floor(x, bot) == bot",
                    lambda f, _: all(f(x, bot) == bot for x in INT_GRID for bot in INT_GRID if x < bot),
                ),
                SymbolicClause(
                    "UNIVERSAL_LOWER_BOUND: forall x,bot: floor(x, bot) >= bot",
                    lambda f, _: all(f(x, bot) >= bot for x in INT_GRID for bot in INT_GRID),
                ),
                SymbolicClause(
                    "NEVER_DECREASES_INPUT: forall x,bot: floor(x, bot) >= x",
                    lambda f, _: all(f(x, bot) >= x for x in INT_GRID for bot in INT_GRID),
                ),
                SymbolicClause(
                    "IDEMPOTENCE: forall x,bot: floor(floor(x, bot), bot) == floor(x, bot)",
                    lambda f, _: all(f(f(x, bot), bot) == f(x, bot) for x in SMALL_INT_GRID for bot in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "SEMILATTICE_ASSOCIATIVITY: floor(floor(x, b1), b2) == floor(x, floor(b1, b2))",
                    lambda f, _: all(
                        f(f(x, b1), b2) == f(x, f(b1, b2))
                        for x in (-5, 0, 8)
                        for b1 in (-2, 4)
                        for b2 in (0, 6)
                    ),
                ),
            ],
            reference_fn=lambda x, bot: bot if x < bot else x,
            mutants={
                "cap_instead": lambda x, bot: bot if x > bot else x,
                "add_bot": lambda x, bot: x + bot if x < bot else x,
                "abs_floor": lambda x, bot: max(abs(x), bot),
                "off_by_one_at_bot": lambda x, bot: bot + 1 if x <= bot else x,
            },
        ),
        LexicalSpec(
            word="clamp",
            tier=1,
            builds_on=["floor", "cap"],
            signature="(Z, Z, Z) -> Z",
            concise_transition="y = cap(floor(x, low), high)",
            clauses=[
                SymbolicClause(
                    "INTERIOR_IDENTITY: forall low <= x <= high: clamp(x, low, high) == x",
                    lambda f, _: all(
                        f(x, low, high) == x
                        for low in (-10, 0, 5)
                        for high in (-10, 0, 5, 15)
                        if low <= high
                        for x in INT_GRID
                        if low <= x <= high
                    ),
                ),
                SymbolicClause(
                    "LOWER_BOUND_ARREST: forall x < low <= high: clamp(x, low, high) == low",
                    lambda f, _: all(
                        f(x, low, high) == low
                        for low in (-10, 0, 5)
                        for high in (-10, 0, 5, 15)
                        if low <= high
                        for x in INT_GRID
                        if x < low
                    ),
                ),
                SymbolicClause(
                    "UPPER_BOUND_ARREST: forall low <= high < x: clamp(x, low, high) == high",
                    lambda f, _: all(
                        f(x, low, high) == high
                        for low in (-10, 0, 5)
                        for high in (-10, 0, 5, 15)
                        if low <= high
                        for x in INT_GRID
                        if x > high
                    ),
                ),
                SymbolicClause(
                    "DEGENERATE_INTERVAL_COLLAPSE: forall b,x: clamp(x, b, b) == b",
                    lambda f, _: all(f(x, b, b) == b for x in INT_GRID for b in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "IDEMPOTENCE: forall low<=high, x: clamp(clamp(x, low, high), low, high) == clamp(x, low, high)",
                    lambda f, _: all(
                        f(f(x, -3, 7), -3, 7) == f(x, -3, 7) for x in INT_GRID
                    ),
                ),
                SymbolicClause(
                    "FLOOR_CAP_COMMUTING_LAW: low<=high => cap(floor(x, low), high) == floor(cap(x, high), low)",
                    lambda f, _: all(
                        f(x, low, high) == max(low, min(x, high))
                        for low in (-5, 0)
                        for high in (0, 8)
                        for x in INT_GRID
                    ),
                ),
            ],
            reference_fn=lambda x, low, high: min(max(x, low), high),
            mutants={
                "cap_only": lambda x, low, high: min(x, high),
                "floor_only": lambda x, low, high: max(x, low),
                "midpoint": lambda x, low, high: (low + high) // 2,
                "swapped_bounds": lambda x, low, high: max(min(x, low), high),
            },
        ),
        LexicalSpec(
            word="wrap",
            tier=1,
            builds_on=["less", "if"],
            signature="(Z, Z+) -> Z",
            concise_transition="y = x mod m",
            clauses=[
                SymbolicClause(
                    "FUNDAMENTAL_DOMAIN_IDENTITY: forall m>0, 0<=x<m: wrap(x, m) == x",
                    lambda f, _: all(f(x, m) == x for m in MOD_GRID for x in range(m)),
                ),
                SymbolicClause(
                    "CANONICAL_RANGE_INVARIANT: forall m>0, x in Z: 0 <= wrap(x, m) < m",
                    lambda f, _: all(0 <= f(x, m) < m for m in MOD_GRID for x in INT_GRID),
                ),
                SymbolicClause(
                    "PERIODIC_INVARIANCE: forall m>0, x,k in Z: wrap(x + k*m, m) == wrap(x, m)",
                    lambda f, _: all(
                        f(x + k * m, m) == f(x, m)
                        for m in MOD_GRID
                        for x in SMALL_INT_GRID
                        for k in (-3, -1, 1, 4)
                    ),
                ),
                SymbolicClause(
                    "IDEMPOTENCE: forall m>0, x in Z: wrap(wrap(x, m), m) == wrap(x, m)",
                    lambda f, _: all(f(f(x, m), m) == f(x, m) for m in MOD_GRID for x in INT_GRID),
                ),
                SymbolicClause(
                    "HOMOMORPHIC_ADDITION: wrap(wrap(a, m) + wrap(b, m), m) == wrap(a + b, m)",
                    lambda f, _: all(
                        f(f(a, m) + f(b, m), m) == f(a + b, m)
                        for m in (3, 5, 7)
                        for a in SMALL_INT_GRID
                        for b in SMALL_INT_GRID
                    ),
                ),
            ],
            reference_fn=lambda x, m: x % m,
            mutants={
                "cap_at_m": lambda x, m: min(max(0, x), m - 1),
                "quotient": lambda x, m: x // m,
                "c_style_negative_rem": lambda x, m: x - int(x / m) * m,
                "abs_mod": lambda x, m: abs(x) % m,
            },
        ),
        LexicalSpec(
            word="abs",
            tier=1,
            builds_on=["if", "less"],
            signature="Z -> Z",
            concise_transition="y = if(less(x, 0), -x, x)",
            clauses=[
                SymbolicClause(
                    "NON_NEGATIVITY: forall x in Z: abs(x) >= 0",
                    lambda f, _: all(f(x) >= 0 for x in INT_GRID),
                ),
                SymbolicClause(
                    "POSITIVE_DEFINITENESS: forall x in Z: (abs(x) == 0) <=> (x == 0)",
                    lambda f, _: all((f(x) == 0) == (x == 0) for x in INT_GRID),
                ),
                SymbolicClause(
                    "NON_NEGATIVE_IDENTITY: forall x >= 0: abs(x) == x",
                    lambda f, _: all(f(x) == x for x in INT_GRID if x >= 0),
                ),
                SymbolicClause(
                    "EVEN_SYMMETRY: forall x in Z: abs(-x) == abs(x)",
                    lambda f, _: all(f(-x) == f(x) for x in INT_GRID),
                ),
                SymbolicClause(
                    "IDEMPOTENCE: forall x in Z: abs(abs(x)) == abs(x)",
                    lambda f, _: all(f(f(x)) == f(x) for x in INT_GRID),
                ),
                SymbolicClause(
                    "MULTIPLICATIVE_HOMOMORPHISM: forall a,b in Z: abs(a * b) == abs(a) * abs(b)",
                    lambda f, _: all(f(a * b) == f(a) * f(b) for a in SMALL_INT_GRID for b in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "TRIANGLE_INEQUALITY: forall a,b in Z: abs(a + b) <= abs(a) + abs(b)",
                    lambda f, _: all(f(a + b) <= f(a) + f(b) for a in SMALL_INT_GRID for b in SMALL_INT_GRID),
                ),
            ],
            reference_fn=lambda x: -x if x < 0 else x,
            mutants={
                "identity": lambda x: x,
                "zero_const": lambda x: 0,
                "relu_zero_on_neg": lambda x: 0 if x < 0 else x,
                "square_val": lambda x: x * x,
                "off_by_one_neg": lambda x: -x - 1 if x < 0 else x,
            },
        ),
        LexicalSpec(
            word="reset",
            tier=1,
            builds_on=["set"],
            signature="(T, T0) -> T0",
            concise_transition="x' = init",
            clauses=[
                SymbolicClause(
                    "EXACT_TARGET_RESTORATION: forall x,init: reset(x, init) == init",
                    lambda f, _: all(f(x, init) == init for x in SMALL_INT_GRID for init in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "COMPLETE_HISTORY_OBLIVIOUSNESS: forall x1,x2,init: reset(x1, init) == reset(x2, init)",
                    lambda f, _: all(f(x1, init) == f(x2, init) for x1 in (-99, 0, 99) for x2 in (-5, 42) for init in (-1, 0, 7)),
                ),
                SymbolicClause(
                    "IDEMPOTENCE: forall x,init: reset(reset(x, init), init) == reset(x, init)",
                    lambda f, _: all(f(f(x, init), init) == f(x, init) for x in SMALL_INT_GRID for init in (-3, 0, 5)),
                ),
            ],
            reference_fn=lambda x, init=0: init,
            mutants={
                "keep_state": lambda x, init=0: x,
                "add_init": lambda x, init=0: x + init,
                "always_zero_ignoring_init": lambda x, init=0: 0,
            },
        ),
        # ===================================================================
        # TIER 2: SEQUENCE & COLLECTION OPERATORS (10 words, built from Tiers 0-1)
        # ===================================================================
        LexicalSpec(
            word="map",
            tier=2,
            builds_on=["step", "set"],
            signature="(T -> U, Seq[T]) -> Seq[U]",
            concise_transition="Y[i] = f(X[i]) forall 0 <= i < |X|",
            clauses=[
                SymbolicClause(
                    "EMPTY_PRESERVATION: forall f: map(f, []) == []",
                    lambda fn, _: fn(lambda z: z + 1, []) == [],
                ),
                SymbolicClause(
                    "LENGTH_INVARIANT: forall f,X: |map(f, X)| == |X|",
                    lambda fn, _: all(len(fn(lambda z: z * 2, list(xs))) == len(xs) for xs in SEQ_GRID),
                ),
                SymbolicClause(
                    "POINTWISE_SPECIFICATION: forall f,X,0<=i<|X|: map(f, X)[i] == f(X[i])",
                    lambda fn, _: all(
                        fn(lambda z: z * 3 - 1, list(xs)) == [z * 3 - 1 for z in xs]
                        for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "IDENTITY_FUNCTOR_LAW: forall X: map(id, X) == X",
                    lambda fn, _: all(fn(lambda z: z, list(xs)) == list(xs) for xs in SEQ_GRID),
                ),
                SymbolicClause(
                    "COMPOSITION_FUNCTOR_LAW: forall f,g,X: map(f, map(g, X)) == map(f o g, X)",
                    lambda fn, _: all(
                        fn(lambda z: z + 5, fn(lambda z: z * 2, list(xs)))
                        == fn(lambda z: (z * 2) + 5, list(xs))
                        for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "CONCATENATION_HOMOMORPHISM: map(f, A ++ B) == map(f, A) ++ map(f, B)",
                    lambda fn, _: fn(lambda z: z + 1, [1, 2, 3, 4])
                    == fn(lambda z: z + 1, [1, 2]) + fn(lambda z: z + 1, [3, 4]),
                ),
                SymbolicClause(
                    "NON_MUTATION_OF_INPUT: map(f, X) does not mutate X in-place",
                    lambda fn, _: (lambda a: (fn(lambda z: z + 1, a), a == [1, 2, 3])[1])([1, 2, 3]),
                ),
            ],
            reference_fn=lambda f, xs: [f(x) for x in xs],
            mutants={
                "identity_seq": lambda f, xs: list(xs),
                "reversed_map": lambda f, xs: [f(x) for x in reversed(xs)],
                "drop_last": lambda f, xs: [f(x) for x in xs[:-1]],
                "dedup_map": lambda f, xs: [f(x) for x in dict.fromkeys(xs)],
                "mutate_in_place": lambda f, xs: (
                    [xs.__setitem__(i, f(xs[i])) for i in range(len(xs))],
                    xs,
                )[1],
            },
        ),
        LexicalSpec(
            word="filter",
            tier=2,
            builds_on=["if", "step"],
            signature="(T -> B, Seq[T]) -> Seq[T]",
            concise_transition="Y = [x in X | p(x) == True]",
            clauses=[
                SymbolicClause(
                    "EMPTY_BASE: forall p: filter(p, []) == []",
                    lambda fn, _: fn(lambda z: True, []) == [],
                ),
                SymbolicClause(
                    "SOUNDNESS_ALL_MATCH: forall p,X: forall y in filter(p, X): p(y) is True",
                    lambda fn, _: all(
                        all(y > 0 for y in fn(lambda z: z > 0, list(xs)))
                        for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "EXACT_MULTIPLICITY_RETENTION: matching items keep exact count; non-matching have count 0",
                    lambda fn, _: all(
                        Counter(fn(lambda z: z % 2 == 0, list(xs)))
                        == Counter(x for x in xs if x % 2 == 0)
                        for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "RELATIVE_ORDER_PRESERVATION: output is an order-preserving subsequence of X",
                    lambda fn, _: all(
                        fn(lambda z: z >= 0, list(xs)) == [x for x in xs if x >= 0]
                        for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "IDEMPOTENCE: forall p,X: filter(p, filter(p, X)) == filter(p, X)",
                    lambda fn, _: all(
                        fn(lambda z: z > 0, fn(lambda z: z > 0, list(xs)))
                        == fn(lambda z: z > 0, list(xs))
                        for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "COMPLEMENT_COUNT_PARTITION: |filter(p, X)| + |filter(not p, X)| == |X|",
                    lambda fn, _: all(
                        len(fn(lambda z: z > 0, list(xs)))
                        + len(fn(lambda z: not (z > 0), list(xs)))
                        == len(xs)
                        for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "CONJUNCTION_COMMUTATIVITY: filter(p1, filter(p2, X)) == filter(p1 & p2, X)",
                    lambda fn, _: all(
                        fn(lambda z: z > 0, fn(lambda z: z % 2 == 0, list(xs)))
                        == fn(lambda z: z > 0 and z % 2 == 0, list(xs))
                        for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "NON_MUTATION_OF_INPUT: filter(p, X) does not mutate X",
                    lambda fn, _: (lambda a: (fn(lambda z: z > 0, a), a == [-1, 2, -3])[1])([-1, 2, -3]),
                ),
            ],
            reference_fn=lambda p, xs: [x for x in xs if p(x)],
            mutants={
                "reject_all": lambda p, xs: [],
                "dedup_filter": lambda p, xs: list(dict.fromkeys(x for x in xs if p(x))),
                "sorted_filter": lambda p, xs: sorted(x for x in xs if p(x)),
                "invert_pred": lambda p, xs: [x for x in xs if not p(x)],
                "first_match_only": lambda p, xs: [x for x in xs if p(x)][:1],
            },
        ),
        LexicalSpec(
            word="fold",
            tier=2,
            builds_on=["set", "step"],
            signature="((A, T) -> A, A, Seq[T]) -> A",
            concise_transition="a_0 = init; a_{i+1} = op(a_i, X[i]); return a_{|X|}",
            clauses=[
                SymbolicClause(
                    "EMPTY_BASE_LAW: forall op,init: fold(op, init, []) == init",
                    lambda fn, _: all(fn(lambda a, x: a + x, init, []) == init for init in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "SINGLETON_LAW: forall op,init,x: fold(op, init, [x]) == op(init, x)",
                    lambda fn, _: all(
                        fn(lambda a, x: a * 3 - x, init, [x]) == init * 3 - x
                        for init in (-2, 0, 5)
                        for x in (-3, 0, 7)
                    ),
                ),
                SymbolicClause(
                    "LEFT_ASSOCIATIVE_STEP_LAW: fold(op, init, X ++ [y]) == op(fold(op, init, X), y)",
                    lambda fn, _: all(
                        fn(lambda a, x: 2 * a + x, 1, list(xs) + [9])
                        == 2 * fn(lambda a, x: 2 * a + x, 1, list(xs)) + 9
                        for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "SEQUENCE_RECONSTRUCTION_UNIVERSALITY: fold((acc, x) => acc ++ [x], [], X) == X",
                    lambda fn, _: all(
                        fn(lambda acc, x: acc + [x], [], list(xs)) == list(xs)
                        for xs in SEQ_GRID
                    ),
                ),
            ],
            reference_fn=lambda op, init, xs: _fold_left(op, init, xs),
            mutants={
                "return_init": lambda op, init, xs: init,
                "right_fold": lambda op, init, xs: _fold_right(op, init, xs),
                "ignore_init": lambda op, init, xs: _fold_left(op, 0, xs) if xs else 0,
                "skip_last": lambda op, init, xs: _fold_left(op, init, xs[:-1]),
            },
        ),
        LexicalSpec(
            word="any",
            tier=2,
            builds_on=["fold", "or"],
            signature="(T -> B, Seq[T]) -> B",
            concise_transition="y = fold((a, x) => or(a, p(x)), False, X)",
            clauses=[
                SymbolicClause(
                    "STRICT_BOOL_CODOMAIN: type(any(p, X)) is bool",
                    lambda fn, _: all(type(fn(lambda z: z > 0, list(xs))) is bool for xs in SEQ_GRID),
                ),
                SymbolicClause(
                    "EMPTY_FALSE_LAW: forall p: any(p, []) is False",
                    lambda fn, _: fn(lambda z: True, []) is False and fn(lambda z: False, []) is False,
                ),
                SymbolicClause(
                    "SINGLETON_LAW: forall p,x: any(p, [x]) == bool(p(x))",
                    lambda fn, _: all(fn(lambda z: z > 0, [x]) is (x > 0) for x in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "CONCATENATION_DISJUNCTION: any(p, A ++ B) == (any(p, A) or any(p, B))",
                    lambda fn, _: all(
                        fn(lambda z: z > 2, list(a) + list(b))
                        is (fn(lambda z: z > 2, list(a)) or fn(lambda z: z > 2, list(b)))
                        for a in SEQ_GRID[:5]
                        for b in SEQ_GRID[:5]
                    ),
                ),
                SymbolicClause(
                    "FILTER_NONEMPTY_EQUIVALENCE: any(p, X) == (len([x in X | p(x)]) > 0)",
                    lambda fn, _: all(
                        fn(lambda z: z < 0, list(xs)) is (len([x for x in xs if x < 0]) > 0)
                        for xs in SEQ_GRID
                    ),
                ),
            ],
            reference_fn=lambda p, xs: any(bool(p(x)) for x in xs),
            mutants={
                "all_instead": lambda p, xs: all(bool(p(x)) for x in xs),
                "empty_true": lambda p, xs: True if not xs else any(bool(p(x)) for x in xs),
                "check_first_only": lambda p, xs: bool(p(xs[0])) if xs else False,
                "check_last_only": lambda p, xs: bool(p(xs[-1])) if xs else False,
            },
        ),
        LexicalSpec(
            word="all",
            tier=2,
            builds_on=["fold", "and"],
            signature="(T -> B, Seq[T]) -> B",
            concise_transition="y = fold((a, x) => and(a, p(x)), True, X)",
            clauses=[
                SymbolicClause(
                    "STRICT_BOOL_CODOMAIN: type(all(p, X)) is bool",
                    lambda fn, _: all(type(fn(lambda z: z > 0, list(xs))) is bool for xs in SEQ_GRID),
                ),
                SymbolicClause(
                    "EMPTY_VACUOUS_TRUTH: forall p: all(p, []) is True",
                    lambda fn, _: fn(lambda z: False, []) is True and fn(lambda z: True, []) is True,
                ),
                SymbolicClause(
                    "SINGLETON_LAW: forall p,x: all(p, [x]) == bool(p(x))",
                    lambda fn, _: all(fn(lambda z: z > 0, [x]) is (x > 0) for x in SMALL_INT_GRID),
                ),
                SymbolicClause(
                    "CONCATENATION_CONJUNCTION: all(p, A ++ B) == (all(p, A) and all(p, B))",
                    lambda fn, _: all(
                        fn(lambda z: z >= 0, list(a) + list(b))
                        is (fn(lambda z: z >= 0, list(a)) and fn(lambda z: z >= 0, list(b)))
                        for a in SEQ_GRID[:5]
                        for b in SEQ_GRID[:5]
                    ),
                ),
                SymbolicClause(
                    "DE_MORGAN_QUANTIFIER_DUALITY: all(p, X) == (not any(not p, X))",
                    lambda fn, _: all(
                        fn(lambda z: z > 0, list(xs)) is (not any(not (x > 0) for x in xs))
                        for xs in SEQ_GRID
                    ),
                ),
            ],
            reference_fn=lambda p, xs: all(bool(p(x)) for x in xs),
            mutants={
                "any_instead": lambda p, xs: any(bool(p(x)) for x in xs),
                "empty_false": lambda p, xs: False if not xs else all(bool(p(x)) for x in xs),
                "check_first_only": lambda p, xs: bool(p(xs[0])) if xs else True,
            },
        ),
        LexicalSpec(
            word="count",
            tier=2,
            builds_on=["fold", "if", "step"],
            signature="(T -> B, Seq[T]) -> Z",
            concise_transition="y = fold((n, x) => if(p(x), step(n), n), 0, X)",
            clauses=[
                SymbolicClause(
                    "EMPTY_ZERO: forall p: count(p, []) == 0",
                    lambda fn, _: fn(lambda z: True, []) == 0,
                ),
                SymbolicClause(
                    "BOUNDED_BY_LENGTH: forall p,X: 0 <= count(p, X) <= |X|",
                    lambda fn, _: all(0 <= fn(lambda z: z > 0, list(xs)) <= len(xs) for xs in SEQ_GRID),
                ),
                SymbolicClause(
                    "FILTER_CARDINALITY_IDENTITY: count(p, X) == |filter(p, X)|",
                    lambda fn, _: all(
                        fn(lambda z: z > 0, list(xs)) == len([x for x in xs if x > 0])
                        for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "COMPLEMENT_SUM_LAW: count(p, X) + count(not p, X) == |X|",
                    lambda fn, _: all(
                        fn(lambda z: z % 2 == 0, list(xs)) + fn(lambda z: z % 2 != 0, list(xs)) == len(xs)
                        for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "ADDITIVITY_OVER_CONCATENATION: count(p, A ++ B) == count(p, A) + count(p, B)",
                    lambda fn, _: all(
                        fn(lambda z: z > 0, list(a) + list(b))
                        == fn(lambda z: z > 0, list(a)) + fn(lambda z: z > 0, list(b))
                        for a in SEQ_GRID[:5]
                        for b in SEQ_GRID[:5]
                    ),
                ),
            ],
            reference_fn=lambda p, xs: sum(1 for x in xs if p(x)),
            mutants={
                "total_len": lambda p, xs: len(xs),
                "unique_count": lambda p, xs: len({x for x in xs if p(x)}),
                "off_by_one_empty": lambda p, xs: sum(1 for x in xs if p(x)) + (1 if not xs else 0),
                "sum_values_instead": lambda p, xs: sum(x for x in xs if p(x)),
            },
        ),
        LexicalSpec(
            word="find",
            tier=2,
            builds_on=["if", "step"],
            signature="(T -> B, Seq[T]) -> T | None",
            concise_transition="y = first x in X where p(x) == True else None",
            clauses=[
                SymbolicClause(
                    "EMPTY_NONE: forall p: find(p, []) is None",
                    lambda fn, _: fn(lambda z: True, []) is None,
                ),
                SymbolicClause(
                    "MISS_RETURNS_NONE: forall p,X: (not any(p, X)) => (find(p, X) is None)",
                    lambda fn, _: all(
                        fn(lambda z: z > 999, list(xs)) is None for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "HIT_SATISFIES_PREDICATE: forall p,X: any(p, X) => p(find(p, X)) is True",
                    lambda fn, _: all(
                        fn(lambda z: z > 0, list(xs)) > 0
                        for xs in SEQ_GRID
                        if any(x > 0 for x in xs)
                    ),
                ),
                SymbolicClause(
                    "EARLIEST_INDEX_PRIORITY: find(p, X) == filter(p, X)[0] whenever any(p, X)",
                    lambda fn, _: all(
                        fn(lambda z: z % 2 != 0, list(xs)) == [x for x in xs if x % 2 != 0][0]
                        for xs in SEQ_GRID
                        if any(x % 2 != 0 for x in xs)
                    ),
                ),
                SymbolicClause(
                    "DISTINGUISHES_ZERO_FROM_NONE: find(x => x == 0, [5, 0, 3]) == 0 (is not None)",
                    lambda fn, _: fn(lambda z: z == 0, [5, 0, 3]) == 0
                    and fn(lambda z: z == 0, [5, 0, 3]) is not None,
                ),
            ],
            reference_fn=lambda p, xs: next((x for x in xs if p(x)), None),
            mutants={
                "find_last": lambda p, xs: next((x for x in reversed(xs) if p(x)), None),
                "return_zero_on_miss": lambda p, xs: next((x for x in xs if p(x)), 0),
                "falsy_zero_bug": lambda p, xs: next((x for x in xs if p(x) and x), None),
            },
        ),
        LexicalSpec(
            word="reverse",
            tier=2,
            builds_on=["swap", "step"],
            signature="Seq[T] -> Seq[T]",
            concise_transition="Y[i] = X[|X| - 1 - i] forall 0 <= i < |X|",
            clauses=[
                SymbolicClause(
                    "LENGTH_PRESERVATION: forall X: |reverse(X)| == |X|",
                    lambda fn, _: all(len(fn(list(xs))) == len(xs) for xs in SEQ_GRID),
                ),
                SymbolicClause(
                    "INDEX_REFLECTION_LAW: forall X,0<=i<|X|: reverse(X)[i] == X[|X| - 1 - i]",
                    lambda fn, _: all(
                        all(fn(list(xs))[i] == xs[len(xs) - 1 - i] for i in range(len(xs)))
                        for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "INVOLUTION: forall X: reverse(reverse(X)) == X",
                    lambda fn, _: all(fn(fn(list(xs))) == list(xs) for xs in SEQ_GRID),
                ),
                SymbolicClause(
                    "ANTI_HOMOMORPHISM_CONCAT: reverse(A ++ B) == reverse(B) ++ reverse(A)",
                    lambda fn, _: all(
                        fn(list(a) + list(b)) == fn(list(b)) + fn(list(a))
                        for a in SEQ_GRID[:5]
                        for b in SEQ_GRID[:5]
                    ),
                ),
                SymbolicClause(
                    "NON_MUTATION_OF_INPUT: reverse(X) does not mutate X in-place",
                    lambda fn, _: (lambda a: (fn(a), a == [1, 2, 3])[1])([1, 2, 3]),
                ),
            ],
            reference_fn=lambda xs: list(reversed(xs)),
            mutants={
                "identity": lambda xs: list(xs),
                "drop_first": lambda xs: list(reversed(xs[1:])),
                "swap_ends_only": lambda xs: [xs[-1]] + list(xs[1:-1]) + [xs[0]] if len(xs) >= 2 else list(xs),
                "in_place_reverse_bug": lambda xs: (xs.reverse(), xs)[1],
            },
        ),
        LexicalSpec(
            word="unique",
            tier=2,
            builds_on=["filter", "any", "equal"],
            signature="Seq[T] -> Seq[T]",
            concise_transition="Y = [X[i] | not(any(y => equal(y, X[i]), X[0:i]))]",
            clauses=[
                SymbolicClause(
                    "PAIRWISE_DISTINCT_ELEMENTS: forall X: |unique(X)| == |set(unique(X))|",
                    lambda fn, _: all(len(fn(list(xs))) == len(set(fn(list(xs)))) for xs in SEQ_GRID),
                ),
                SymbolicClause(
                    "EXACT_SET_COVERAGE: forall X: set(unique(X)) == set(X)",
                    lambda fn, _: all(set(fn(list(xs))) == set(xs) for xs in SEQ_GRID),
                ),
                SymbolicClause(
                    "FIRST_OCCURRENCE_ORDER_PRESERVED: index of first appearance in X is strictly increasing",
                    lambda fn, _: all(
                        [xs.index(y) for y in fn(list(xs))]
                        == sorted([xs.index(y) for y in fn(list(xs))])
                        for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "IDEMPOTENCE: forall X: unique(unique(X)) == unique(X)",
                    lambda fn, _: all(fn(fn(list(xs))) == fn(list(xs)) for xs in SEQ_GRID),
                ),
                SymbolicClause(
                    "NON_MUTATION_OF_INPUT: unique(X) does not mutate X",
                    lambda fn, _: (lambda a: (fn(a), a == [3, 1, 3])[1])([3, 1, 3]),
                ),
            ],
            reference_fn=lambda xs: list(dict.fromkeys(xs)),
            mutants={
                "sorted_set": lambda xs: sorted(list(set(xs))),
                "keep_dups": lambda xs: list(xs),
                "last_occurrence_order": lambda xs: list(reversed(list(dict.fromkeys(reversed(xs))))),
                "drop_all_repeated_items": lambda xs: [x for x in xs if xs.count(x) == 1],
            },
        ),
        LexicalSpec(
            word="slice",
            tier=2,
            builds_on=["clamp", "step"],
            signature="(Seq[T], Z, Z) -> Seq[T]",
            concise_transition="s = clamp(start, 0, |X|); e = clamp(end, s, |X|); Y = X[s:e]",
            clauses=[
                SymbolicClause(
                    "EXACT_CLAMPED_LENGTH: forall X,s,e: |slice(X, s, e)| == clamp(e, clamp(s,0,|X|), |X|) - clamp(s,0,|X|)",
                    lambda fn, _: all(
                        len(fn(list(xs), s, e))
                        == max(max(0, min(s, len(xs))), min(e, len(xs))) - max(0, min(s, len(xs)))
                        for xs in SEQ_GRID[:6]
                        for s in (-2, 0, 1, 3, 10)
                        for e in (-2, 0, 2, 4, 10)
                    ),
                ),
                SymbolicClause(
                    "CONTIGUOUS_ELEMENT_MAPPING: forall valid 0<=s<=e<=|X|, 0<=i<e-s: slice(X,s,e)[i] == X[s+i]",
                    lambda fn, _: all(
                        fn(list(xs), s, e) == list(xs[s:e])
                        for xs in SEQ_GRID
                        for s in range(0, len(xs) + 1)
                        for e in range(s, len(xs) + 1)
                    ),
                ),
                SymbolicClause(
                    "INVERTED_BOUNDS_EMPTY: forall X, s >= e: slice(X, s, e) == []",
                    lambda fn, _: all(
                        fn(list(xs), s, e) == []
                        for xs in SEQ_GRID
                        for s, e in ((3, 1), (2, 2), (5, 0), (0, -3))
                    ),
                ),
                SymbolicClause(
                    "ADJACENT_SPLIT_RECONSTRUCTION: forall 0<=k<=|X|: slice(X, 0, k) ++ slice(X, k, |X|) == X",
                    lambda fn, _: all(
                        fn(list(xs), 0, k) + fn(list(xs), k, len(xs)) == list(xs)
                        for xs in SEQ_GRID
                        for k in range(0, len(xs) + 1)
                    ),
                ),
                SymbolicClause(
                    "FULL_SPAN_IDENTITY: forall X: slice(X, 0, |X|) == X",
                    lambda fn, _: all(fn(list(xs), 0, len(xs)) == list(xs) for xs in SEQ_GRID),
                ),
            ],
            reference_fn=lambda xs, s, e: list(
                xs[max(0, min(s, len(xs))) : max(max(0, min(s, len(xs))), min(e, len(xs)))]
            ),
            mutants={
                "full_copy": lambda xs, s, e: list(xs),
                "inclusive_end": lambda xs, s, e: list(xs[max(0, s) : e + 1]),
                "python_negative_index_wrap_bug": lambda xs, s, e: list(xs[s:e]),
                "swap_bounds_when_inverted": lambda xs, s, e: list(xs[min(max(0, s), max(0, e)) : max(max(0, s), max(0, e))]),
            },
        ),
        # ===================================================================
        # TIER 3: COMPOSITIONAL DATA-STRUCTURE & TRANSACTION WORDS (10 words)
        # ===================================================================
        LexicalSpec(
            word="sort",
            tier=3,
            builds_on=["less", "swap", "fold"],
            signature="Seq[Z] -> Seq[Z]",
            concise_transition="Y is the unique non-decreasing permutation of X",
            clauses=[
                SymbolicClause(
                    "MULTISET_PERMUTATION_INVARIANT: forall X: Counter(sort(X)) == Counter(X)",
                    lambda fn, _: all(Counter(fn(list(xs))) == Counter(xs) for xs in SEQ_GRID),
                ),
                SymbolicClause(
                    "MONOTONIC_NON_DECREASING: forall X, 0 <= i < |X|-1: sort(X)[i] <= sort(X)[i+1]",
                    lambda fn, _: all(
                        all(y[i] <= y[i + 1] for i in range(len(y) - 1))
                        for xs in SEQ_GRID
                        for y in [fn(list(xs))]
                    ),
                ),
                SymbolicClause(
                    "IDEMPOTENCE: forall X: sort(sort(X)) == sort(X)",
                    lambda fn, _: all(fn(fn(list(xs))) == fn(list(xs)) for xs in SEQ_GRID),
                ),
                SymbolicClause(
                    "REVERSE_INVARIANCE: forall X: sort(reverse(X)) == sort(X)",
                    lambda fn, _: all(fn(list(reversed(xs))) == fn(list(xs)) for xs in SEQ_GRID),
                ),
                SymbolicClause(
                    "NON_MUTATION_OF_INPUT: sort(X) does not mutate X in-place",
                    lambda fn, _: (lambda a: (fn(a), a == [3, 1, 2])[1])([3, 1, 2]),
                ),
            ],
            reference_fn=lambda xs: sorted(xs),
            mutants={
                "descending_sort": lambda xs: sorted(xs, reverse=True),
                "unique_sort_drops_dups": lambda xs: sorted(set(xs)),
                "identity": lambda xs: list(xs),
                "in_place_sort_bug": lambda xs: (xs.sort(), xs)[1],
            },
        ),
        LexicalSpec(
            word="zip",
            tier=3,
            builds_on=["map", "slice", "cap"],
            signature="(Seq[T], Seq[U]) -> Seq[(T, U)]",
            concise_transition="Y[i] = (A[i], B[i]) forall 0 <= i < min(|A|, |B|)",
            clauses=[
                SymbolicClause(
                    "MIN_LENGTH_LAW: forall A,B: |zip(A, B)| == min(|A|, |B|)",
                    lambda fn, _: all(
                        len(fn(list(a), list(b))) == min(len(a), len(b))
                        for a in SEQ_GRID[:6]
                        for b in SEQ_GRID[:6]
                    ),
                ),
                SymbolicClause(
                    "INDEXWISE_PAIRING: forall A,B, 0<=i<min(|A|,|B|): zip(A, B)[i] == (A[i], B[i])",
                    lambda fn, _: all(
                        fn(list(a), list(b)) == [(a[i], b[i]) for i in range(min(len(a), len(b)))]
                        for a in SEQ_GRID[:6]
                        for b in SEQ_GRID[:6]
                    ),
                ),
                SymbolicClause(
                    "EMPTY_ANNIHILATION: zip([], B) == [] and zip(A, []) == []",
                    lambda fn, _: fn([], [1, 2]) == [] and fn([1, 2], []) == [],
                ),
                SymbolicClause(
                    "SWAP_COMMUTATIVITY: map(swap, zip(A, B)) == zip(B, A)",
                    lambda fn, _: all(
                        [(y, x) for (x, y) in fn(list(a), list(b))] == fn(list(b), list(a))
                        for a in SEQ_GRID[:5]
                        for b in SEQ_GRID[:5]
                    ),
                ),
            ],
            reference_fn=lambda a, b: list(zip(a, b)),
            mutants={
                "zip_reversed_b": lambda a, b: list(zip(a, reversed(b))),
                "drop_last_pair": lambda a, b: list(zip(a, b))[:-1],
                "swapped_columns": lambda a, b: [(y, x) for x, y in zip(a, b)],
            },
        ),
        LexicalSpec(
            word="partition",
            tier=3,
            builds_on=["filter", "not"],
            signature="(T -> B, Seq[T]) -> (Seq[T], Seq[T])",
            concise_transition="(Y_pass, Y_fail) = (filter(p, X), filter(not o p, X))",
            clauses=[
                SymbolicClause(
                    "EXACT_FILTER_DECOMPOSITION: partition(p, X) == (filter(p, X), filter(not p, X))",
                    lambda fn, _: all(
                        fn(lambda z: z >= 0, list(xs))
                        == ([x for x in xs if x >= 0], [x for x in xs if not (x >= 0)])
                        for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "MULTISET_CONSERVATION: Counter(pass ++ fail) == Counter(X)",
                    lambda fn, _: all(
                        Counter(fn(lambda z: z % 2 == 0, list(xs))[0] + fn(lambda z: z % 2 == 0, list(xs))[1])
                        == Counter(xs)
                        for xs in SEQ_GRID
                    ),
                ),
                SymbolicClause(
                    "NEGATION_SWAP_DUALITY: partition(not p, X) == swap(*partition(p, X))",
                    lambda fn, _: all(
                        fn(lambda z: not (z > 0), list(xs))
                        == (fn(lambda z: z > 0, list(xs))[1], fn(lambda z: z > 0, list(xs))[0])
                        for xs in SEQ_GRID
                    ),
                ),
            ],
            reference_fn=lambda p, xs: ([x for x in xs if p(x)], [x for x in xs if not p(x)]),
            mutants={
                "swapped_halves": lambda p, xs: ([x for x in xs if not p(x)], [x for x in xs if p(x)]),
                "dedup_halves": lambda p, xs: (list(dict.fromkeys(x for x in xs if p(x))), [x for x in xs if not p(x)]),
                "empty_second": lambda p, xs: ([x for x in xs if p(x)], []),
            },
        ),
        LexicalSpec(
            word="take",
            tier=3,
            builds_on=["slice", "clamp"],
            signature="(Z, Seq[T]) -> Seq[T]",
            concise_transition="Y = slice(X, 0, clamp(n, 0, |X|))",
            clauses=[
                SymbolicClause(
                    "CLAMPED_LENGTH_LAW: forall n,X: |take(n, X)| == clamp(n, 0, |X|)",
                    lambda fn, _: all(
                        len(fn(n, list(xs))) == max(0, min(n, len(xs)))
                        for xs in SEQ_GRID
                        for n in (-3, 0, 1, 2, 5, 10)
                    ),
                ),
                SymbolicClause(
                    "PREFIX_MATCH_LAW: forall n,X, 0<=i<|take(n,X)|: take(n, X)[i] == X[i]",
                    lambda fn, _: all(
                        fn(n, list(xs)) == list(xs[: max(0, n)])
                        for xs in SEQ_GRID
                        for n in (-2, 0, 1, 3, 10)
                    ),
                ),
                SymbolicClause(
                    "NON_POSITIVE_EMPTY: forall n <= 0, X: take(n, X) == []",
                    lambda fn, _: all(fn(n, list(xs)) == [] for xs in SEQ_GRID for n in (-5, -1, 0)),
                ),
                SymbolicClause(
                    "MIN_COMPOSITION_LAW: forall m>=0, n>=0, X: take(m, take(n, X)) == take(min(m, n), X)",
                    lambda fn, _: all(
                        fn(m, fn(n, list(xs))) == fn(min(m, n), list(xs))
                        for xs in SEQ_GRID
                        for m in (0, 2, 4)
                        for n in (1, 3, 5)
                    ),
                ),
            ],
            reference_fn=lambda n, xs: list(xs[: max(0, min(n, len(xs)))]),
            mutants={
                "python_neg_slice_bug": lambda n, xs: list(xs[:n]),
                "drop_instead": lambda n, xs: list(xs[max(0, n) :]),
                "take_from_end": lambda n, xs: list(xs[len(xs) - max(0, min(n, len(xs))) :]),
            },
        ),
        LexicalSpec(
            word="drop",
            tier=3,
            builds_on=["slice", "clamp"],
            signature="(Z, Seq[T]) -> Seq[T]",
            concise_transition="Y = slice(X, clamp(n, 0, |X|), |X|)",
            clauses=[
                SymbolicClause(
                    "REMAINDER_LENGTH_LAW: forall n,X: |drop(n, X)| == |X| - clamp(n, 0, |X|)",
                    lambda fn, _: all(
                        len(fn(n, list(xs))) == len(xs) - max(0, min(n, len(xs)))
                        for xs in SEQ_GRID
                        for n in (-3, 0, 1, 2, 5, 10)
                    ),
                ),
                SymbolicClause(
                    "TAKE_DROP_RECONSTRUCTION: forall n,X: take(n, X) ++ drop(n, X) == X",
                    lambda fn, _: all(
                        list(xs[: max(0, min(n, len(xs)))]) + fn(n, list(xs)) == list(xs)
                        for xs in SEQ_GRID
                        for n in (-3, 0, 1, 2, 5, 10)
                    ),
                ),
                SymbolicClause(
                    "NON_POSITIVE_IDENTITY: forall n <= 0, X: drop(n, X) == X",
                    lambda fn, _: all(fn(n, list(xs)) == list(xs) for xs in SEQ_GRID for n in (-5, -1, 0)),
                ),
                SymbolicClause(
                    "ADDITIVE_COMPOSITION_LAW: forall m>=0, n>=0, X: drop(m, drop(n, X)) == drop(m + n, X)",
                    lambda fn, _: all(
                        fn(m, fn(n, list(xs))) == fn(m + n, list(xs))
                        for xs in SEQ_GRID
                        for m in (0, 1, 3)
                        for n in (0, 2, 4)
                    ),
                ),
            ],
            reference_fn=lambda n, xs: list(xs[max(0, min(n, len(xs))) :]),
            mutants={
                "python_neg_slice_bug": lambda n, xs: list(xs[n:]),
                "take_instead": lambda n, xs: list(xs[: max(0, n)]),
                "drop_from_end": lambda n, xs: list(xs[: len(xs) - max(0, min(n, len(xs)))]),
            },
        ),
        LexicalSpec(
            word="push",
            tier=3,
            builds_on=["step", "set"],
            signature="(Seq[T], T) -> Seq[T]",
            concise_transition="Y = X ++ [v]",
            clauses=[
                SymbolicClause(
                    "LENGTH_STEP_LAW: forall X,v: |push(X, v)| == |X| + 1",
                    lambda fn, _: all(len(fn(list(xs), 99)) == len(xs) + 1 for xs in SEQ_GRID),
                ),
                SymbolicClause(
                    "LAST_ELEMENT_LAW: forall X,v: push(X, v)[-1] == v",
                    lambda fn, _: all(fn(list(xs), v)[-1] == v for xs in SEQ_GRID for v in (-7, 0, 42)),
                ),
                SymbolicClause(
                    "PREFIX_FRAME_PRESERVATION: forall X,v: push(X, v)[:-1] == X",
                    lambda fn, _: all(fn(list(xs), 99)[:-1] == list(xs) for xs in SEQ_GRID),
                ),
                SymbolicClause(
                    "NON_MUTATION_OF_INPUT: push(X, v) does not mutate caller's X in-place",
                    lambda fn, _: (lambda a: (fn(a, 9), a == [1, 2])[1])([1, 2]),
                ),
            ],
            reference_fn=lambda xs, v: list(xs) + [v],
            mutants={
                "prepend_front": lambda xs, v: [v] + list(xs),
                "overwrite_last": lambda xs, v: list(xs[:-1]) + [v] if xs else [v],
                "in_place_append_bug": lambda xs, v: (xs.append(v), xs)[1],
            },
        ),
        LexicalSpec(
            word="pop",
            tier=3,
            builds_on=["slice", "if"],
            signature="Seq[T] -> (Seq[T], T | None)",
            concise_transition="if |X|==0: ([], None) else (X[0:|X|-1], X[|X|-1])",
            clauses=[
                SymbolicClause(
                    "EMPTY_UNDERFLOW_GUARD: pop([]) == ([], None)",
                    lambda fn, _: fn([]) == ([], None),
                ),
                SymbolicClause(
                    "NON_EMPTY_RECONSTRUCTION: forall |X|>0: pop(X)[0] ++ [pop(X)[1]] == X",
                    lambda fn, _: all(
                        fn(list(xs))[0] + [fn(list(xs))[1]] == list(xs)
                        for xs in SEQ_GRID
                        if len(xs) > 0
                    ),
                ),
                SymbolicClause(
                    "PUSH_POP_INVERSE_LAW: forall X,v: pop(X ++ [v]) == (X, v)",
                    lambda fn, _: all(
                        fn(list(xs) + [v]) == (list(xs), v)
                        for xs in SEQ_GRID
                        for v in (-5, 0, 8)
                    ),
                ),
                SymbolicClause(
                    "NON_MUTATION_OF_INPUT: pop(X) does not mutate caller's X in-place",
                    lambda fn, _: (lambda a: (fn(a), a == [1, 2, 3])[1])([1, 2, 3]),
                ),
            ],
            reference_fn=lambda xs: (list(xs[:-1]), xs[-1]) if xs else ([], None),
            mutants={
                "pop_front": lambda xs: (list(xs[1:]), xs[0]) if xs else ([], None),
                "keep_last_in_rest": lambda xs: (list(xs), xs[-1]) if xs else ([], None),
                "in_place_pop_bug": lambda xs: (xs, xs.pop()) if xs else ([], None),
            },
        ),
        LexicalSpec(
            word="merge",
            tier=3,
            builds_on=["set", "fold"],
            signature="(Map[K,V], Map[K,V]) -> Map[K,V]",
            concise_transition="M[k] = (B[k] if k in B else A[k]) forall k in keys(A) U keys(B)",
            clauses=[
                SymbolicClause(
                    "KEY_SET_UNION: keys(merge(A, B)) == keys(A) U keys(B)",
                    lambda fn, _: set(fn({"a": 1, "b": 2}, {"b": 9, "c": 3}).keys()) == {"a", "b", "c"},
                ),
                SymbolicClause(
                    "RIGHT_BIAS_ON_COLLISION: forall k in B: merge(A, B)[k] == B[k]",
                    lambda fn, _: all(
                        fn({"a": 1, "b": 2}, {"b": v, "c": 3})["b"] == v
                        for v in (-5, 0, None, 42)
                    ),
                ),
                SymbolicClause(
                    "LEFT_RETENTION_ON_DISJOINT: forall k in A \\ B: merge(A, B)[k] == A[k]",
                    lambda fn, _: fn({"a": 1, "b": 2}, {"b": 9, "c": 3})["a"] == 1,
                ),
                SymbolicClause(
                    "EMPTY_IDENTITY_LEFT_AND_RIGHT: merge({}, A) == A and merge(A, {}) == A",
                    lambda fn, _: fn({}, {"a": 1}) == {"a": 1} and fn({"a": 1}, {}) == {"a": 1},
                ),
                SymbolicClause(
                    "ASSOCIATIVITY: merge(merge(A, B), C) == merge(A, merge(B, C))",
                    lambda fn, _: fn(fn({"a": 1, "b": 2}, {"b": 3, "c": 4}), {"c": 5, "d": 6})
                    == fn({"a": 1, "b": 2}, fn({"b": 3, "c": 4}, {"c": 5, "d": 6})),
                ),
                SymbolicClause(
                    "IDEMPOTENCE: merge(A, A) == A",
                    lambda fn, _: fn({"a": 1, "b": 2}, {"a": 1, "b": 2}) == {"a": 1, "b": 2},
                ),
                SymbolicClause(
                    "NON_MUTATION_OF_INPUTS: merge(A, B) mutates neither A nor B",
                    lambda fn, _: (
                        lambda a, b: (fn(a, b), a == {"x": 1} and b == {"x": 2, "y": 3})[1]
                    )({"x": 1}, {"x": 2, "y": 3}),
                ),
            ],
            reference_fn=lambda a, b: {**a, **b},
            mutants={
                "left_bias": lambda a, b: {**b, **a},
                "keep_only_b": lambda a, b: dict(b),
                "keep_only_a": lambda a, b: dict(a),
                "in_place_update_bug": lambda a, b: (a.update(b), a)[1],
            },
        ),
        LexicalSpec(
            word="cas",
            tier=3,
            builds_on=["if", "equal", "set"],
            signature="(T, T, T) -> (T, B)",
            concise_transition="if equal(cur, exp): (new, True) else (cur, False)",
            clauses=[
                SymbolicClause(
                    "MATCH_UPDATES_AND_REPORTS_TRUE: forall cur==exp, new: cas(cur, exp, new) == (new, True)",
                    lambda fn, _: all(
                        fn(v, v, nw) == (nw, True)
                        for v in SMALL_INT_GRID
                        for nw in SMALL_INT_GRID
                    ),
                ),
                SymbolicClause(
                    "MISMATCH_PRESERVES_AND_REPORTS_FALSE: forall cur!=exp, new: cas(cur, exp, new) == (cur, False)",
                    lambda fn, _: all(
                        fn(cur, exp, nw) == (cur, False)
                        for cur in SMALL_INT_GRID
                        for exp in SMALL_INT_GRID
                        if cur != exp
                        for nw in (-9, 0, 9)
                    ),
                ),
                SymbolicClause(
                    "SECOND_ATTEMPT_AFTER_CHANGE_FAILS: cur==exp != new => cas(cas(cur, exp, new)[0], exp, other) == (new, False)",
                    lambda fn, _: fn(fn(5, 5, 10)[0], 5, 99) == (10, False),
                ),
            ],
            reference_fn=lambda cur, exp, new: (new, True) if cur == exp else (cur, False),
            mutants={
                "unconditional_write": lambda cur, exp, new: (new, True),
                "invert_guard": lambda cur, exp, new: (new, True) if cur != exp else (cur, False),
                "wrong_flag_on_match": lambda cur, exp, new: (new, False) if cur == exp else (cur, False),
            },
        ),
        LexicalSpec(
            word="transfer",
            tier=3,
            builds_on=["if", "and", "less", "set"],
            signature="(Z, Z, Z) -> (Z, Z, B)",
            concise_transition="if (0 < amt <= a): (a - amt, b + amt, True) else (a, b, False)",
            clauses=[
                SymbolicClause(
                    "STRICT_SUM_CONSERVATION: forall a,b,amt: transfer(a,b,amt)[0] + transfer(a,b,amt)[1] == a + b",
                    lambda fn, _: all(
                        fn(a, b, amt)[0] + fn(a, b, amt)[1] == a + b
                        for a in (-5, 0, 10, 50)
                        for b in (-5, 0, 20)
                        for amt in (-10, 0, 5, 10, 15, 60)
                    ),
                ),
                SymbolicClause(
                    "FUNDED_POSITIVE_SUCCESS: forall a >= amt > 0: transfer(a, b, amt) == (a - amt, b + amt, True)",
                    lambda fn, _: all(
                        fn(a, b, amt) == (a - amt, b + amt, True)
                        for a in (5, 10, 50)
                        for b in (0, 20)
                        for amt in (1, 5, 10)
                        if a >= amt
                    ),
                ),
                SymbolicClause(
                    "INSUFFICIENT_FUNDS_ABORT: forall amt > a: transfer(a, b, amt) == (a, b, False)",
                    lambda fn, _: all(
                        fn(a, b, amt) == (a, b, False)
                        for a in (0, 5, 10)
                        for b in (0, 20)
                        for amt in (6, 11, 50)
                        if amt > a
                    ),
                ),
                SymbolicClause(
                    "NON_POSITIVE_AMOUNT_ABORT: forall amt <= 0: transfer(a, b, amt) == (a, b, False)",
                    lambda fn, _: all(
                        fn(a, b, amt) == (a, b, False)
                        for a in (0, 10, 50)
                        for b in (0, 20)
                        for amt in (-20, -1, 0)
                    ),
                ),
                SymbolicClause(
                    "NO_OVERDRAFT_INVARIANT: forall a >= 0, b, amt: transfer(a, b, amt)[0] >= 0",
                    lambda fn, _: all(
                        fn(a, b, amt)[0] >= 0
                        for a in (0, 5, 25)
                        for b in (0, 10)
                        for amt in (-5, 0, 5, 10, 30)
                    ),
                ),
            ],
            reference_fn=lambda a, b, amt: (
                (a - amt, b + amt, True) if (amt > 0 and a >= amt) else (a, b, False)
            ),
            mutants={
                "unfunded_overdraft": lambda a, b, amt: (a - amt, b + amt, True),
                "allows_zero_or_neg_amt": lambda a, b, amt: (a - amt, b + amt, True) if a >= amt else (a, b, False),
                "strict_greater_fails_full_balance": lambda a, b, amt: (a - amt, b + amt, True) if (amt > 0 and a > amt) else (a, b, False),
                "cap_partial_transfer": lambda a, b, amt: (
                    a - min(max(a, 0), max(amt, 0)),
                    b + min(max(a, 0), max(amt, 0)),
                    amt > 0 and a >= amt,
                ),
            },
        ),
    ]
    return specs


def _safe_check(clause: SymbolicClause, fn: Callable) -> bool:
    try:
        return bool(clause.check(fn, None))
    except Exception:
        return False


def verify_lexical_calculus() -> Dict[str, Any]:
    """Run exhaustive correctness verification across all 36 words (Tiers 0-3)."""
    specs = build_tiered_lexical_calculus()
    word_set = {s.word for s in specs}

    total_mutants = 0
    rejected_mutants = 0
    total_clauses = 0
    dag_valid = True

    tier_summary: List[Dict[str, Any]] = []

    for spec in specs:
        for dep in spec.builds_on:
            if dep not in word_set:
                dag_valid = False

        ref_passes = all(_safe_check(c, spec.reference_fn) for c in spec.clauses)

        spec_mutants_rejected = 0
        for m_name, m_fn in spec.mutants.items():
            total_mutants += 1
            passes_all = all(_safe_check(c, m_fn) for c in spec.clauses)
            if not passes_all:
                rejected_mutants += 1
                spec_mutants_rejected += 1

        total_clauses += len(spec.clauses)
        tier_summary.append(
            {
                "word": spec.word,
                "tier": spec.tier,
                "builds_on": spec.builds_on,
                "signature": spec.signature,
                "concise_transition": spec.concise_transition,
                "clause_count": len(spec.clauses),
                "reference_valid": ref_passes,
                "mutants_rejected": f"{spec_mutants_rejected}/{len(spec.mutants)}",
            }
        )

    # All-Pairs Cross-Word Non-Collision Matrix (36 x 35 = 1,260 off-diagonal pairs):
    # Every single word's reference implementation is tested against every other
    # word's formal specification with zero exceptions.
    cross_pairs_tested = 0
    cross_collisions = 0
    for s_spec in specs:
        for t_spec in specs:
            if s_spec.word == t_spec.word:
                continue
            cross_pairs_tested += 1
            if all(_safe_check(c, t_spec.reference_fn) for c in s_spec.clauses):
                cross_collisions += 1

    mutant_collision_rate = 1.0 - (rejected_mutants / max(1, total_mutants))
    return {
        "total_words": len(specs),
        "tier0_words": [s.word for s in specs if s.tier == 0],
        "tier1_words": [s.word for s in specs if s.tier == 1],
        "tier2_words": [s.word for s in specs if s.tier == 2],
        "tier3_words": [s.word for s in specs if s.tier == 3],
        "dag_hierarchy_valid": dag_valid,
        "all_reference_impls_pass": all(t["reference_valid"] for t in tier_summary),
        "total_mutants_tested": total_mutants,
        "mutants_rejected": rejected_mutants,
        "collision_rate": round(mutant_collision_rate, 4),
        "cross_word_pairs_tested": cross_pairs_tested,
        "cross_word_collisions": cross_collisions,
        "total_symbolic_clauses": total_clauses,
        "mean_clauses_per_word": round(total_clauses / len(specs), 2),
        "specs": tier_summary,
        "lexical_calculus_verified": bool(
            dag_valid
            and rejected_mutants == total_mutants
            and cross_collisions == 0
            and all(t["reference_valid"] for t in tier_summary)
        ),
    }

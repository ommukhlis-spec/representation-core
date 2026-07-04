# Design Decisions

**Why the framework is built the way it is.**

---

## 1. Why Quality is a Vector, not a Scalar

### Decision

$Q(R) = (C, H, I, E, X)$ — five dimensions. Ranking is partial order
(Pareto dominance).

### Rationale

Collapsing multiple dimensions into one scalar hides information and
forces premature trade-off decisions. Is a representation with lower
complexity but fewer invariants better? It DEPENDS on the downstream
task. The vector preserves this information.

### Counterfactual

If we used a single scalar (e.g., Tension), we would lose:
- The ability to detect when two representations are incomparable
- The ability to optimize for specific dimensions based on task
- Scientific transparency about WHY one R ranks above another

### Precedent

Multi-objective optimization (Pareto, 1906). The insight that quality
is inherently multidimensional is not new — but applying it to
representation evaluation IS.

---

## 2. Why Applicability is Separate from Quality

### Decision

$\alpha(R, T) \in [0,1]$ is computed independently from $Q(R)$.

### Rationale

Quality measures the *structural goodness* of a representation.
Applicability measures the *fit* between representation and task.

A representation can have high quality (well-defined, internally
consistent) but low applicability (wrong lens for this task).
Symmetry representation has high quality but α ≈ 0 on random grids.

Separating them allows:
- Filtering: don't even evaluate quality if α is too low
- Explanation: "this rep is good but not for THIS problem"
- Learning: α can be learned from task features

### Counterfactual

If we merged α into Q as a sixth dimension, we would conflate
"is this a good representation?" with "is this representation
good for THIS task?" — they are different questions.

---

## 3. Why Failure Modes are First-Class Citizens

### Decision

Every canonical task declares a `failure_mode`, and the
Failure Mode → Task → Representation mapping is explicit.

### Rationale

In most AI systems, failure is binary: "it worked" or "it didn't."
But HOW something fails tells you what to do next.

Object representation fails with `overlap_ambiguity` → try region.
Object representation fails with `non_discrete` → try topology or constraint.

The failure mode is the PUSH signal that drives representational
transition. Without structured failure modes, transitions are
random search over representations.

### Precedent

- VanLehn (1988): Impasse-driven learning uses impasse type to select repair
- Type theory: error messages as types that guide debugging
- Kuhn (1962): Anomalies drive paradigm shifts

---

## 4. Why Canonical Tasks, not just ARC

### Decision

The primary benchmark is 20 synthetic canonical tasks, not ARC.

### Rationale

ARC tasks are black-box: you don't know which representation is
"supposed" to work. A system might solve an ARC task using the
wrong representation for the wrong reason.

Canonical tasks have KNOWN ground truth. We declare which
representation SHOULD win and WHY. This makes evaluation
scientific rather than empirical.

ARC becomes the *external* benchmark — the one we use to see
if the framework generalizes beyond synthetic tasks. But the
*internal* validator is the canonical suite.

### Counterfactual

If we only used ARC, we would never know whether the framework
is correctly identifying the best representation. We would only
know whether the system solved the task.

---

## 5. Why the Admission Test is Non-Negotiable

### Decision

New representations MUST pass `admission_test()` to enter the zoo.

### Rationale

Without an admission gate, the zoo becomes a "collection of ideas"
— representations accumulated because they seemed interesting, not
because they were necessary.

The admission test enforces:
- **Unique value**: the new rep MUST solve something new
- **No regression**: the new rep MUST NOT break existing results

This is analogous to how LLVM optimization passes must improve
at least one benchmark without regressing any other.

### Counterfactual

Without admission, the zoo could grow to 50+ representations
with no way to distinguish essential from redundant. The
framework would lose its scientific credibility.

---

## 6. Why RCS is Continuous, not Binary

### Decision

`contribution_score()` returns a continuous score, not just accept/reject.

### Rationale

Binary admission ("accepted" or "rejected") is useful as a gate,
but doesn't distinguish between "barely useful" and "transformative."

RCS = coverage_gain - 2·redundancy - 0.5·complexity

This allows:
- Ranking: which new representation is most valuable?
- Budgeting: if we can only maintain N representations, which stay?
- Evolution: if a new rep subsumes an old one, the old one can be retired

### Precedent

- h-index: continuous score for researcher impact
- Contribution scores in open-source (lines changed, reviews, issues)

---

## 7. Why Python ABC, not Duck Typing

### Decision

`Representation` is an abstract base class with `@abstractmethod`.

### Rationale

The contract must be EXPLICIT. Every representation must implement
every method. If a method is missing, the error appears at import
time, not at runtime.

Duck typing ("if it quacks like a representation...") would allow
partial implementations that fail silently. The ABC enforces the
full contract.

### Counterfactual

Without ABC, a "representation" could be any object with an `encode`
method. Missing methods would cause runtime errors deep in the
evaluation pipeline, making debugging a nightmare.

---

## 8. Why Mathematics First, Code Second

### Decision

The formal document (`papers/formal-definition.md`) defines the theory.
The Python code is an *implementation*, not the definition.

### Rationale

If the theory is defined by the code, it cannot be reimplemented
in another language without reverse-engineering. If the theory is
defined mathematically, anyone can implement it in Rust, Julia,
Haskell, or Lean.

The mathematics also forces precision. "What IS a representation?"
must have a formal answer. If it can't be formalized, it's not yet
a theory.

### Precedent

- Relational algebra (Codd, 1970): defined mathematically before SQL
- Lambda calculus (Church, 1936): defined before Lisp

---

## 9. Why ARC-Specific Observation Type

### Decision

`Observation` is currently a 2D integer grid (ARC-native).

### Rationale

Good abstractions are born from concrete pressure. Starting with
a generic `Observation` type would produce an interface that is
abstract but semantically empty.

By starting ARC-specific, every design decision is forced by real
requirements. When the framework later generalizes to text, graphs,
or continuous signals, the generalization will be grounded in
experience rather than speculation.

### Counterfactual

A generic `Observation` from day one would be:
```python
class Observation:
    data: Any  # could be anything
```
This tells us nothing about what a representation must handle.

---

## 10. Why the Benchmark Report is Automated

### Decision

`python benchmarks/report.py` produces `report.md` and `leaderboard.json`.

### Rationale

Manual benchmark reports become stale. Automated reports are always
current. This is a CI gate: if coverage drops, merge is blocked.

### Precedent

- Rust's `rustc-perf` benchmark suite
- LLVM's `test-suite`
- Any modern CI pipeline

---

## Summary

| Decision | Principle |
|----------|-----------|
| Quality as vector | Don't collapse information prematurely |
| Separate applicability | Quality ≠ fitness for task |
| Failure modes as first-class | HOW you fail determines what to do next |
| Canonical tasks over ARC | Ground truth enables scientific evaluation |
| Admission test mandatory | Every representation must earn its existence |
| RCS continuous | Nuance matters; binary gates are coarse |
| ABC over duck typing | Explicit contracts prevent silent failures |
| Math first, code second | Theory must be language-independent |
| ARC-specific, not generic | Good abstractions require concrete pressure |
| Automated benchmark | Stale reports are useless reports |

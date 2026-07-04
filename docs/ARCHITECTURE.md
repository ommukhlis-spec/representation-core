# Architecture

**Representation Engineering: A Formal Framework for Adaptive Reasoning**

---

## The Big Picture

```
                          ┌──────────────────────────┐
                          │      OBSERVATION         │
                          │   (ARC grid, 2D array)   │
                          └──────────┬───────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │           REPRESENTATION ZOO                │
              │                                              │
              │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │
              │  │  Pixel  │ │ Object  │ │Symmetry │  ...  │
              │  └────┬────┘ └────┬────┘ └────┬────┘       │
              │       │           │           │             │
              │       ▼           ▼           ▼             │
              │  R = (S, φ, I, T)  ←── unified contract    │
              └──────────────────┬───────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ Quality  │ │Applicab. │ │ Failure  │
              │ Vector   │ │  α(R,T)  │ │Detection │
              │ Q=(C,H,  │ │          │ │          │
              │  I,E,X)  │ │          │ │          │
              └────┬─────┘ └────┬─────┘ └────┬─────┘
                   │            │            │
                   └────────────┼────────────┘
                                │
                                ▼
              ┌──────────────────────────────────────────────┐
              │           EVALUATION LAYER                   │
              │                                              │
              │  ┌──────────────────┐  ┌──────────────────┐ │
              │  │ Canonical Tasks  │  │ Transition Engine│ │
              │  │  (20 synthetic)  │  │  (Failure→NextR) │ │
              │  └────────┬─────────┘  └────────┬─────────┘ │
              │           │                     │           │
              │           ▼                     ▼           │
              │  ┌──────────────────┐  ┌──────────────────┐ │
              │  │ Admission Test   │  │ RCS (Contrib.    │ │
              │  │ (binary gate)    │  │  Score)          │ │
              │  └────────┬─────────┘  └────────┬─────────┘ │
              │           │                     │           │
              └───────────┼─────────────────────┼───────────┘
                          │                     │
                          ▼                     ▼
              ┌──────────────────────────────────────────────┐
              │           BENCHMARK REPORT                   │
              │                                              │
              │  report.md  +  leaderboard.json  +  CI gate  │
              └──────────────────────────────────────────────┘
```

---

## Layer Architecture

The framework is organized in four layers:

### Layer 1: Representation Contract

Defines WHAT a representation IS.

- `R = (S_R, φ_R, I_R, T_R)` — the mathematical structure
- `Representation` ABC — the operational contract
- Each concrete representation implements: encode, decode, complexity, invariants, hypothesis_space_size, estimated_reasoning_cost, applicability, failure_detect, transition_candidates, explain

### Layer 2: Metrics

Defines HOW to evaluate a representation.

- `QualityVector(C, H, I, E, X)` — multidimensional quality
- Partial order (Pareto dominance), not total order
- Scalar heuristics: tension, compression ratio, invariance density
- `Applicability α(R, T) ∈ [0, 1]` — how well R fits task T

### Layer 3: Dynamics

Defines HOW representations interact and transition.

- `FailureSignature` — structured description of WHY R fails
- `Transition` — movement from R_i to R_j guided by failure + invariants
- `Transition Engine` — deterministic push/pull based on failure modes

### Layer 4: Evaluation

Defines HOW to validate the entire framework.

- `CanonicalTask` — synthetic tasks with known ground truth
- `Admission Test` — binary gate for new representations
- `RCS` — continuous contribution score
- `Benchmark Report` — automated leaderboard + CI gate

---

## Data Flow

### During representation selection:

```
Observation
    │
    ▼
For each R in Zoo:
    α = R.applicability(obs)     ← is R even relevant?
    Q = compute_quality(R, obs)  ← how good is R?
    │
    ▼
Pareto front of (α, Q)           ← candidate representations
    │
    ▼
Select R* based on task requirements
    │
    ▼
Reason under R*
```

### When representation fails:

```
R_current + ReasoningProgress
    │
    ▼
R.failure_detect() → FailureSignature
    │
    ▼
FailureSignature.mode → candidate next representations
    │
    ├── PUSH (failure): "overlap" → try region
    ├── PULL (invariant): "symmetry detected" → try symmetry
    │
    ▼
R_next = select from candidates
    │
    ▼
Continue reasoning under R_next
```

### During admission of a new representation:

```
NewRepresentation
    │
    ▼
admission_test(R_new, R_existing, canonical_tasks)
    │
    ├── unique_value?  → does R_new solve something new?
    ├── no_regression? → does R_new break existing results?
    │
    ▼
contribution_score(R_new, R_existing, all_tasks)
    │
    ├── coverage_gain
    ├── redundancy
    ├── complexity_penalty
    │
    ▼
Verdict: strong_accept | weak_accept | reject
```

---

## Design Principles

1. **Mathematics first, code second.** Every concept has a formal definition.
   The Python implementation is one embodiment, not the definition.

2. **Metrics are vectors, not scalars.** Quality is multidimensional.
   Premature aggregation hides information.

3. **Ranking is partial order.** Not all representations are comparable.
   Pareto dominance is the correct comparison primitive.

4. **Representations earn their existence.** No representation enters the
   zoo without passing the admission test. Every representation must
   handle a failure mode that existing representations cannot.

5. **ARC is a laboratory, not the goal.** The framework is designed to
   generalize to any domain. ARC is the first experimental arena.

6. **Falsifiability is non-negotiable.** Every hypothesis (H−1 through H4)
   has a clear empirical test that could refute it.

---

## Key Abstractions

| Abstraction | Mathematical Form | Python Type |
|-------------|-------------------|-------------|
| Observation | $D \in \mathcal{O}$ | `Observation` |
| Representation | $R = (\mathcal{S}, \phi, \mathcal{I}, \mathcal{T})$ | `Representation` (ABC) |
| Invariant | $\iota \in \mathcal{I}_R$ | `Invariant` |
| Quality | $Q(R) = (C, H, I, E, X)$ | `QualityVector` |
| Applicability | $\alpha(R, T) \in [0, 1]$ | `float` |
| Failure | $(\text{mode}, \text{evidence}, \text{confidence})$ | `FailureSignature` |
| Transition | $\tau: R_i \xrightarrow{\sigma} R_j$ | `Transition` |
| Canonical Task | $(D, R_{\text{best}}, R_{\text{worst}}, \text{mode})$ | `CanonicalTask` |
| Contribution | $\text{RCS}(R) = \text{gain} - \text{redundancy} - \text{cost}$ | `contribution_score()` |

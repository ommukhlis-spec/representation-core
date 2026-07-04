# A Formal Framework for Representation Engineering

**Status:** Working Draft v0.2  
**Date:** 2026-07-04  
**Phase 0 — Foundation**

---

## Abstract

We propose **Representation Engineering** — a formal framework for
defining, measuring, comparing, selecting, and transforming the internal
structures through which an intelligent system encodes observations to
make reasoning tractable.

The central claim is that the bottleneck of modern AI is not search
capacity or model scale, but *representation selection*. We formalize
this as a set of falsifiable hypotheses grounded in measurable
quantities, and provide an operational definition of representation
that is implementation-independent: the mathematics defines the
structure; code (Python, Rust, Lean, or otherwise) is merely one
embodiment.

ARC-AGI serves as the first experimental arena — not the goal.

---

## 1. Motivation

### 1.1 The Problem

Given an observation $D \in \mathcal{O}$, an intelligent system must
produce a correct action. The naive approach is direct search:

$$D \rightarrow \text{Search}(\text{ProgramSpace}) \rightarrow \text{Action}$$

For a $H \times W$ grid with 10 colors:

$$|\mathcal{H}_{\text{raw}}| = 10^{HW}$$

For a $10 \times 10$ grid: $10^{100}$ — computationally infeasible.

Yet humans solve such tasks in seconds.

### 1.2 The Hypothesis

Humans do not search $10^{100}$. They *re-represent* the observation
into a structure where the effective hypothesis space collapses.
The key to intelligence is not better search — it is better
representation.

This is **falsifiable**: if varying the representation changes the
hypothesis space size and reasoning cost by orders of magnitude under
controlled conditions, the hypothesis is supported. If all
representations yield similar reasoning costs, it is refuted.

---

## 2. Formal Definition of Representation

**Important**: the definition below is the mathematical structure.
The Python class `Representation` in `representation-core` is an
*implementation* of this structure — not its definition. The theory
stands regardless of programming language.

### 2.1 Core Structure

Let:
- $\mathcal{O}$ be the space of observations
- $\mathcal{T}$ be the space of tasks

A **representation** $R$ is a 4-tuple:

$$\boxed{R = (\mathcal{S}_R, \phi_R, \mathcal{I}_R, \mathcal{T}_R)}$$

where:

| Symbol | Name | Definition |
|--------|------|------------|
| $\mathcal{S}_R$ | Symbolic space | The target space of the encoding |
| $\phi_R: \mathcal{O} \to \mathcal{S}_R$ | Encoding map | Maps observation to symbolic structure |
| $\mathcal{I}_R \subseteq \text{Prop}(\mathcal{O})$ | Invariants | Properties of $\mathcal{O}$ made explicit in $\mathcal{S}_R$ |
| $\mathcal{T}_R \subseteq \mathcal{S}_R \to \mathcal{S}_R$ | Admissible transformations | Operations closed on $\mathcal{S}_R$ |

### 2.2 Design Requirements

A representation must satisfy:

1. **Structural closure**: $\forall \tau \in \mathcal{T}_R, \forall s \in \mathcal{S}_R: \tau(s) \in \mathcal{S}_R$
2. **Invariant preservation**: $\forall \iota \in \mathcal{I}_R$: detecting $\iota$ in $\mathcal{S}_R$ is computationally trivial ($O(1)$ or $O(n)$ in $|\mathcal{S}_R|$)
3. **Loss control**: $\phi_R$ may be lossy, but $\mathcal{I}_R$ must be preserved under $\phi_R^{-1} \circ \phi_R$
4. **Explainability**: for any $s \in \mathcal{S}_R$, the mapping $\phi_R^{-1}(s)$ and the structural choices in $s$ must be human-interpretable

### 2.3 Representation Quality (Vector, not Scalar)

The quality of a representation is **not a single number**. It is a
vector:

$$\boxed{Q(R \mid D, T) = \begin{pmatrix} C \\ H \\ I \\ E \\ X \end{pmatrix}}$$

| Component | Name | Definition |
|-----------|------|------------|
| $C$ | Complexity | $C_{\text{rep}}(R \mid D)$ — cost of encoding $D$ under $R$ |
| $H$ | Hypothesis space | $\log_2|\mathcal{H}_R|$ — log-size of effective hypothesis space |
| $I$ | Invariance | $|\mathcal{I}_R|$ — number of invariants made explicit |
| $E$ | Explainability | Structural interpretability score |
| $X$ | Execution cost | $\hat{C}_{\text{reason}}(T \mid R)$ — estimated reasoning cost |

**Ranking is a partial order**, not a total order. $R_i \succ R_j$ only
if $R_i$ strictly dominates $R_j$ on at least one dimension and is not
worse on any other. When neither dominates, the choice depends on the
downstream task requirements.

### 2.4 Applicability

Every representation has a domain of applicability:

$$\boxed{\alpha(R, T) \in [0, 1]}$$

where $\alpha(R, T)$ is the **applicability** of representation $R$ to
task $T$ — the degree to which $R$'s structural assumptions match $T$'s
underlying regularities.

A representation built for symmetry detection ($R_{\text{sym}}$) has
high $\alpha$ on rotation tasks and low $\alpha$ on counting tasks.

Applicability is the heart of **adaptive reasoning**: the system does
not blindly apply one representation, nor does it randomly search over
representations. It evaluates which representations are *plausible*
given the structural signals detected in the observation.

---

## 3. Metrics and Measurement

### 3.1 Representation Cost

$$C_{\text{rep}}(R \mid D) = \ell(\phi_R(D))$$

where $\ell(\cdot)$ denotes description length in bits-equivalent.

For pixel representation: $C_{\text{rep}}(\text{pixel} \mid D) = HW \cdot \log_2(10)$

For object representation: $C_{\text{rep}}(\text{object} \mid D) \approx N_{\text{obj}} \cdot (\log_2(10) + c_{\text{shape}})$

### 3.2 Reasoning Cost (Estimated)

$$\hat{C}_{\text{reason}}(T \mid R) = \log_2\left(\frac{|\mathcal{H}_R|}{1 + |\mathcal{I}_R|}\right)$$

Each invariant constrains the effective search space.

### 3.3 Representation Tension (Heuristic Proxy)

$$\text{Tension}(R) = C_{\text{rep}}(R \mid D) + \alpha \cdot \log_2(|\mathcal{H}_R|)$$

where $\alpha$ is a weighting parameter (default: 0.5).

**Important qualification**: Tension is a heuristic proxy for
$C_{\text{reason}}$, not a theorem. Its validity MUST be empirically
validated by correlating tension with actual reasoning cost on benchmark
tasks. Until that validation is done, tension should be treated as an
exploratory metric, not a confirmed predictor.

### 3.4 Empirical Claim (Conservative)

Rather than claiming "Object representation is 5.7× better than pixel,"
the framework's valid empirical claim at this stage is:

> **Different representations induce measurably different reasoning
> characteristics under a common evaluation protocol.**

The exact ratios depend on metric definitions and task selection.
What matters is that the differences are *structural* (orders of
magnitude in $|\mathcal{H}|$) rather than incidental.

---

## 4. Representation Algebra

This section is the core theoretical contribution we are building toward.

### 4.1 Transition Operator

A **transition operator** is a map between representations:

$$\boxed{\tau: \mathcal{R} \times \Sigma \to \mathcal{R}}$$

where $\mathcal{R}$ is the space of representations and $\Sigma$ is the
*signal* triggering the transition (failure signature, emerging
invariant, or both).

$$\tau(R_i, \sigma) = R_j$$

### 4.2 Formal Properties of Transition Operators

Transitions can be classified by their algebraic properties:

| Property | Definition | Meaning |
|----------|------------|---------|
| **Deterministic** | $\forall \sigma: |\tau(R, \sigma)| = 1$ | Same signal always leads to same next $R$ |
| **Stochastic** | $\tau(R, \sigma) \sim P(R' \mid R, \sigma)$ | Signal biases but does not determine next $R$ |
| **Reversible** | $\exists \tau^{-1}: \tau^{-1}(\tau(R, \sigma), \sigma') = R$ | Can return to previous representation |
| **Irreversible** | No such $\tau^{-1}$ exists | Information is lost in transition (abstraction) |
| **Monotonic** | $Q(\tau(R)) \succeq Q(R)$ | Quality never decreases |
| **Cost-preserving** | $C_{\text{rep}}(\tau(R)) \leq C_{\text{rep}}(R) + \delta$ | Transition cost is bounded |
| **Invariant-preserving** | $\mathcal{I}_R \subseteq \mathcal{I}_{\tau(R)}$ | Accumulated invariants survive transition |
| **Idempotent** | $\tau(\tau(R, \sigma), \sigma) = \tau(R, \sigma)$ | Applying same transition twice has no extra effect |

### 4.3 Transition as Algebraic Structure

The space $(\mathcal{R}, \mathcal{T})$ where $\mathcal{T}$ is a set of
transition operators forms a **representation algebra** if:

1. **Closure**: $\forall \tau \in \mathcal{T}, \forall R \in \mathcal{R}: \tau(R) \in \mathcal{R}$
2. **Identity**: $\exists \iota \in \mathcal{T}: \iota(R) = R$ (staying in the same representation)
3. **Composability**: $\tau_2 \circ \tau_1$ is a valid transition sequence

Whether associativity, inverses, or other algebraic properties hold
depends on the specific representation space and is an **empirical
question** to be investigated.

### 4.4 Failure Signature as a Type

A **FailureSignature** is not merely an error code — it is a structured
description of *why* $R$ fails. Formally:

$$\text{FailureSignature} = (\text{mode}, \text{evidence}, \text{confidence})$$

where:
- $\text{mode} \in \mathcal{F}$ — a failure mode from the catalog $\mathcal{F}$
- $\text{evidence}$ — verifiable facts supporting the diagnosis
- $\text{confidence} \in [0, 1]$ — degree of certainty

The critical insight: the failure mode **determines the direction**
of the transition:

$$\text{overlap} \rightarrow \text{region}$$
$$\text{non\_discrete} \rightarrow \text{region or topology}$$
$$\text{partial\_symmetry} \rightarrow \text{topology or constraint}$$

This is a **push** signal. The complementary **pull** signal comes from
emerging invariants — structural regularities that become visible even
within the failing representation.

### 4.5 Transition Dynamics (Full)

$$R_{t+1} = \begin{cases}
R_t & \text{if } \text{Progress}(R_t, t) > \theta_{\text{stay}} \\[6pt]
\underset{R' \in \text{Neighborhood}(R_t, \sigma_t)}{\arg\min} \text{Tension}(R') & \text{if stagnation detected}
\end{cases}$$

where $\sigma_t = (\text{FailureSignature}_t, \text{EmergingInvariants}_t)$

and $\text{Neighborhood}(R, \sigma)$ is:

$$\{R' \in \mathcal{R} : \mathcal{I}_R \subseteq \mathcal{I}_{R'} \text{ and } d(R, R') < \epsilon\}$$

The distance $d(R_i, R_j) = |\mathcal{I}_{R_i} \triangle \mathcal{I}_{R_j}|$
measures invariant set difference.

---

## 5. Hypotheses

### H−1: Iterative Representational Refinement

> Intelligence is the capacity for iterative representational
> refinement guided by failure structure.

**Status**: Normative core.  
**Falsifiable**: Build two systems — one with iterative representation
refinement, one with a single fixed representation. If the former does
NOT outperform the latter on novel-task generalization, H−1 is refuted.

### H0: Minimum Total Cost

> $R^* = \arg\min_R \left[ C_{\text{rep}}(R \mid D) + C_{\text{reason}}(T \mid R) \right]$

**Status**: Special case of H−1 when $\mathcal{R}$ is fully enumerated.  
**Falsifiable**: Compare explicit cost minimization against random
representation baseline.

### H1: Representation Bottleneck

> The bottleneck of modern AI is representation, not search.

**Falsifiable**: Vary representation while fixing search budget, then
vary search budget while fixing representation. If the latter has
larger effect on task success, H1 is refuted.

### H2: No Canonical Representation

> There is no universal representation optimal for all tasks.

**Falsifiable**: Find a single representation that dominates all others
on >95% of diverse tasks. (Has not been found.)

### H3: Tension as Heuristic Proxy

> Representation Tension correlates with actual reasoning cost.

**Falsifiable**: Compute Tension($R$) for multiple $R$ across multiple
tasks. Measure actual reasoning cost (steps to solution). If
$\rho(\text{Tension}, C_{\text{reason}}) \approx 0$, H3 is refuted.

### H4: Applicability Predicts Performance

> $\alpha(R, T)$ correlates with reasoning success on task $T$.

**Falsifiable**: Rank representations by applicability for each task.
If the ranking does NOT predict which representation leads to the
fastest or most accurate solution, H4 is refuted.

---

## 6. Implementation

The `representation-core` Python library is *one embodiment* of this
framework. The mathematical definitions in this document are the
specification; the code is the operationalization.

### 6.1 Current Status (v0.2)

| Component | Status |
|-----------|--------|
| Abstract contract | ✅ `Representation` with full interface |
| Pixel representation | ✅ Baseline (identity encoding) |
| Object representation | ✅ Connected-component segmentation |
| Quality vector | ✅ Multidimensional metrics |
| Transition operators | ✅ Failure detection + candidate proposal |
| Applicability | ✅ Abstract method defined |
| ARC loader | ✅ JSON → Observation |

### 6.2 Initial Validation

The framework demonstrates that different representations induce
**measurably different reasoning characteristics** under a common
evaluation protocol. On structured ARC-like grids:

- Object representation reduces $|\mathcal{H}|$ by ~26 orders of
  magnitude vs. pixel
- Object representation detects 2× more invariants
- All metrics are finite, monotonic, and computable

These are *sanity checks*, not scientific claims. Full validation
requires the ARC dataset (Phase 4).

---

## 7. Open Questions

1. **Algebraic structure of $\mathcal{R}$**: Does $(\mathcal{R}, \mathcal{T})$
   possess associativity? Inverses? A lattice structure?
2. **Origin of $R_0$**: How does a system bootstrap its first
   representation? (Candidates: compression pressure, developmental
   curriculum, action affordances.)
3. **FailureSignature as type**: Can failure modes be formalized as
   proofs of representational inadequacy?
4. **Equivalence of representations**: Is
   $R_i \equiv R_j \iff \mathcal{I}_{R_i} = \mathcal{I}_{R_j}$
   sufficient, or do we need operational equivalence
   ($\forall D: \phi_{R_i}(D) \cong \phi_{R_j}(D)$)?
5. **Metric independence**: Are the five dimensions of $Q(R)$
   independent, or do they covary? (If they covary, the vector
   collapses to fewer effective dimensions.)

---

## 8. Relationship to Existing Work

| Concept | Related Framework | Key Difference |
|---------|-------------------|----------------|
| Representation as structure | MDL (Rissanen, 1978) | We optimize for reasoning cost, not description length |
| Invariant detection | Inductive Logic Programming | Invariants are properties of $R$, not of the world |
| Multiple representations | Society of Mind (Minsky, 1986) | We emphasize dynamics (transitions), not static coexistence |
| Representation selection | Meta-learning | We select representations, not model parameters |
| Failure-driven change | Impasse-Driven Learning (VanLehn, 1988) | Applied at representation level, not skill level |
| Paradigm shifts | Kuhn (1962) | We seek algorithmic mechanisms, not historical descriptions |
| Transition algebra | Abstract rewriting systems | We add quality metrics and failure-driven triggers |

---

## 9. Roadmap

```
Phase 0 ✅ : Foundation — formal definition, pixel + object, quality vector
Phase 1    : Grow representation zoo (region, symmetry, topology, constraint)
Phase 2    : Formalize representation space — metric, neighborhood, manifold
Phase 3    : Representation algebra — formalize transition operators
Phase 4    : Representation selection — applicability estimation, adaptation
Phase 5    : ARC as first benchmark, not final goal
Phase 6    : Paper
```

---

## References

- Boden, M. (1990). *The Creative Mind: Myths and Mechanisms.*
- Chollet, F. (2019). On the Measure of Intelligence. *arXiv:1911.01547.*
- Kuhn, T. (1962). *The Structure of Scientific Revolutions.*
- Minsky, M. (1986). *The Society of Mind.*
- Rissanen, J. (1978). Modeling by shortest data description. *Automatica.*
- VanLehn, K. (1988). Toward a theory of impasse-driven learning. In *Learning Issues for Intelligent Tutoring Systems.*
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience.*

---

*Working draft v0.2. All definitions are tentative. The companion code
at `representation-core/` is the operational ground truth — if this
document and the code disagree, the code wins (and this document must
be revised).*

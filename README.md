# representation-core

**Representation Engineering: A Formal Foundation for Adaptive Reasoning**

> Not an ARC solver. A formal framework for defining, measuring,
> comparing, selecting, and transforming representations.

---

## What is Representation Engineering?

Prompt Engineering is about what you say to the model.
Context Engineering is about what information you give it.
**Representation Engineering** is about how the system *sees* the problem
before reasoning even begins.

Before search, before planning, before learning — a system must CHOOSE
how to encode its observations. That choice determines everything that
follows. A good representation makes the solution obvious. A bad
representation makes it computationally invisible.

This repository provides the formal vocabulary, metrics, and a growing
zoo of concrete representations — all implementing a shared mathematical
contract.

---

## Why ARC?

ARC-AGI is not the goal. It is the **laboratory**.

ARC grids are simple enough to implement many representations, but
structurally rich enough that representation choice dramatically changes
reasoning difficulty. Prove the framework here first. Generalize later.

---

## Core Definition

A representation is a 4-tuple:

$$R = (\mathcal{S}_R, \phi_R, \mathcal{I}_R, \mathcal{T}_R)$$

| Symbol | Meaning |
|--------|---------|
| $\mathcal{S}_R$ | Symbolic space |
| $\phi_R: \mathcal{O} \to \mathcal{S}_R$ | Encoding map |
| $\mathcal{I}_R$ | Invariants made explicit |
| $\mathcal{T}_R$ | Admissible transformations |

Quality is a **vector**, not a scalar:

$$Q(R) = (C, H, I, E, X)$$

Ranking is a **partial order** — one representation may dominate another,
or two may be incomparable.

See [`papers/formal-definition.md`](papers/formal-definition.md) for the
full mathematical framework.

---

## Structure

```
representation-core/
├── core/                  # Abstract contracts + data structures
│   └── representation.py  #   R = (S, φ, I, T) as Python ABC
├── representations/       # Concrete implementations (the "zoo")
│   ├── pixel.py           #   Baseline: identity encoding
│   └── object.py          #   Connected-component object graph
├── metrics/               # Quality vector + comparative metrics
│   └── metrics.py         #   QualityVector, Pareto front, dominance
├── transitions/           # Dynamics between representations
│   └── transition.py      #   Failure detection, transition proposal
├── benchmarks/            # Task loaders (ARC first)
│   └── arc/
│       └── loader.py
├── papers/                # Formal definitions and research notes
│   └── formal-definition.md
└── tests/                 # Contract validation + comparative tests
    ├── test_representation.py
    └── test_object_representation.py
```

---

## Quick Start

```python
import numpy as np
from core.representation import Observation
from representations import PixelRepresentation, ObjectRepresentation
from metrics import compute_quality, find_pareto_front, dominance_matrix

# Create a simple observation
grid = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
obs = Observation(grid=grid)

# Compute quality vectors
pixel = PixelRepresentation()
obj = ObjectRepresentation()

q_pixel = compute_quality(pixel, obs)
q_obj = compute_quality(obj, obs)

print(f"Pixel quality: {q_pixel}")
print(f"Object quality: {q_obj}")
print(f"Object dominates pixel? {q_obj.dominates(q_pixel)}")

# Find Pareto front
front = find_pareto_front([pixel, obj], obs)
print(f"Pareto front: {[r.name for r in front]}")

# Check applicability
print(f"Object applicability: {obj.applicability(obs):.2f}")
print(f"Pixel applicability: {pixel.applicability(obs):.2f}")
```

---

## The Framework Demonstrates

> Different representations induce **measurably different** reasoning
> characteristics under a common evaluation protocol.

Representation choice changes hypothesis space size, reasoning cost,
and invariant density — not by percentages, but by **orders of
magnitude**. This is the core empirical claim, and it is falsifiable.

---

## Hypotheses

| ID | Claim | Status |
|----|-------|--------|
| H−1 | Intelligence = iterative representational refinement | Normative core |
| H0 | $R^* = \arg\min[C_{\text{rep}} + C_{\text{reason}}]$ | Special case of H−1 |
| H1 | Bottleneck of AI is representation, not search | Falsifiable |
| H2 | No canonical representation exists | Falsifiable |
| H3 | Tension correlates with reasoning cost | Falsifiable |
| H4 | Applicability $\alpha(R,T)$ predicts task performance | Falsifiable |

---

## Tests

```bash
cd representation-core
python -m pytest tests/ -v
```

All representations validated against the abstract contract.
All metrics tested for consistency and finiteness.

---

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Foundation — formal definition, pixel + object, quality vector | ✅ Done |
| 1 | Grow representation zoo (region, symmetry, topology, constraint) | ⬜ Next |
| 2 | Formalize representation space — metric, neighborhood, manifold | ⬜ |
| 3 | Representation algebra — formalize transition operators | ⬜ |
| 4 | Representation selection — applicability estimation, adaptation | ⬜ |
| 5 | ARC as first benchmark, not final goal | ⬜ |
| 6 | Paper | ⬜ |

---

## License

TBD

---

*"The question is not how to solve the problem. The question is how to
see the problem so that the solution becomes simple."*

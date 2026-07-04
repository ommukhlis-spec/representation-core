# Representation Benchmark Report
**Generated:** 2026-07-04T23:50:38.190071
**Tasks:** 20
**Representations:** 3

## Summary
Suite pass rate: **40%** (8/20)
Best-representation accuracy: **30%**

## Leaderboard
| Rank | Representation | Coverage | RCS Verdict | RCS Score | Avg α |
|------|---------------|----------|-------------|-----------|-------|
| 1 | **object** | 15/20 | strong_accept | 3.8 | 0.80 |
| 2 | **symmetry** | 4/20 | weak_accept | 1.9 | 0.39 |
| 3 | **pixel** | 1/20 | reject | -0.1 | 1.00 |

## Per-Representation Details
### pixel (v0.1.0)
- **Coverage:** 1/0 canonical tasks
- **Failure modes covered:** ['symmetry_overapplication']
- **Avg complexity:** 176.9
- **Avg applicability:** 1.000
- **RCS:** -0.1 (reject)
- **Tasks solved:** no_symmetry

### object (v0.1.0)
- **Coverage:** 15/5 canonical tasks
- **Failure modes covered:** ['adjacency_ambiguity', 'color_grouping_ambiguity', 'connectivity_ambiguity', 'conservation_law_blindness', 'containment_hierarchy_failure', 'fragmentation_ambiguity', 'gradient_discretization_failure', 'hole_counting_failure', 'hole_detection_failure', 'interior_detection_failure', 'object_merge_ambiguity', 'overlap_ambiguity', 'relational_rule_blindness', 'rotation_ambiguity', 'size_invariance_blindness']
- **Avg complexity:** 89.0
- **Avg applicability:** 0.800
- **RCS:** 3.8 (strong_accept)
- **Tasks solved:** count_discrete_objects, count_by_color, uniform_object_size, many_small_objects, rotational_symmetry_4, single_hole, multiple_holes, connectivity_check, nested_containment, overlapping_regions, continuous_gradient, spatial_adjacency_zones, region_filling, color_ordering_rule, color_conservation
- **Tasks failed:** no_symmetry

### symmetry (v0.1.0)
- **Coverage:** 4/3 canonical tasks
- **Failure modes covered:** ['logical_constraint_blindness', 'reflection_ambiguity', 'sequence_pattern_blindness', 'translation_ambiguity']
- **Avg complexity:** 135.2
- **Avg applicability:** 0.392
- **RCS:** 1.9 (weak_accept)
- **Tasks solved:** horizontal_reflection, translational_symmetry, sequential_pattern, xor_pattern
- **Tasks failed:** rotational_symmetry_4

## Failure Mode → Representation Mapping
| Failure Mode | Canonical Task | Required Representation |
|-------------|---------------|------------------------|
| adjacency_ambiguity | Spatial Adjacency Zones | **region** |
| color_grouping_ambiguity | Count Objects by Color | **object** |
| connectivity_ambiguity | Check If Two Regions Are Connected | **topology** |
| conservation_law_blindness | Color Conservation | **constraint** |
| containment_hierarchy_failure | Nested Containment (Object in Hole in Object) | **topology** |
| fragmentation_ambiguity | Many Small Objects — Fragmentation | **object** |
| gradient_discretization_failure | Continuous Color Gradient | **region** |
| hole_counting_failure | Count Multiple Holes | **topology** |
| hole_detection_failure | Detect a Single Enclosed Hole | **topology** |
| interior_detection_failure | Region Filling (Flood Fill) | **region** |
| logical_constraint_blindness | XOR / Mutual Exclusion Pattern | **constraint** |
| object_merge_ambiguity | Count Discrete Objects | **object** |
| overlap_ambiguity | Overlapping Regions (Venn-like) | **region** |
| reflection_ambiguity | Horizontal Reflection Symmetry | **symmetry** |
| relational_rule_blindness | Color Ordering Rule (Red above Blue) | **constraint** |
| rotation_ambiguity | Rotational Symmetry (Order 4) | **symmetry** |
| sequence_pattern_blindness | Sequential Pattern (Arithmetic) | **constraint** |
| size_invariance_blindness | Detect Uniform Object Size | **object** |
| symmetry_overapplication | Random Pattern — No Symmetry | **object** |
| translation_ambiguity | Translational Symmetry (Wallpaper) | **symmetry** |

## CI Gate Status
- ❌ **pixel**: 1 coverage (min: 3)
- ✅ **object**: 15 coverage (min: 3)
- ✅ **symmetry**: 4 coverage (min: 3)

**Gate: FAILED** ❌ — one or more representations below minimum coverage.
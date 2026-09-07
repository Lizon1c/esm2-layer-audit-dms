# Which ESM-2 layer should you feed a mutation-effect predictor?

Companion code, data, and manuscript for the paper:

> **Which ESM-2 layer should you feed a mutation-effect predictor?
> A 33-layer × 4-protein audit**

## Headline findings

- The best ESM-2 readout layer for DMS prediction is **protein-specific** (RBD L10, CI L32,
  TIM/CASP3 L33): both common defaults — "always last layer" and "always some middle layer" —
  fail on half of the systems.
- Fusing a structure-based second stream (Z_II) improves prediction at essentially every layer
  (131/132 positive) but **never relocates the optimal layer**; the fusion marginal is itself
  layer-structured, largest where the PLM layer is weak.
- The literature picture "mid-layers ≈ structure" holds on only half the systems (RBD/CI support
  it; TIM/CASP3 invert it, with the last layer most structure-aligned).
- Linear redundancy ≠ uselessness: TIM's fusion marginal recovers at L33 exactly where CKA
  redundancy is maximal. Mechanistic hint: last-layer representations collapse to ~1/4 the
  effective rank of the best middle layer (PR 44.5 vs 205.7).

## Layout

- `paper/` — manuscript (`main.tex`, compiled `main.pdf`, figures) and `TRACEABILITY.md`.
- `scripts/` — the layer-sweep and diagnosis scripts (E74/E75/E75b/E77), archived as-run.
- `results/` — per-protein JSON aggregates for all 33 layers × 3 splits × 2 inits × 2 architectures.
- `figures/` — the four-protein profile figure in standalone form.

## Reproduction notes

Absolute paths inside scripts refer to the original environment. All comparisons are split-paired
with a pre-registered ±0.02 Spearman-ρ noise floor (see Methods).

## License

MIT (see LICENSE).

# TRACEABILITY — M3: "Which ESM-2 layer should you feed a mutation-effect predictor?"

Every quantitative claim in `main.tex` is mapped below to its source file. All JSON files
were re-read and re-derived on 2026-09-07; where the AGENTS.md ledger and the raw JSON
agree (all headline numbers), the JSON is cited. Experiments: E74 (`e74_l12_diagnosis`),
E75 (`e75_layer_sweep`, SP single-stream), E75b (`e75b_layer_sweep_xa`, XAttn fusion),
E77 (`e77_layer_redundancy`, CKA). Ledger: `$P/AGENTS.md` lines 248 (E74), 250 (E75/E75b/E77).

## Sources

- `SP[p]`  = `/mnt/k/output_heads/<p>/e75_layer_sweep/results.json` (p ∈ rbd, casp3, ci, tim)
- `XA[p]`  = `/mnt/k/output_heads/<p>/e75b_layer_sweep_xa/results.json`
- `CKA`    = `/mnt/k/output_heads/e77_layer_redundancy/results.json`
- `E74`    = `/mnt/k/output_heads/rbd/e74_l12_diagnosis/{diag.json, sp_results.json}`
- `Noise`  = `AGENTS.md:112` (E14: identical cell across GPUs/processes differs by 0.021
  → ±0.02 floor)

## Number map (text → source)

| Claim in paper | Value | Source (verified) |
|---|---|---|
| RBD SP best L10 = 0.5141; L33 = 0.4536; Δ +0.0605 | ✓ | SP[rbd] `L10.mean`, `L33.mean` |
| CI SP best L32 = 0.5416; L33 = 0.5037; Δ +0.0380 | ✓ | SP[ci] |
| TIM SP best L33 = 0.5992 | ✓ | SP[tim] |
| CASP3 SP best L33 = 0.5826 | ✓ | SP[casp3] |
| XA peaks L10/L31/L32/L33 = 0.5706/0.5912/0.5729/0.6568; XA L33 = 0.5038/0.5897/0.5283/0.6568 | ✓ | XA[·] |
| CASP3 XA L31 vs L33: +0.0015 < floor → no relocation | ✓ | XA[casp3] 0.5912−0.5897 |
| Marginal at SP-optimal layer: RBD +0.0565, CASP3 +0.0071, TIM +0.0576, CI +0.0313 | ✓ | XA−SP at L10/L33/L33/L32 |
| Marginal 131/132 positive, 119/132 above floor | ✓ | recomputed 2026-09-07 from SP/XA JSONs |
| TIM marginal U: L1 +0.179, L15 −0.004, L33 +0.058 (max L2 +0.186) | ✓ | recomputed; L1 +0.1791, L15 −0.0040, L33 +0.0577, L2 +0.1856 |
| RBD marginal L1 +0.133, L31 +0.079 | ✓ | +0.1328 / +0.0785 |
| CASP3 marginal L33 +0.007 below floor | ✓ | +0.0071 |
| Per-protein marginal means +0.047…+0.057 | ✓ | rbd +0.0489, casp3 +0.0471, ci +0.0535, tim +0.0574 |
| Pearson(SP,XA) across layers: 0.98/0.95/0.92/0.78 (CASP3/CI/TIM/RBD) | ✓ | recomputed 2026-09-07 (0.975/0.946/0.916/0.779) |
| CKA: RBD peak L12 0.0446; CI peak L7 0.3653; TIM max L33 0.1150; CASP3 U min L9–15 (min L10 0.0103), max L33 0.0438 | ✓ | CKA json |
| Spearman(SP ρ, CKA) across 33 layers: TIM +0.78, CASP3 +0.22, RBD +0.14, CI −0.22 | ✓ | recomputed 2026-09-07 (+0.775/+0.220/+0.135/−0.224) |
| E74: PR L12 205.7 vs L33 44.5 (4.6×); eff90 702 vs 325 | ✓ | E74 diag.json (205.66/44.50; 702/325) |
| E74 SP check: L12 0.4955 ≥ L33 0.4821 | ✓ | E74 sp_results.json |
| Profile-shape values (RBD L1 0.366, L5 0.493, L12 0.496, L24 0.458, L32 0.466; CI L5 0.285, L15 0.459; TIM L1 0.333, L10 0.488; CASP3 L1 0.358, L10 0.421) | ✓ | SP[·] per-layer means |
| Per-layer s.d. medians 0.026–0.061; CI 0.061 | ✓ | recomputed: rbd 0.039, casp3 0.026, ci 0.061, tim 0.047 |
| Dataset N: RBD 3998, CASP3 1469, TIM 1519, CI 351 | ✓ | e75 log.txt "Loading …" lines |
| Design: 33 layers × 3 splits (42–44) × 2 inits (7,107) × 250 ep; BS 128 (CI 32); AdamW 1e-4/0.05 cosine; position-held-out 30% | ✓ | `e75_layer_sweep.py`, `e75b_layer_sweep_xa.py` source |
| CKA protocol: ≤300 mutants × 20 positions, linear CKA, ridge λ=100 80/20 | ✓ | `e77_layer_redundancy.py` source |

## Ledger vs raw-JSON discrepancies

None for headline numbers. Minor notes:

1. **TIM marginal maximum layer**: ledger text quotes "TIM L1 +0.179"; the raw JSON gives the
   maximum at L2 (+0.1856), L1 is +0.1791. Paper cites L1 +0.179 (as ledger) and reports the
   U-shape endpoints; Fig. 2 shows the full curve including L2.
2. **E74 vs E77 CKA values differ in absolute scale** (E74 RBD: L12 0.077 / L33 0.026; E77
   RBD: L12 0.0446 / L33 0.0363) due to different subsampling protocols; directions agree
   (L12 > L33). Paper uses E77 for all cross-protein CKA claims; E74 only for effective rank
   and the matched-protocol L12-vs-L33 SP check.
3. **CASP3 dataset attribution**: ledger shorthand "Roychowdhury 2020" resolves to
   Roychowdhury & Romero, *Cell Death Discovery* 8:7 (2022), doi:10.1038/s41420-021-00799-0
   (per AGENTS.md:600 resolution note). Cited as 2022.
4. **TRPC_THEMA organism**: ProteinGym reference file says *Thermus thermophilus* (not
   *T. maritima*); paper uses *T. thermophilus*.
5. **CI Z_II artifact** (AGENTS.md:300): legacy slice indexing, numerically close to
   corrected extraction (0.374 vs 0.364 baseline). Flagged in Methods and in the CKA results
   paragraph.

## Bibliography verification (all via web, 2026-09-07)

- Rives et al. 2021, PNAS 118(15):e2016239118 — verified.
- Lin et al. 2023, Science 379(6637):1123–1130, doi:10.1126/science.ade2574 — verified.
- Notin et al. 2023, NeurIPS 36:64331–64379 — verified.
- Tenney et al. 2019, ACL:4593–4601, doi:10.18653/v1/P19-1452 — verified.
- Ethayarajh 2019, EMNLP-IJCNLP:55–65, doi:10.18653/v1/D19-1006 — verified.
- Beshkov & Malthe-Sørenssen 2025, arXiv:2509.24895 — verified (title/authors match ledger
  citation "Beshkov 2025").
- Kornblith et al. 2019, ICML PMLR 97:3519–3529 — verified.
- Corley et al. 2025 (AtomWorks/RF3), bioRxiv 2025.08.14.670328 — verified via PubMed 40832246.
- Starr et al. 2020, Cell 182(5):1295–1310.e20, doi:10.1016/j.cell.2020.08.012 — verified via Crossref.
- Roychowdhury & Romero 2022, Cell Death Discovery 8:7 — verified via AGENTS.md resolution.
- Chan et al. 2017, Nat Commun 8:14614, doi:10.1038/ncomms14614 — verified.
- Li et al. 2019, Nat Commun 10:3886, doi:10.1038/s41467-019-11735-3 — verified via Crossref
  (authors Li, Lalić, Baeza-Centurion, Dhar, Lehner).

## Figures

- `figures/fig1_layer_profiles.png` = copy of `$P/figures/e75_e77_four_proteins.png`
  (project main figure; visually confirmed 2026-09-07).
- `figures/fig2_fusion_marginal.{pdf,png}` = generated from SP/XA JSONs (script inline in
  build session; recomputed means, ±0.02 band added). Read back and verified.
- `figures/fig3_effective_rank.{pdf,png}` = generated from E74 diag.json. Read back and
  verified.

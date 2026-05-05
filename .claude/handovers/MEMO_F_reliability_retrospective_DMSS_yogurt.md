# F-stat reliability retrospective: DMSS yogurt application

**Date:** 2026-05-01
**Branch:** `feat/f-reliability-redesign`
**Data:** DMSS (Duarte, Magnolfi, Sølvsten, Sullivan 2024) yogurt panel
(1273 products, 30 markets, 37 firms)
**Script:** `/tmp/retrospective_dmss_yogurt.py` (reproducible)

## Why this retrospective

Lorenzo's 2026-04-29 audit ran the original F-reliability diagnostic on
the CarRV dataset and found 15/15 pairwise cells flagged as
"near-degenerate". The diagnostic's verdict label was alarming, but the
empirical F values were reproducible across pyRVtest versions to ~0.008
absolute (on F values around 6-17). Lorenzo's note proposed that the
diagnostic was over-firing — the geometry was delicate but the test
conclusion was stable.

The redesigned diagnostic (PR `feat/f-reliability-redesign`) split the
old single-threshold gate into two thresholds with different jobs:

- λ < 0.05 — *informational* low-separation geometry footnote
- λ < 10⁻¹⁰ — mpmath transparently replaces F̂ and ρ̂² (numerical
  safety net)

This memo reports cell counts on the DMSS yogurt application, the
canonical published example for the package.

## Setup

15 pairwise cells from a 6-model setup chosen to mirror CarRV's 6
markup columns:

- Bertrand (closed-form from logit α)
- Monopoly (closed-form joint maximization)
- PerfectCompetition (zero markup)
- RuleOfThumb(φ=1.5)
- RuleOfThumb(φ=2.0)
- RuleOfThumb(φ=3.0)

Logit α estimated by pyblp 2SLS with BLP instruments at α̂ = −0.99.
Instruments: 2 BLP-style instruments (size, light, flavor_Plain,
log_nflavors of rivals).

## Findings

### Verdict counts (15 total cells)

| Verdict | Count |
|---|---|
| `robust` | 15 |
| `weak` | 0 |
| `trivially-degenerate` | 0 |

### Threshold-firing counts

| Threshold | Cells flagged |
|---|---|
| λ < 0.05 (descriptive footnote) | 1 |
| λ < 10⁻¹⁰ (mpmath safety net) | 0 |

### Lambda distribution

- min: 5.2 × 10⁻⁴
- median: 0.21
- max: 0.67

### Float64 vs mpmath agreement on F̂

All 15 cells: |F_double − F_mpmath| / F_double = **0** (agreement to
20+ decimal digits). Float64 was sufficient for every cell.

## Comparison to PRE-redesign verdict

Under the original λ < 0.05 gate that produced "near-degenerate":

- **1 cell** would have been flagged as "near-degenerate" with the
  original alarming language ("F's denominator has lost most of its
  scale to cancellation; F's value is numerically unreliable here").

Under the redesigned diagnostic that same cell:

- Receives a `robust` verdict (DMSS-standard pass).
- Gets a single descriptive `low-separation geometry` footnote noting
  ρ̂² is in the cancellation regime — purely informational, no claim
  about F̂ being unreliable.

## What this confirms

1. **Float64 is more than enough** for typical conduct-testing
   applications on the order of DMSS yogurt. F̂ agrees with mpmath to
   ~20 digits across all 15 cells. The mpmath safety net would correctly
   *not* fire on this application.
2. **The descriptive footnote is calibrated about right.** Only 1 of 15
   cells (≈7%) gets the footnote, which is below an attention-grabbing
   threshold but above zero — it surfaces when the geometry is genuinely
   delicate without being chatty.
3. **The verdict label change is consequential.** Pre-redesign that 1
   cell would have generated an alarming "near-degenerate" verdict that
   could mislead a reader of paper output. Post-redesign the same cell
   reads as robust with a calm footnote.

## Edge case worth noting (not a bug)

For pairs of RuleOfThumb models with different φ, the markups are
linear in price (markup_φ = (φ−1)/φ · p). Two such models produce
*linearly collinear* moment vectors → ρ̂² = 1.0 exactly. The CV-table
lookup clamps to ρ² = 0.99 for these cells. Under DMSS's K=2-9 size
auto-claim, the verdict is `robust` (size guarantee is automatic at
moderate K regardless of F̂). Six of the 15 cells fall in this regime
(all the ROT-vs-ROT and ROT-vs-PC pairs).

This is correct DMSS behavior — the K=2-9 size guarantee is the
relevant statement — but worth noting that the diagnostic doesn't
specifically flag these as "structurally collinear models" beyond the
λ-based footnote (which doesn't fire at moderate λ even with ρ̂² = 1).
The standard DMSS test outcome (F vs CV) handles them correctly; the
F-reliability layer doesn't add commentary.

## Recommendation

The DMSS yogurt evidence supports keeping the redesigned thresholds
(λ < 0.05 descriptive, λ < 10⁻¹⁰ for mpmath) without modification. The
diagnostic is appropriately quiet on a published, well-behaved
application: 14/15 cells get no footnote, the 1 footnote that fires is
informational rather than alarming, and the mpmath safety net never
needs to engage.

The same retrospective on CarRV would give a different cell count (15/15
near-degenerate under the old diagnostic, per Lorenzo's audit). The
expected new-diagnostic outcome on CarRV: most cells get the `robust`
verdict with a low-separation footnote on the cells that previously
flagged. To execute that retrospective, the script in this memo can be
re-pointed at CarRV's product_data + markup columns.

## Reproducibility

```bash
# From the pyRVtest repo on feat/f-reliability-redesign:
python3 /tmp/retrospective_dmss_yogurt.py
```

Inputs (paths used in the script):

- `~/Dropbox/Economics/claude/degeneracy-conduct-testing/data/dmss_replication/product_data.csv`

Outputs:

- Stdout: full F-stat table, F_reliability_summary DataFrame,
  threshold counts.
- No file artifacts; the script is read-only.

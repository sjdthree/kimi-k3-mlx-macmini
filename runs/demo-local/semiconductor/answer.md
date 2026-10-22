# Root-Cause Analysis: Lot OR-26-091 Bridge Defect Excursion

## 1. Conclusion: Most Likely Mechanism

**A mis-seated ("proud") edge gas-ring seal, installed during the Sep 14 18:10 PM on ETCH-7 chamber C2, perturbed gas injection/conductance at the wafer edge, causing locally elevated etch rate at radius >70 mm.** The resulting edge over-etch produced the observed low-k thinning (~–2.5 nm edge-to-center) and positive CD bias (~+4 nm) at the edge, which narrowed inter-feature spaces until bridge defects formed.

Mechanism chain: PM seal replacement → proud seat → altered edge gas flow/conductance (or a small edge leak path) → radially nonuniform etch (edge-fast) → edge CD bias and low-k thinning → space CD collapse at edge → bridges. The defect signature is therefore **radial, chamber-specific, and PM-time-locked** — exactly what the data show.

## 2. Supporting Evidence

| Evidence | Implication |
|---|---|
| **Chamber split within the same lot**: W01/W02 (C2) show edge thickness 59.7/59.4 nm vs center 62.1/62.0 nm, edge CD bias +3.9/+4.2 nm, edge defects 1.55/1.71/cm². W03 (C1, same recipe) is flat: 62.2→61.7 nm, +0.3→+0.8 nm, 0.08→0.14/cm² | Isolates the cause to **C2 hardware**, not the lot, recipe, materials, or environment |
| **Radial co-localization of all three signals**: defects (0.10→0.46→1.55), thickness (62.1→61.4→59.7), CD bias (+0.4→+1.2→+3.9) all rise monotonically toward the edge; mid-radius is partially affected | A single edge-centered etch perturbation explains all three; signals are ~10× the stated repeatability (±0.25 nm ellipsometry, ±0.4 nm CD-SEM), so they are real |
| **Time lock**: edge seal replaced Sep 14 18:10; first affected lot starts Sep 15 07:20. No other C2 change is recorded | PM is the only candidate change on the affected chamber |
| **Reversibility (Table 5)**: C2 after seal reseat matches C1 (–0.5 vs –0.4 nm; +0.9 vs +0.8 nm; 0.18 vs 0.14/cm²) | Strongest single causal datum: reseating the seal returns C2 to C1 behavior |
| **Negative controls**: MFC calibration normal, no alarm history (Sep 15 17:30); litho overlay, CMP maps, incoming low-k all in control | A mechanical mis-seat is invisible to MFC calibration and endpoint alarms — consistent, not contradictory |
| **Commonality exclusions**: new resist lot used on **both** affected (C2) and unaffected (C1) splits; humidity window shared with unaffected lots | Resist lot and ambient humidity cannot explain a chamber-specific signature |
| **Density split**: high-density 0.73 vs low-density 0.69/cm² | Explicitly falsifies the yield team's pattern-density suspicion; the "outer die" clustering was a **radial** effect misread as a layout effect |

The escape route is also explained: the Sep 14 19:05 post-PM qualification checked **only center-point endpoint and blanket etch rate** — both blind to an edge-localized nonuniformity.

## 3. Contradicting / Reconciling Evidence

- **Post-PM qual passed** — reconciled by center-only sampling; an edge seal fault would not be detected. This is a detection gap, not evidence against the hypothesis.
- **The "proud seal" note is informal and unquantified** — the PM checklist has no seating-height measurement. The case rests on the reseat reversibility and chamber split, not this anecdote; the seat height must be measured (Experiment E1 below).
- **Chilled-water pressure fluctuation (Sep 15 09:12)** — tool telemetry shows no chuck-temperature excursion; a transient facility event would not produce a *stable* radial signature across all six C2 wafers (persisting as a steady state in Table 5), and C1 wafers processed later the same day on a shared facility were clean. Ruled down, not out (close it with E4).
- **Resist lot** — used on both splits; cannot produce chamber specificity.

## 4. Ranked Alternatives

1. **Proud/mis-seated edge gas-ring seal → edge over-etch (primary).** Fits every observation including reversibility.
2. **Wrong or damaged seal part / edge-ring hardware defect from the same PM** (wrong P/N, nicked seal → edge leak). Same macro-signature; distinguished from (1) only by part inspection and leak test.
3. **C2 gas-distribution fault independent of the seal** (partially clogged edge injector, MFC nonlinearity at recipe flow). Argued against by normal MFC calibration and the reseat fix — but a clog coincidentally cleared during reseat is possible; check that reseat was a single-variable intervention.
4. **Edge thermal nonuniformity** (ESC edge zone; chilled-water event). No telemetry excursion; transient vs. stable signature; shared facility with C1 unaffected. Low.
5. **Materials / litho / CMP / resist / humidity.** Effectively excluded by commonality across splits or in-control data.
6. **Pattern density (initial suspicion).** Excluded by the density split (0.73 vs 0.69/cm²).

## 5. Minimum Discriminating Experiments

The smallest decisive set — three are wafer-free:

- **E0 (free, existing data): Azimuthal re-mapping of W01/W02.** Defect, CD, and thickness coordinates already exist in the inspection/metrology databases. A mis-seated seal should produce an **azimuthally asymmetric** signature weighted toward the seat location; an azimuthally uniform ring would point to a symmetric flow or thermal cause. Discriminates (1)/(2) vs. (3)/(4) at zero cost.
- **E1 (free, no wafers): Dimensional metrology of the installed seal seat** on C2 vs. C1 and spec (seating height at multiple azimuths), plus inspection of the removed seal for damage or wrong part number. Directly tests (1) vs. (2).
- **E2 (free): Helium leak / rate-of-rise test on C2 vs. C1**, or RGA during a dummy run for N₂/O₂. Distinguishes the **leak sub-mechanism** from **flow-redistribution**, and further separates (1)/(2) from (3). (An air ingress path is chemically plausible: local O₂ would preferentially ash organic low-k, matching the observed edge thinning plus CD growth.)
- **E3 (1–2 wafers): Radial blanket etch-rate maps** (≥9-point, center/mid/edge) on C2 and C1. Reproduces the gradient directly and becomes the corrected post-PM qual. The post-reseat leg is effectively already demonstrated by Table 5.
- **E4 (free): Pull C2 chuck/edge-zone temperature telemetry** for Sep 15 07:20–13:40 to formally close alternative (4).

**Logic:** E0–E2 (all wafer-free) discriminate among hypotheses 1–4. Only if those are inconclusive is E3 required. Note also that the proximal defect driver appears to be the edge **CD bias** (spaces shrinking), with thickness loss a co-symptom of the same over-etch — defects track CD bias monotonically across radial bands (0.10@+0.4 → 0.46@+1.2 → 1.55@+3.9). This distinction is not needed for containment, since the fix removes both.

## 6. Corrective Gap (brief)

Two systemic holes enabled the excursion: (a) the post-PM qual samples only the wafer center — add radial thickness/CD uniformity for any PM touching gas distribution or edge hardware; (b) the PM checklist lacks a quantitative seal seating-height measurement — add it. C2 should remain restricted from patterned product pending E1–E3; W01–W06 require disposition for edge-die scrap.
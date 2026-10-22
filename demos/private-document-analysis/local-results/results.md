# Kimi-K3 Private Document Evaluation — local run

Model: full Kimi-K3 checkpoint, run by this repository on a 64 GiB M4 Pro Mac mini
Inputs: [`local/inputs/`](../local/inputs/) (PDF text plus the question; one user message, no system prompt)
Maximum completion tokens: 20,000

---

# 1. semiconductor_root_cause

## Source Document

`01_semiconductor_confidential_excursion.pdf`

## Question

Analyze this confidential fab excursion package.

Determine the most likely physical mechanism causing the defect excursion,
identify evidence supporting and contradicting the hypothesis, rank plausible
alternatives, and propose the minimum experiments needed to distinguish them.

Do not assume that any engineer's stated suspicion is correct.

Cite specific evidence from the supplied document for your conclusions.

## Kimi-K3 Answer

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

## Run Metadata

```json
{
  "status": "done",
  "finish": "end_of_message",
  "prompt_tokens": 1278,
  "completion_tokens": 7623,
  "reasoning_tokens": 5763,
  "answer_tokens": 1847,
  "format_tokens": 13,
  "decode_seconds_per_token_median": 34.25,
  "decode_seconds_per_token_max": 73.4,
  "generation_hours": 72.6,
  "peak_mlx_gib": 24.27,
  "peak_mlx_gib_decode": 12.63,
  "started": "2026-10-04T02:40:09",
  "finished": "2026-10-10T02:04:32",
  "interruptions": [
    {
      "stopped": "2026-10-05T11:13:02",
      "restarted": "2026-10-08T09:27:19",
      "hours_down": 70.2
    }
  ],
  "settings": {
    "temperature": 1.0,
    "top_p": 0.95,
    "seed": 0,
    "thinking_effort": "max",
    "max_tokens": 20000,
    "resident_gb": 8.0,
    "mode": "runtime"
  },
  "messages": "demos/private-document-analysis/local/inputs/semiconductor.json",
  "messages_sha256": "a63d47ed21c4fd06f424d29ff88e5c71f9be6d43b846d0ad4780bc5bd3493784",
  "model_config_sha256": "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
  "commits": [
    "42df3b1"
  ]
}
```

---

# 2. medical_ai_adverse_event

## Source Document

`02_medical_ai_confidential_incident.pdf`

## Question

Reconstruct the adverse event from this confidential package.

Separate model failure, software failure, clinical-policy failure,
deployment/change-control failure, and human error.

Identify the evidence for each conclusion, unresolved facts that could change
the analysis, and the highest-priority corrective controls.

Do not infer legal liability merely from contractual responsibility.

Cite specific evidence from the supplied document for your conclusions.

## Kimi-K3 Answer

# Adverse-Event Reconstruction and Causal Analysis

## 1. Event Reconstruction

**Causal timeline (all evidence from §4 unless noted):**

1. **08:12** — Patient begins the *revised* fluid-resuscitation protocol (2-hour reassessment per the March 2026 revision, §2).
2. **10:31** — Model score reaches **0.47**, satisfying the ≥0.35 high-risk condition (§1). The model detected the deterioration correctly and on schedule.
3. **10:31** — The policy service returns `suppression_interval=4h`, sourced from **NMC-SEP-17 rev 2024-11** — the *pre-revision* policy object.
4. **10:31** — Software logs `ALERT_SUPPRESSED_POLICY`. No clinician-facing alert is shown. Because 10:31 falls within 4 hours of 08:12 (window closes 12:12), suppression is technically "correct" execution of the stale configuration.
5. **11:04** — Lactate and blood pressure worsen.
6. **11:58** — Nurse escalates on clinical signs alone (87 minutes after the suppressed alert).
7. **12:17** — Rapid-response team arrives (106 minutes after the suppressed alert).

**The decisive counterfactual:** Had the suppression interval reflected the revised protocol (2 hours), the suppression window would have closed at **10:12**. The 10:31 alert fired 19 minutes *after* that — it would have been displayed. The stale 4-hour interval is therefore a **but-for cause of the non-alert**. (Whether an earlier alert would have changed the clinical outcome is *not* established by this package — see §4 below.)

---

## 2. Findings by Category

| Category | Verdict |
|---|---|
| Model failure | **Not supported** |
| Software implementation failure | **Not supported** (design-safeguard gap is a separate question) |
| Clinical-policy failure | **Supported — latent cause** |
| Deployment / change-control failure | **Supported — proximal process failure** |
| Human error (active) | **Not supported at bedside; organizational knowledge gaps only** |

### A. Model — no failure
- §5: releases 4.7, 4.8, **and a frozen pre-adaptation model** all cross threshold between 10:27–10:33. The monthly recalibration neither caused, masked, nor delayed the alert.
- §3: 4.8 contained "no feature or threshold changes"; "New self-learning feature weights: **Not implemented**." No unauthorized adaptation or drift occurred.
- §4: the high-risk condition *was* satisfied at 10:31 — the model output was generated; suppression occurred downstream in a separate policy layer.
- **Caveat:** replay demonstrates cross-version consistency, not clinical *adequacy* (whether 10:31 was acceptably timely relative to physiology is unaddressed by this package).

### B. Software — no implementation defect
- §5: "Independent code review finds the suppression function implemented **exactly as specified**."
- §4: the policy service was *available* and returned a value, so the §3 fallback ("uses last validated configuration") was never invoked and is not implicated.
- **Open design question (not a defect):** §7 obligates the vendor to provide "reasonable safeguards against configuration inconsistency **where technically feasible**." The package shows no staleness detection, no config-version checking, and no fail-toward-alerting default. Whether this is a vendor-side safeguard gap depends on a feasibility/reasonableness assessment the package does not contain. Absence of a safeguard is not the same as a malfunction.

### C. Clinical policy — stale policy object (latent failure)
- §2: the sepsis committee shortened reassessment to 2 hours in March 2026, "but **NMC-SEP-17 was not updated**." The 4-hour deferral interval driving automated suppression was inconsistent with the live bedside protocol.
- §6 (Apr 03, hospital analyst): "I wasn't aware the AI suppression interval was sourced from the older policy object" — the data lineage from policy document to AI configuration was not understood by the hospital's own staff.
- **Compounding effect:** the stale policy operated at *two* levels — software suppression **and** the nursing instruction to defer escalation on alerts within 4 hours of fluid initiation. Even if the alert had displayed, SEP-17 would have instructed deferral absent instability criteria. Policy reconciliation is thus doubly causal.

### D. Deployment / change control — proximal process failure
- §5: the March revision "was approved in committee minutes but **never entered into the policy-service change-control queue**." Governance approved the clinical change; the integration pipeline that would propagate it to the AI configuration was never triggered. This is the cleanest single break in the chain.
- §6 (Mar 22, vendor clinical integration lead): noticed "no corresponding SEP-17 ticket" and "**assumed** their governance team owns that" — a discrepancy was actually observed but no escalation mechanism existed or was used.
- §7 assigns the hospital responsibility for "timely notification of policy changes affecting integration behavior"; no notification occurred. This informs causal allocation only — it is **not** a legal conclusion.
- §1 permits suppression "only through validated unit-level configuration with clinical governance approval." The package does not establish whether the 4-hour interval was so validated at inception, or whether any re-validation trigger existed when upstream policy changed.

### E. Human error — none identified at the bedside
- The nurse escalated at 11:58 on clinical signs (§4) — consistent with the intended-use envelope (§1: "not a stand-alone diagnostic"). Because the alert was suppressed at the software layer, SEP-17's deferral instruction never actually governed a nursing decision; there was no visible alert to defer.
- The analyst's unawareness (§6, Apr 03) and the vendor lead's assumption (§6, Mar 22) are **latent organizational conditions** (training, ownership ambiguity, absent escalation duty), not culpable active errors by identifiable individuals.

---

## 3. Causal Chain Synthesis

```
March 2026 protocol revision approved
   └─► Never entered policy-service change-control queue (§5)          ← proximal break
        └─► NMC-SEP-17 remains at 4h; policy service serves stale value (§2, §4)
             └─► 10:31 alert suppressed (but-for cause of non-alert)
                  └─► No clinician-facing signal until 11:58 clinical escalation
Latent/contributing conditions:
   • No staleness/inconsistency safeguard (§7 feasibility question)
   • Observed discrepancy (Mar 22) with no escalation duty or channel
   • Staff unaware of config lineage (Apr 03)
   • No re-validation trigger for suppression configs after policy changes
Exonerated: model behavior, recalibration mechanism, code implementation, fallback logic, bedside nursing
```

---

## 4. Unresolved Facts That Could Change the Analysis

1. **Clinical causation/preventability:** the patient's outcome and whether a 10:31 alert would have altered it. Alert delay ≠ established harm causation; this requires separate clinical review.
2. **Incident date relative to the Apr 03 analyst email.** If the hospital knew of the config-lineage gap *before* the event and did not act, the change-control finding sharpens considerably.
3. **Instability criteria:** were SEP-17's "specified instability criteria" met at 11:04 (or earlier)? If so, escalation timing and policy-compliance analysis change.
4. **Validation status of the 4-hour suppression configuration** at inception under §1's "validated unit-level configuration with clinical governance approval" requirement.
5. **Whether any defined procedure existed** requiring the sepsis committee to route changes into the policy-service queue — and who owned queue entry.
6. **Safeguard feasibility assessment:** what "reasonable safeguards" (§7) were evaluated by the vendor, and what was technically feasible (staleness alerts, version mismatch detection, fail-visible defaults).
7. **Scope of exposure:** how many other alerts were suppressed under the stale interval between March 2026 and the incident; whether other units/policies have similar drift.
8. **Model timeliness:** whether threshold crossing at ~10:31 was acceptably early relative to the deterioration trajectory (replay establishes consistency, not clinical performance).

---

## 5. Highest-Priority Corrective Controls

**Immediate (P0)**
- Reconcile the suppression interval with the current fluid protocol, or suspend suppression pending governance review.
- **Lookback audit** of all alerts suppressed under the stale interval since March 2026; clinical review of affected patients.
- Audit every unit-level suppression configuration against current clinical policy across the deployment.

**Technical (P1)**
- **Fail toward visibility:** on detected config/policy inconsistency or staleness, default to *alerting*, not suppressing.
- Config versioning with staleness detection (flag when a linked clinical policy's revision date postdates config validation).
- Surface suppressed alerts on a reviewable dashboard (suppression should never be invisible).
- Auto-generate policy-service change tickets from approved clinical-policy revisions.

**Governance (P1)**
- Single joint vendor–hospital change-control pathway with explicit RACI for policy-object updates; dual acknowledgment of changes affecting integration behavior.
- A defined **escalation duty** when either party observes a discrepancy — the Mar 22 observation must trigger a formal query, not an assumption (§6).
- Periodic re-validation/attestation of suppression configs under §1's governance-approval requirement.

**Sustaining (P2)**
- Train hospital analysts and committee staff that the AI suppression interval is *sourced from* the policy object (closing the Apr 03 knowledge gap).
- Document the §7 "reasonable safeguards" determination and review it each release.
- Preserve the architecture: the Feb 18 design decision (policy service authoritative, no hard-coding, §6) is sound; the failure was in the governance and safeguards *around* it, not the design itself.

---

## 6. Note on Responsibility vs. Liability

Per the instructions and §7, the responsibility allocations above (hospital: workflow policies, configuration approval, timely notification; vendor: validated operation, documented update mechanism, feasible safeguards) are used solely to structure the **causal** analysis. The facts show a change-control and policy-maintenance breakdown on the hospital side and a possible, unassessed safeguard gap on the vendor side — but no conclusion about legal liability follows from this package, and none is drawn here.

*Analysis of synthetic material only — no real entity, patient, or data involved.*

## Run Metadata

```json
{
  "status": "done",
  "finish": "end_of_message",
  "prompt_tokens": 1039,
  "completion_tokens": 8225,
  "reasoning_tokens": 5943,
  "answer_tokens": 2269,
  "format_tokens": 13,
  "decode_seconds_per_token_median": 35.01,
  "decode_seconds_per_token_max": 53.59,
  "generation_hours": 79.9,
  "peak_mlx_gib": 22.01,
  "peak_mlx_gib_decode": 12.68,
  "started": "2026-10-10T02:04:32",
  "finished": "2026-10-13T10:24:17",
  "interruptions": [],
  "settings": {
    "temperature": 1.0,
    "top_p": 0.95,
    "seed": 0,
    "thinking_effort": "max",
    "max_tokens": 20000,
    "resident_gb": 8.0,
    "mode": "runtime"
  },
  "messages": "demos/private-document-analysis/local/inputs/medical_ai.json",
  "messages_sha256": "c68263d4900d608ffd6d41e9d8d4f9fe9bb86e9719485fc16c3732b4d671e119",
  "model_config_sha256": "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
  "commits": [
    "42df3b1"
  ]
}
```

---

# 3. rd_tradesecret_invention_strategy

## Source Document

`03_rd_tradesecret_invention_strategy.pdf`

## Question

Act as a board-level R&D / Trade Secret Invention Strategy team.

Using only this confidential package:

1. Identify technical concepts that appear materially distinct from the
   existing patent claims.

2. Separate potentially patentable inventions from know-how that may be more
   valuable as trade secrets.

3. Identify combinations of technical concepts that may be more defensible
   or valuable than the individual features.

4. Assess what appears exposed to reverse engineering, disclosure, supplier
   access, or loss of trade-secret protection.

5. Identify potentially valuable inventions or combinations that management
   may not realize it has created.

6. Recommend the next diligence experiments, documentation, invention
   disclosures, and protection decisions that should be considered by
   management and the board.

Cite specific evidence from the supplied document for every material
conclusion.

Explicitly identify where outside prior-art searching, patent counsel,
trade-secret counsel, or other legal analysis would be required.

Do not assume that something is patentable merely because it is not covered
by the existing patents.

## Kimi-K3 Answer

# HelioForge Systems — R&D / Trade-Secret / Invention Strategy Review

**Basis and limits of this review.** This analysis uses only the supplied package. Statements below about what is "materially distinct" from HF-101/118/133 are comparisons to the *simplified claims as given*, not patentability opinions. Patentability, freedom-to-operate, inventorship, and trade-secret enforceability all require outside counsel and prior-art searching, and are flagged throughout. Nothing here assumes an invention is patentable merely because it is unclaimed.

---

## Executive Summary

- The package documents at least **four unclaimed technical concepts** (Exp. 26-041, 26-052, 26-061; §4), none of which has an invention disclosure on file (§6).
- The strongest patent candidates are the **combined predictive-control system** (urgent — a competitor is already advertising "predictive liquid cooling using workload telemetry," §5) and the **per-unit adaptive-geometry manufacturing method** (no known competitor, §5, but the concept is exposed by teardown, §5).
- The strongest trade-secret assets are the **transformation script/coefficients** and the **calibration table** (§3, §4) — low detectability, already partially access-controlled.
- The single most valuable *unrecognized* invention may be the **estimator ↔ accumulator pre-charge coupling**: the documented fact that the effect disappears beyond a >12% pre-charge change (Exp. 26-052) is evidence of a non-obvious interdependency and a natural replication moat.
- Trade-secret program gaps (no disclosures, no visible NDA/marking/notebook-witnessing regime, supplier aggregation risk) must be remediated before the financing diligence.

---

## 1. Concepts Materially Distinct from the Existing Claims

| # | Concept | Evidence | Why distinct from HF-101/118/133 (§1) |
|---|---------|----------|----------------------------------------|
| A | **Power-derivative feed-forward control** — dP/dt of accelerator workload predicts hotspot 180–260 ms before the temperature sensor responds | Exp. 26-041; §4 ("No" public) | HF-133 claims flow adjustment "as a function of **measured package temperature**." Concept A uses a different input (workload telemetry derivative) and is anticipatory, not reactive. The 180–260 ms lead (Exp. 26-041) is the technical effect. |
| B | **Online compressibility estimator** from pressure-wave response | Exp. 26-052; §4 | No claim covers estimating a fluid property. HF-118 claims pressure *sensors* as hardware in a redundant-pump manifold, not estimation methods. (Whether B can leverage HF-118's claimed sensors is a portfolio-layering question — confirm hardware with engineering; counsel on claim interplay.) |
| C | **Estimator ↔ accumulator pre-charge coupling** (calibration table) | Exp. 26-052 (>12% sensitivity); §4 | No existing claim mentions an accumulator at all (§1). |
| D | **Per-unit adaptive microchannel geometry** — ±6% width variation generated from *each package's measured post-assembly thermal map*, not a fixed geometry | Exp. 26-061; §3 | HF-101 claims a cold plate *structure* ("parallel microchannels"). Concept D is a **manufacturing method** (measure → transform → machine per unit) and a **unit-specific product**. Caveat for counsel: HF-101 may still literally read on the new plates (parallel channels of varying width) — that affects what HF-101 protects, not whether D is new. |

**Standing caveat:** distinctness ≠ patentability. Feed-forward/model-predictive cooling, fluid-property estimation, and variable microchannel geometries are active research areas, and a competitor is already publicly in the feed-forward space (§5). Each of A–D requires an independent prior-art search before any filing decision.

---

## 2. Patent Candidates vs. Trade-Secret Know-How

Disposition logic: patent what is (a) detectable in a competitor's product or behavior, or (b) at high risk of independent invention; keep secret what is undetectable and well-contained.

| Asset | Evidence | Detectability / exposure | Proposed disposition |
|---|---|---|---|
| Combined predictive-control method (A+B+C) | Exp. 26-052 (37% reduction in 99.9th-percentile overshoot) | Partially observable via customer-accessible pressure/telemetry (§5); competitor active (§5) | **Patent (if search supports)** — secrecy has a shelf life; independent-invention risk is documented. Quantified 37% effect supports enablement and non-obviousness. |
| Per-unit geometry-generation **method** (D) | Exp. 26-061; §3 (yield 91.2%→97.8%) | Concept inferable from multi-unit teardown; algorithm not (§5) | **Patent method + product claims (if search supports)** — first-mover space (§5), and teardown exposure makes secrecy time-limited. |
| Transformation **script and coefficients** | §3 | Not exposed; access-controlled server; supplier sees only coordinates (§3) | **Trade secret** — undetectable in competitor products; a patent would disclose without enforcement upside. |
| Calibration table (estimator ↔ pre-charge) | §4 | Not exposed (firmware-locked, §5) | **Trade secret**; optionally claim the *coupling method* broadly while retaining the table. Counsel: enablement/best-mode tension. |
| Optimal parameter values (12% window, 180–260 ms, ±6%) | Exp. 26-041, 26-052, 26-061 | — | **Claim as ranges in patents; keep optima/implementation data secret.** |
| Thermal-map → geometry → yield dataset | §3 | Not exposed | **Trade secret (data asset)** — flag as a future ML/analytics asset. |
| Anything deprioritized after search | — | — | **Defensive publication** to bar competitor patents (counsel to weigh against secret value). |

---

## 3. Combinations More Defensible Than Individual Features

1. **A+B+C as one control system (the strongest combination).** Exp. 26-052 documents that the 37% overshoot reduction depends on the combination and *disappears* when accumulator pre-charge shifts >12% — direct evidence of interdependence/synergy rather than mere aggregation. This supports (i) non-obviousness arguments (counsel), and (ii) a replication moat: a competitor copying feed-forward alone (as advertised, §5) may fail silently without the coupling. Feed-forward *alone* is the weakest, most-crowded claim.
2. **B + HF-118 hardware.** The estimator uses "pressure-wave response" (§4); HF-118 already claims manifold pressure sensors (§1). A layered claim (claimed hardware + new estimation function, i.e., a "virtual sensor") ties the new method into the existing portfolio. Flag: also an obviousness risk vector — counsel/search.
3. **D + per-unit control calibration (hypothesis — no supporting evidence yet).** The post-assembly thermal map (Exp. 26-061) could plausibly tune each unit's controller. Not documented; recommend as a diligence experiment (§6 below), not a conclusion.
4. **D + supplier data-split architecture.** The protection of D is itself a system: algorithm on the access-controlled build server, operators see only job IDs, supplier receives coordinates only (§3). The technical method plus this information architecture is the real trade-secret system — document it as such.

---

## 4. Exposure Assessment

**Reverse engineering**
- *Exposed:* physical channel geometry via destructive teardown (§5). Although the generation algorithm is protected (customers lack pre-machining thermal maps, §5), **comparing two or more torn-down units reveals the *concept* of per-unit ±6% variation** (Exp. 26-061). Once the concept is known, independent reimplementation is unprotected by trade-secret law → argues for patenting D.
- *Partially exposed:* control dynamics. Customers can measure external coolant pressure, temperatures, and performance (§5); correlating workload telemetry with pressure response could reveal the existence and approximate lead time of feed-forward (Exp. 26-041). Firmware itself is hardened (secure boot, signed updates, §5).
- *Contained:* coefficients/script (§3), calibration table (§4).

**Disclosure**
- Sales decks already disclose "predictive cooling" and "factory-calibrated cold plates" (§6). On their face these are non-enabling result statements, but counsel must (i) confirm against exact wording and dates and (ii) assess **foreign absolute-novelty** exposure. Marketing is also signaling direction to competitors — the competitor advertisement (§5) may already reflect that.
- First-to-file risk on feed-forward is live (§5). Dated notebooks 26-041/26-052 are critical evidence for invention dates, derivation defenses, and possible prior-user rights — preserve and have them witnessed now (counsel).

**Supplier (§3)**
- Good controls exist (coordinates only, no maps/algorithm, reverse-engineering and reuse prohibitions, 30-day retention). Residual risks: **statistical aggregation across many jobs** (coordinate streams plus knowledge of the blank plate could reveal the ±6% scheme); no audit rights, deletion certification, derived-data prohibition, or subcontractor flow-downs are mentioned. Contract review by trade-secret counsel.

**Program gaps (§3, §4, §6)**
- Zero invention disclosures filed; engineering treats all four items as "implementation details" (§6) → accidental-disclosure culture risk (conferences, papers, supplier conversations).
- No mention of employee NDAs/invention assignments, notebook witnessing, access logging beyond the build server, or exit procedures — these are the "reasonable measures" trade-secret statutes require (trade-secret counsel).
- Key-person concentration: the feed-forward observation is attributed to a single engineer (L. Chen, Exp. 26-041). Confirm assignment agreements and retention; capture know-how in writing.

---

## 5. Inventions Management May Not Realize It Has

1. **The coupling discovery itself** — co-managing the compressibility estimator and accumulator pre-charge within a tolerance band (Exp. 26-052). Engineers filed this as an "implementation detail" (§6); it is arguably the most defensible element of the control stack.
2. **The thermal-map → laser-offset transformation as a manufacturing *method***, not merely a script (Exp. 26-061; §3) — closed-loop, post-assembly-measurement-driven machining, with documented yield effect (91.2%→97.8%, §3) usable as technical-effect evidence.
3. **The compressibility estimator as a "virtual sensor"** deriving a fluid property from pressure-wave response (§4) — a claim framing distinct from "control method."
4. **The predictive lead window** (180–260 ms, Exp. 26-041) as a claimed operating range.
5. **The per-unit data asset** (maps, geometries, outcomes, §3) — cumulative know-how competitors cannot shortcut.
6. **(Hypothesis)** Per-unit controller calibration from the machining map — untested; see §6.

---

## 6. Recommended Next Steps

**Immediate (0–30 days)**
1. File invention disclosures for A, B, C, D, and the transformation/calibration assets (closing the §6 gap); interview Exp. 26-041/052/061 authors; inventorship determination by counsel.
2. Commission prior-art searches in five domains: workload/derivative feed-forward and MPC cooling; fluid compressibility estimation; accumulator pre-charge/control coupling; variable/adaptive microchannel geometries; thermal-map-driven per-unit machining — **including the competitor's filings and publications** (§5).
3. Freeze new external technical disclosures; counsel review of the "predictive cooling"/"factory-calibrated cold plates" decks for enablement, dates, and foreign-novelty impact (§6).
4. Evidence preservation: witness and archive notebooks 26-041/052/061; version-control snapshots of the script, coefficients, and calibration table (§3, §4).
5. Trade-secret counsel: reasonable-measures audit (NDAs, assignments, marking, access logs, exit process); supplier-contract gap review (aggregation, derived data, deletion certification, audit, flow-downs) (§3).
6. Prepare provisional applications on the two patent candidates *before* any further marketing or financing data-room circulation, given competitor activity (§5).

**30–90 days — diligence experiments**
7. **Red-team leakage studies:** (i) infer the controller from customer-accessible signals only (§5); (ii) multi-unit teardown inference of the geometry scheme (§5); (iii) give a cleared third party only the supplier-visible data (§3) and measure what they can reconstruct. Results drive the final patent-vs-secret split.
8. **Robustness/statistics:** replicate Exp. 26-052 across units, coolant batches, and operating ranges; map the 12% pre-charge boundary precisely; put confidence intervals on the 37% figure (needed for patent enablement *and* marketing substantiation — advertising counsel).
9. Test the per-unit control-calibration hypothesis (reusing the Exp. 26-061 map).
10. Final disposition per asset: file / keep secret / defensively publish; foreign-filing decisions per counsel.

**Board / financing readiness**
11. Assemble an IP audit package: portfolio map (issued/pending/secrets), risk register, chain-of-title verification for all contributors (incl. L. Chen, Exp. 26-041).
12. Standing processes: quarterly disclosure triage, publication review board, competitor monitoring (§5), key-person retention plan.

---

## Where Outside Expertise Is Required (Consolidated)

- **Patent counsel / prior-art searcher:** patentability of A–D and all combinations; HF-101 claim-scope construction vs. varied-width plates; detectability-oriented claim drafting; enablement/best-mode strategy where calibration data stays secret; provisional/PCT timing; prior-user-rights analysis.
- **Trade-secret counsel:** reasonable-measures program, DTSA/UTSA posture, supplier contract terms, employee agreements, identification of secrets with particularity for litigation readiness.
- **Advertising/regulatory counsel:** substantiation of the 37% (Exp. 26-052) and yield (§3) claims used commercially.
- **Counsel investigation of facts not in the package:** dates/content of all external disclosures (§6), existing employment/assignment terms, supplier identity and jurisdiction.

**Bottom line for the board:** the package shows real, quantified, unclaimed inventions — but the two most valuable ones are on a secrecy clock (teardown exposure, §5; competitor momentum, §5). The fastest value-preserving moves are cheap: disclosures, notebooks witnessed, provisionals evaluated, and a disclosure freeze — all before the financing data room opens.

## Run Metadata

```json
{
  "status": "done",
  "finish": "end_of_message",
  "prompt_tokens": 1239,
  "completion_tokens": 11651,
  "reasoning_tokens": 8425,
  "answer_tokens": 3213,
  "format_tokens": 13,
  "decode_seconds_per_token_median": 34.88,
  "decode_seconds_per_token_max": 70.92,
  "generation_hours": 112.5,
  "peak_mlx_gib": 23.9,
  "peak_mlx_gib_decode": 13.21,
  "started": "2026-09-29T09:40:49",
  "finished": "2026-10-04T02:40:08",
  "interruptions": [],
  "settings": {
    "temperature": 1.0,
    "top_p": 0.95,
    "seed": 0,
    "thinking_effort": "max",
    "max_tokens": 20000,
    "resident_gb": 8.0,
    "mode": "runtime"
  },
  "messages": "demos/private-document-analysis/local/inputs/rd_tradesecret.json",
  "messages_sha256": "be7464d4721068583b71e38fe13deefe10852eaa34e7c6f1f07d3909f0ef01a7",
  "model_config_sha256": "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
  "commits": [
    "42df3b1"
  ]
}
```

---

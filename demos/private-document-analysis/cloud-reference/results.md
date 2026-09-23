# Kimi-K3 Private Document Evaluation

Model: `moonshotai/kimi-k3`
Run: 2026-09-22T19:49:28
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

# Root-Cause Analysis: Lot OR-26-091 Bridge-Defect Excursion

## Conclusion (high confidence)

The most likely mechanism is a **mis-seated edge gas-ring seal on ETCH-7 Chamber C2**, installed during the Sep 14 18:10 PM, which perturbed edge gas distribution and produced a **locally enhanced etch rate at the wafer edge**. This caused a radial over-etch signature — thinned low-k and positive CD bias at the edge — leading to edge-concentrated bridge defects (most plausibly via erosion/collapse of over-etched, mechanically fragile low-k features).

This is an inference, but it is the only hypothesis consistent with *all* of the evidence, including the near-decisive A/B result in the cross-lot comparison.

## Causal chain

1. Sep 14 18:10 — edge gas-ring seal replaced on C2; engineer later notes it "felt slightly proud," and the PM checklist has **no quantitative seating-height measurement** (p.2, §4).
2. Sep 14 19:05 — post-PM qual used **center-point endpoint and blanket etch-rate checks only** (p.1, §3) — structurally incapable of detecting an edge-localized non-uniformity. This explains why a defective chamber passed qualification.
3. Sep 15 07:20 — OR-26-091 W01–W06 processed on C2 → stable, reproducible radial gradient: low-k thickness −2.4 to −2.6 nm edge vs. center, CD bias +3.9/+4.2 nm at edge, defects 1.55–1.71/cm² at edge vs. 0.10–0.12 at center (p.1, §2). These gradients are ~10x metrology noise (±0.25 nm / ±0.4 nm).
4. Seal reseat on C2 → edge delta recovers to −0.5 nm, CD bias +0.9 nm, defects 0.18/cm² — statistically indistinguishable from C1 (−0.4 nm / +0.8 nm / 0.14/cm²) (p.2, §5). **Restoring the seal geometry restores baseline performance: this is the strongest causal evidence in the package.**

## Supporting evidence

- **Radial co-variation (dose–response):** defects, thickness loss, and CD bias all rise monotonically with radius on C2 (center 0.11 → mid 0.49 → edge 1.62/cm²; p.1, §1–2). A hardware-induced edge perturbation predicts exactly this; a transient or material cause does not.
- **Chamber specificity:** same recipe on C1 is flat and clean (W03; W07–W09 per dispatch history). This single split exonerates every factor common to both chambers: the new photoresist lot (used on both splits, p.2 §4), incoming low-k material, litho, CMP, and ambient humidity.
- **Temporal correlation:** excursion onset coincides with the C2 PM/seal replacement; the first post-PM production lot is the affected one.
- **Reproducibility:** W01 and W02 show nearly identical gradients — a stable hardware condition, not a transient.
- **MFC calibration normal / no alarms (p.1 §3):** consistent, not contradictory — a proud seal is a *geometric* flow-distribution fault; MFCs would read normally.

## Contradicting evidence, and why it does not overturn the hypothesis

- **Post-PM qual passed:** explained by center-point-only measurement (see above). This is a process-gap finding, not exoneration.
- **Chilled-water pressure fluctuation, Sep 15 09:12 (p.2 §4):** tool telemetry shows no chuck-temperature excursion; a thermal transient would not produce a *stable, reproducible radial* gradient across wafers; and the seal-reseat recovery is unexplained by facilities. (One residual check remains — see Experiments, item 1.)
- **Yield engineering's pattern-density suspicion is directly refuted:** high-density die 0.73/cm² vs. low-density 0.69/cm² (p.1 §1) — no density dependence. The "outer die" appearance was a **radius effect confounded with die position**, not density.

## Ranked alternatives

| Rank | Hypothesis | Verdict |
|---|---|---|
| 1 | Mis-seated C2 edge seal → edge etch-rate enhancement → radial over-etch → bridging | Supported by all evidence incl. reseat recovery |
| 2 | Seal-related sub-variants (installation debris, seal material erosion, minor leak at edge) | Same root cause family; needs defect compositional data to separate from #1 |
| 3 | Chilled-water event → edge thermal effect | Contradicted by telemetry, stability of gradient, reseat recovery |
| 4 | New resist lot | Exonerated by chamber split (used on clean C1 wafers) |
| 5 | Humidity / ambient | Exonerated — other lots in same window unaffected |
| 6 | Incoming low-k, litho overlay, CMP | Exonerated — in control and common to both chambers |
| 7 | Pattern density | Directly refuted by data |
| 8 | MFC/gas-flow drift | Calibration normal; would not be fixed by a seal reseat |

## Minimum discriminating experiments

1. **Per-wafer timestamp correlation (uses existing data, zero cost):** W01–W06 processing began 07:20; the chilled-water event was 09:12. If W01/W02 — which already show the full gradient — were etched before 09:12, the facilities hypothesis is definitively dead. Per-wafer timestamps are not in the package and should be pulled.
2. **Quantitative as-found measurement of the removed seal's seating height** vs. spec, plus inspection for erosion/deformation — converts the informal "felt slightly proud" into documented physical evidence.
3. **Defect morphology/composition (SEM + EDX) on edge bridges:** distinguishes over-etch line collapse (supports mechanism #1) from polymer/residue or seal-material particles (supports #2). This refines the sub-mechanism but does not change the root cause.
4. **Formalize the already-run experiment:** the C2-reseat vs. C1 comparison (p.2 §5) is effectively the decisive A/B test; repeat once with a patterned monitor wafer mapped at center/mid/edge to document chamber matching post-repair.

A seal seating-height DOE (intentional height sweep vs. radial etch response) would prove dose–response causality but exceeds the *minimum* needed.

## Missing information and stated assumptions

- **Missing:** per-wafer process timestamps; defect compositional data; as-found seal seating dimension; endpoint/OES traces for C2 during the excursion; confirmation that C1 and C2 share the same chilled-water loop (assumed).
- **Assumptions:** "proud" means protruding into the gas-flow path; positive CD bias + thinner low-k indicates net over-etch at the edge (sign conventions are not defined in the package); the bridge defects are etch-related feature failures rather than deposited foreign material — item 3 above tests this.

## Systemic findings (beyond this lot)

- The **post-PM qualification is blind to radial non-uniformity** (center-point endpoint + blanket etch rate only). Add edge/radial metrology to post-PM qual for any PM touching edge hardware.
- The **PM checklist lacks a quantitative seal seating-height check** — the exact gap that allowed a tactile, subjective assessment ("felt slightly proud") to pass.

## Items requiring external confirmation

Defect compositional analysis (lab SEM/EDX) and the seal seating specification/tolerance (equipment vendor documentation) require evidence outside this package before the sub-mechanism (collapse vs. residue vs. particle) can be closed out. The chamber-level root cause, however, is adequately supported by the supplied data alone.

## Run Metadata

- Finish reason: `stop`
- Prompt tokens: 1485
- Completion tokens: 6057
- Reasoning tokens: 4325
- Total tokens: 7542
- Elapsed: 187.46 sec
- Cost: $0.06932169

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

# Adverse-Event Reconstruction and Failure Analysis

## 1. Event reconstruction

**The decisive arithmetic:** The patient began the *revised* fluid protocol at 08:12 (§4). The model crossed the alert threshold at 10:31 with a score of 0.47 (≥ 0.35 per §1). The policy service returned `suppression_interval=4h` from NMC-SEP-17 (§4), and 10:31 falls 2h19m after protocol start — **inside** the stale 4-hour window (ends 12:12), so the alert was suppressed. Under the revised 2-hour reassessment protocol adopted in March 2026 (§2), the window would have closed at 10:12, and **the 10:31 alert would have fired**. This counterfactual is the single strongest causal evidence in the package: the stale policy object, not the model or code, is the proximate reason no clinician was alerted.

The human safety net eventually functioned: the nurse escalated at 11:58 on clinical signs (§4) — 87 minutes after threshold crossing — and the RRT arrived at 12:17. Note also that the event's temporal association with release 4.8 appears to be a red herring (see §2.1 below); the board should resist anchoring on the release.

## 2. Failure-category verdicts

### 2.1 Model failure — **Ruled out as proximate cause**
- Offline replay of release 4.7, release 4.8, and a frozen pre-adaptation model **all** cross the 0.35 threshold between 10:27 and 10:33 (§5). The 4.8 calibration refresh shifted the crossing by at most a few minutes — immaterial.
- Release 4.8 changed only intercept/slope per the permitted monthly recalibration; "no feature or threshold changes"; self-learning weights "not implemented" (§3) — consistent with the locked operating envelope (§1).
- The patient's subsequent deterioration (11:04 lactate/BP worsening, §4) is directionally consistent with the 0.47 score being clinically meaningful, not spurious.
- *Caveat (inference):* the package contains no calibration/validation metrics for the 4.8 refresh, so "the model behaved as designed" is supported, but "the model was well-calibrated" is not independently established.

### 2.2 Software failure — **No defect; residual design/safeguard question**
- Independent code review: the suppression function was "implemented exactly as specified" (§5). The fallback rule (§3) was never triggered because the policy service was *available* — it simply returned stale data.
- Two design-level gaps remain (inference, not defect):
  1. The fallback addresses **availability**, not **currency/validity**. The 4-hour interval was presumably "validated" at some point, so the system's notion of "last validated configuration" (§3) never checked consistency against the live clinical protocol.
  2. NMC-SEP-17 contains an instability-criteria exception ("unless specified instability criteria are present," §2), but the software appears to consume only a numeric interval from the policy service (§3, §4). Whether the exception is encoded at all is undocumented — a potential policy-*encoding* gap.

### 2.3 Clinical-policy failure — **Yes; the root cause**
- The sepsis committee shortened reassessment to 2 hours in March 2026, but "NMC-SEP-17 was not updated" (§2). The machine-readable policy object and the actual bedside protocol diverged, and the AI's suppression behavior tracked the stale object.

### 2.4 Deployment/change-control failure — **Yes; the operative mechanism, spanning both organizations**
- The March revision "was approved in committee minutes but never entered into the policy-service change-control queue" (§5). Approval without propagation is the break in the chain.
- Vendor side: the clinical integration lead saw the discrepancy on Mar 22 — "I don't see a corresponding SEP-17 ticket; assuming their governance team owns that" (§6) — and did not escalate. This is a noticed-but-unacted-on inconsistency weeks before the incident.
- Hospital side: the analyst "wasn't aware the AI suppression interval was sourced from the older policy object" (§6) — the integration dependency was invisible to the people changing the protocol.

### 2.5 Human error — **No at the bedside; yes organizationally**
- The bedside nurse could not override, defer, or ignore an alert that was never displayed (§4: `ALERT_SUPPRESSED_POLICY`, "no clinician-facing alert"). Escalation at 11:58 was initiated on clinical signs — the safety net worked, late.
- The human failures are upstream and organizational: the committee's failure to route the change, the integration lead's assumption in lieu of escalation, and the hospital's unawareness of the dependency (all §5–6). These are better classified as change-control failures with human contributors than as individual error.

## 3. Causal chain

1. **Root cause:** stale policy object (NMC-SEP-17, 4h) inconsistent with the revised 2h protocol (§2).
2. **Mechanism:** committee approval never entered policy-service change control (§5).
3. **Contributing:** vendor design treats the policy service as authoritative without consistency safeguards (§3, §6 Feb 18); vendor staff noticed the gap and assumed rather than escalated (§6 Mar 22); hospital staff unaware of the dependency (§6 Apr 03).
4. **Proximate event:** 10:31 score 0.47 suppressed by the 4h window (§4).
5. **Mitigation:** nurse escalation 11:58; RRT 12:17 (§4).

## 4. Unresolved facts that could change the analysis

1. **Were NMC-SEP-17's "instability criteria" present at 10:31?** If yes, suppression was improper even under the stale policy — re-opening software/specification failure (does the rule encode the exception?).
2. **Was the revised protocol's 2-hour reassessment performed at ~10:12, and what did it show?** If missed, an independent clinical-operations failure exists; if performed and reassuring, the suppression matters more.
3. **The 54-minute gap** between detected worsening (11:04) and escalation (11:58) — were instability criteria met at 11:04?
4. **Patient outcome and harm causation** — "serious adverse event" is asserted but no outcome data is supplied; whether the 87-minute delay caused the harm is a clinical-causation question.
5. **Validation records for the 4.8 recalibration** — bears on the vendor's "validated operation" obligation (§7).
6. **What safeguards were considered** under the vendor's "reasonable safeguards against configuration inconsistency where technically feasible" clause (§7) — staleness checks, version effective-dates, suppressed-alert visibility.
7. **Whether formal notification** of the March revision was ever given (the Mar 22 email shows informal vendor awareness, §6), and the exact incident date relative to Mar 22.
8. **Operational ownership of policy-service content** — hospital governance or vendor-managed — is not stated.

## 5. Highest-priority corrective controls

1. **Immediate:** reconcile NMC-SEP-17 with the 2-hour protocol and push the corrected interval through change control; audit *all* policy objects for similar drift.
2. **Couple clinical approval to deployment:** no protocol go-live without a corresponding policy-service ticket that must be closed/acknowledged — closing the committee-to-queue gap (§5).
3. **Vendor safeguard (maps to §7 obligation):** policy-object versioning with effective dates and staleness/consistency alerts when a referenced protocol is newer than the suppression config; fail-safe toward alerting on ambiguity.
4. **Surface suppressed alerts** (e.g., passive bedside indicator that an alert was policy-suppressed) so clinicians are not blind to suppression events; audit `ALERT_SUPPRESSED_POLICY` logs as near-misses.
5. **Escalation duty for noticed discrepancies:** the Mar 22 pattern (§6) — observed inconsistency resolved by assumption — should trigger mandatory cross-organizational escalation.
6. **Encode the instability-criteria exception** in the suppression rule, or document explicitly why it cannot be.

## 6. Responsibility mapping (not liability)

The contract (§7) assigns the hospital clinical policies, configuration approval, and timely change notification — all implicated by §2 and §5. It assigns the vendor validated software operation (supported by §5 code review and §3 release notes) **plus** a safeguard obligation that the Feb 18 design choice (runtime dependence on the hospital policy service, §6) and the unescalated Mar 22 observation (§6) leave contested. Per instructions, I draw no liability conclusion: allocation depends on the unresolved facts in §4 (especially notice, feasibility of safeguards, and harm causation) and requires legal review.

## Run Metadata

- Finish reason: `stop`
- Prompt tokens: 1248
- Completion tokens: 6856
- Reasoning tokens: 4899
- Total tokens: 8104
- Elapsed: 71.92 sec
- Cost: $0.1064139

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

# HelioForge Systems — Board-Level R&D / Trade Secret / Invention Strategy Memo

**Basis:** Confidential board package (2 pp.). Facts are cited to the document; inferences and assumptions are labeled. Nothing below is a patentability opinion — every patent candidate requires outside prior-art searching and patent counsel.

---

## Executive Summary — Five Highest-Value Findings

1. **The company is in a priority race it has not entered.** A competitor already advertises "predictive liquid cooling using workload telemetry" (§5), and HelioForge's own sales materials disclose "predictive cooling" and "factory-calibrated cold plates" (§6) — yet **zero invention disclosures exist** for the four underlying inventions (§6). First-sale and first-disclosure dates must be established immediately; statutory clocks may already be running.
2. **The most defensible invention is a combination, not a feature.** Exp. 26-052 shows the 37% overshoot reduction emerges only from feed-forward + compressibility estimation + accumulator pre-charge coupling (the effect vanishes with >12% pre-charge change). The combination is both harder to design around and harder to attack as obvious than any element alone.
3. **There is an internal contradiction in the obvious strategy.** "Patent the controller, keep the pre-charge coupling secret" may not be coherent: because the benefit *depends* on the pre-charge coupling (Exp. 26-052), a patent application that omits it may fail enablement. This must be resolved with counsel **before** drafting.
4. **The per-unit geometry concept is already exposed; only the transformation is protected.** Destructive teardown reveals the channel geometry (§5). Secrecy of the *concept* is therefore already lost upon sale — the choice is patent or nothing. The generation algorithm remains protectable (§5).
5. **The supplier is a larger inference channel than the customer, and the document's own reverse-engineering analysis ignores this.** The supplier accumulates many units' unique machining coordinates over rolling 30-day retention (§3); the customer-facing analysis in §5 does not address this threat model.

---

## 1. Concepts Materially Distinct from the Existing Claims

The existing families cover: a **static structure** (HF-101: parallel microchannel cold plate), **rack plumbing hardware** (HF-118: manifold, redundant pumps, pressure sensors), and **reactive feedback control** (HF-133: flow as a function of *measured package temperature*).

| # | Concept | Evidence | Why distinct from HF-101/118/133 |
|---|---------|----------|----------------------------------|
| A | **Feed-forward control from the derivative of workload power demand** | Exp. 26-041 (power derivative predicts hotspot 180–260 ms before the temperature sensor responds); §4 | HF-133 is feedback from measured temperature. This is a different control paradigm (predictive vs. reactive), a different input signal, and it cures a failure mode HF-133 cannot: overshoot persists "even when average coolant flow is increased" (Exp. 26-041). |
| B | **Online compressibility estimation from pressure-wave response** | Exp. 26-052; §4 | No existing claim covers fluid-property estimation. Note (inference): HF-118 claims pressure *sensors*; if the estimator uses those sensors, this is a new *use* of patented hardware — a layering opportunity (see §3 below). Hardware dependency must be confirmed — missing information. |
| C | **Accumulator pre-charge coupling / calibration table linking the estimator to pre-charge** | Exp. 26-052 (effect disappears beyond ±12% pre-charge change); §4 | Wholly absent from the portfolio. This is a discovered dependency, not merely a setting. |
| D | **Per-unit adaptive microchannel geometry generated from post-assembly thermal maps** | Exp. 26-061 (±6% width variation counteracting per-package heat-flux asymmetry); §3 | HF-101 claims a fixed geometry. This is a *manufacturing method* (measure → transform → machine) producing a per-unit product. |
| E | **The integrated per-unit thermal co-design methodology** (D for spatial asymmetry + A/B/C for transient overshoot) | Exp. 26-041/052/061 read together | No single claim covers a factory-to-firmware per-unit customization loop. |

---

## 2. Patent Candidates vs. Trade-Secret Know-How

**Decision logic used:** (i) can a competitor detect the feature from outside the product; (ii) is the concept already becoming public; (iii) can infringement be proven if patented; (iv) does patent disclosure destroy a more valuable secret.

| Item | Preliminary recommendation | Rationale (evidence) |
|------|---------------------------|----------------------|
| A. Feed-forward controller | **Patent candidate (urgent)** | The high-level concept is already being marketed by a competitor (§5) and by HelioForge itself (§6) — independent invention and first-to-file loss are live risks. External behavioral signature may be detectable: customers "can measure external coolant pressure" (§5), so a step-workload test could reveal predictive actuation — which also makes infringement *provable*. **Obviousness risk is real** (feed-forward is a known control technique generally); prior-art search required. |
| D. Geometry-generation method | **Patent candidate** | The product itself discloses the feature on teardown (§5); trade secret on the *concept* is fragile by construction. A method claim (thermal map → transformation → machining) covers what a competitor must replicate. Quantified value exists (yield 91.2% → 97.8%, §3) — useful for prosecution and damages. |
| B. Compressibility estimator | **Patent candidate, likely filed with A** | Exp. 26-052 used A+B together; if B is necessary to enable A's claimed benefit, it will be disclosed anyway. Filing separately or as dependent claims is a counsel decision. |
| C. Pre-charge coupling / calibration table | **Split decision required** | The *dependency* may have to be disclosed to enable A/B (Exp. 26-052). The *specific calibration table values* should remain trade secret if enablement permits — counsel must resolve this tension (flagged in Summary #3). |
| Transformation coefficients, proprietary script, thermal-map processing parameters | **Trade secret** | Stored on access-controlled build server; operators see only job IDs (§3). Not externally detectable (§5). Classic know-how. |
| Pre-charge sensitivity data (±12% window), yield data, the 180–260 ms prediction window | **Trade secret** | This characterization data is competitively valuable and not needed in marketing; publish nothing. |
| Supplier information-splitting architecture | **Trade secret (do not patent)** | Disclosing it in a patent would defeat its purpose (§3). |

**Caution (per instructions):** non-coverage by HF-101/118/133 establishes only novelty *over HelioForge's own portfolio*, not over the prior art. The competitor's advertising (§5) suggests an active field.

---

## 3. Combinations More Defensible Than Individual Features

1. **A + B + C (feed-forward + compressibility estimation + pre-charge coupling).** Exp. 26-052 demonstrates the technical effect is *emergent*: the 37% improvement disappears when pre-charge shifts >12%. A combination claim (i) tracks the actual product, easing infringement proof; (ii) resists design-around, because dropping an element sacrifices the benefit; (iii) is less vulnerable to an obviousness attack on feed-forward control alone.
2. **HF-133 feedback + A feed-forward (hybrid controller).** The product almost certainly practices both (§4 lists PID feedback as current implementation). A hybrid claim blocks a competitor from escaping by arguing "feed-forward only."
3. **HF-118 hardware + B (compressibility estimation using the manifold's pressure sensors).** *Inference — hardware dependency unconfirmed.* If confirmed, this layers new method claims onto existing apparatus claims, extending the effective scope of HF-118.
4. **D + A/B/C (per-unit geometry + per-unit control calibration) as a system claim.** Fact: D addresses steady-state spatial asymmetry (Exp. 26-061); A/B/C address transient overshoot (Exp. 26-041/052). Inference (no demonstrated interaction is documented): together they define a "measure-and-customize per unit" system spanning factory and firmware — a broader narrative than any element and well matched to how the product is sold ("factory-calibrated cold plates," §6).

---

## 4. Exposure Assessment

**Reverse engineering (customers/competitors):**
- Geometry: **exposed** via destructive teardown (§5). Only the generation algorithm is protected, and only because inferring it requires multiple units *plus* pre-machining thermal maps customers lack (§5).
- Controller: firmware is protected (secure boot, signed updates, §5), but **black-box system identification is possible** through customer-accessible pressure/temperature/performance measurements (§5). The competitor's advertisement (§5) shows the concept space is already populated.

**Supplier (under-analyzed in the document):**
- The supplier receives final coordinates for every unit, retained 30 days (§3). The document's inference-resistance argument (§5) addresses *customers*; a supplier holding thousands of per-unit coordinate sets could statistically characterize the ±6% variation and correlate it with package SKUs — and coordinates plus one torn-down unit begin to map inputs to outputs. This is inference, not documented fact, but the threat model is absent from §5 and should be closed.
- Contractual protections exist (no reverse engineering, no reuse; §3) but audit rights, jurisdiction, and sole-source status are **missing information**.

**Marketing / statutory disclosure:**
- Sales presentations disclose the *existence* of both capabilities (§6). Whether any deck, demo, or white paper crossed into enabling disclosure — and on what dates — is **missing information** and is the single most time-sensitive legal question (grace periods / absolute novelty / on-sale bar, especially since §3 and §4 show these features are *in production and in current implementation*).

**Internal:**
- No invention disclosures; engineering treats four inventions as "implementation details" (§6). Risks: departure of named inventor L. Chen (Exp. 26-041) and unnamed contributors; inadvertent publication; weak conception records. Whether notebooks are dated/witnessed and whether assignment agreements exist is **missing information**.
- **Cross-disciplinary risk:** the ±12% pre-charge sensitivity (Exp. 26-052) is also a *product* risk. Field drift of accumulator pre-charge could silently erase the marketed "predictive cooling" benefit — creating warranty/misrepresentation exposure that interacts directly with the sales claims in §6. Whether service procedures or service tools expose the calibration table is **missing information**.

---

## 5. Inventions Management May Not Realize It Has

1. **The pre-charge coupling itself** (Exp. 26-052) — treated as a detail, but it is a non-obvious discovered dependency without which the headline 37% result does not exist. A plausible extension (inference, not documented): a *self-diagnostic method* that detects pre-charge drift via the compressibility estimator and re-calibrates — worth an invention disclosure and an engineering experiment.
2. **The manufacturing data pipeline as a method** (thermal map → coefficients → laser offsets, §3) — a patentable manufacturing method with quantified effect (yield 91.2% → 97.8%), currently treated as "a script."
3. **The per-unit co-design system** (factory geometry customization + per-unit control calibration) — a system-level invention larger than any single disclosure.
4. **A new use of HF-118's patented pressure sensors** (compressibility estimation) — potential to deepen an existing family (inference pending hardware confirmation).
5. **Test/characterization methodology** (the 180–260 ms prediction window, the 99.9th-percentile overshoot metric) — potentially protectable metrology know-how; at minimum, keep secret.

---

## 6. Recommended Next Steps

**Diligence experiments (engineering):**
1. **Identifiability test:** can an external observer using only customer-accessible measurements (§5) distinguish feed-forward from feedback control? Drives the patent-vs-secret call and claim drafting.
2. **Ablation/design-around matrix:** feed-forward alone; estimator alone; fixed asymmetric geometry vs. per-unit geometry. Establishes which elements are essential — informs claim breadth and what must stay secret.
3. **Red-team geometry inference:** attempt to reconstruct the transformation from N torn-down units without thermal maps; validate §5's assumption and quantify the supplier-inference risk.
4. **Pre-charge drift study:** field-relevant drift rates vs. the ±12% window; decide whether a self-calibrating estimator is needed (generates both a product fix and a new disclosure).

**Documentation (immediate):**
5. File **four invention disclosures** (A, B, C, D) plus the drift-monitoring extension; conduct inventorship interviews (L. Chen and contributors); ensure notebooks are dated and witnessed.
6. **Marketing disclosure inventory:** collect every deck/demo with dates and audiences — counsel needs this for grace-period analysis.
7. Verify employee/contractor invention assignments, NDAs, exit procedures, build-server access logs, and supplier audit rights.

**Protection decisions (with counsel, before the financing):**
8. Prior-art searches in four areas (feed-forward thermal control on power telemetry; compressibility estimation; adaptive microchannel manufacturing; accumulator-coupled cooling control) **plus FTO and a patent watch on the competitor** (§5).
9. Resolve the **enablement/secrecy tension** on the pre-charge coupling before drafting the control application.
10. Determine **first-sale/offer dates and jurisdictions** for units embodying A–D; assess on-sale/public-use consequences domestically and in absolute-novelty jurisdictions.
11. Sequence filings: control-combination application and manufacturing-method application first; keep coefficients, calibration tables, and characterization data as documented trade secrets; consider defensive publication for anything deliberately not filed.

---

## Matters Requiring Outside Expertise (explicit)

- **Patent counsel:** patentability/obviousness of A–D; enablement vs. trade-secret conflict on the pre-charge coupling; inventorship; continuation/divisional strategy relative to HF-101/118/133 (status of those families is missing information); foreign filing given manufacturing footprint (location unknown).
- **Prior-art search firm:** the four search areas above; competitor filing surveillance.
- **Trade-secret counsel:** adequacy of "reasonable measures" (supplier contract, 30-day retention, access controls); supplier jurisdiction and enforceability; employee mobility risk.
- **Patent litigator/FTO counsel:** exposure to the competitor's "predictive liquid cooling" program.
- **Commercial counsel:** whether marketing statements (§6) create disclosure, warranty, or misrepresentation exposure given the pre-charge sensitivity (Exp. 26-052).

*All conclusions above are drawn solely from the supplied package; items flagged as inference or missing information should be validated before board action.*

## Run Metadata

- Finish reason: `stop`
- Prompt tokens: 1461
- Completion tokens: 9821
- Reasoning tokens: 6567
- Total tokens: 11282
- Elapsed: 98.59 sec
- Cost: $0.1512957

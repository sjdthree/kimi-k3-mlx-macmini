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
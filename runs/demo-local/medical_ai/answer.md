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
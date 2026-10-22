Let me carefully analyze this semiconductor fabrication excursion case.

Key facts from the document:

**Excursion summary:**
- Bridge defect density: 0.08 → 0.71/cm² (8.9x increase)
- Strong radial gradient: center near baseline (0.11), mid elevated (0.49), edge very high (1.62)
- No pattern density dependence (high-density 0.73 vs low-density 0.69) — kills the yield engineer's pattern density suspicion

**Wafer metrology:**
- W01, W02 (Chamber C2): thickness gradient center→edge 62.1→59.7 nm (~-2.4 nm), CD bias +0.4→+3.9 nm (edge), defects 0.10→1.55
- W03 (Chamber C1): flat thickness (62.2→61.7, -0.5 nm), CD bias +0.3→+0.8, defects low (0.08→0.14)
- Measurement repeatability: ellipsometry ±0.25 nm, CD-SEM ±0.4 nm — so the gradient is real

**Tool history:**
- Sep 14 18:10: ETCH-7/C2 PM completed; edge gas-ring seal replaced
- Sep 14 19:05: Post-PM qual passed — but only center-point endpoint and blanket etch-rate checks (no radial/edge checks!)
- Sep 15 07:20: Lot OR-26-091 begins on C2, W01-W06
- Sep 15 13:40: W07-W09 processed on C1 after dispatch reassignment — these are fine
- Sep 15 17:30: MFC calibration normal, no alarms

**Notes/distractors:**
- Litho: no overlay excursion
- Incoming low-k material: passed specs
- CMP: within control
- Humidity 6% above mean, but other lots in same ambient window unaffected
- New photoresist lot used on BOTH affected and unaffected chamber splits — so resist lot is common cause candidate but chamber split difference rules it out
- Chilled-water pressure fluctuation Sep 15 09:12 — but telemetry shows no chuck temp excursion; also would presumably affect C1 too (facilities-level) and wouldn't produce a fixed radial signature tied to one chamber
- Chamber engineer noted replacement edge seal "felt slightly proud" — no quantitative seating-height measurement in PM checklist

**Cross-lot comparison (the decisive table):**
- C2 immediately post-PM: edge thickness delta -2.5 nm, edge CD bias +4.0 nm, edge bridge defects 1.66/cm²
- C2 after seal reseat: -0.5 nm, +0.9 nm, 0.18/cm²
- C1 same recipe: -0.4 nm, +0.8 nm, 0.14/cm²

This is the smoking gun: reseating the edge seal on C2 returns the chamber to C1-like behavior. The mechanism: improperly seated/proud edge gas-ring seal → altered edge gas flow/conductance → local over-etch at wafer edge (thinner low-k at edge, -2.4 to -2.5 nm) → positive CD bias (CD larger, meaning... wait, let me think about what CD bias means here).

CD bias +3.9 nm at edge with thickness loss of 2.4 nm. If this is an etch process for low-k interconnect (trench etch for damascene?), positive CD bias could mean trenches etched wider than target — over-etch/isotropic component → trench widening → adjacent lines closer together → bridge defects (metal bridging after fill, or resist/low-k bridges). Actually "post-etch bridge defects" — bridges between features after etch. If trenches are etched too wide (CD bias positive), the remaining low-k lines between trenches become too thin and can collapse or the spacing shrinks causing subsequent metal bridging. Alternatively, if CD bias positive means lines are wider (space narrower), bridges occur.

Either way, the physics: edge gas-ring seal proud → perturbed gas injection/conductance at edge → locally higher etch rate at edge → thinner low-k at edge + larger CD bias → bridges concentrated at edge.

The mechanism chain:
1. PM replaced edge gas-ring seal on C2 (Sep 14 18:10)
2. Seal seated "proud" (slightly raised) — informal note, no quantitative check
3. Post-PM qual only checked center-point endpoint and blanket etch rate — missed radial nonuniformity
4. Proud seal → altered edge gas flow distribution (either enhanced flow or changed local pressure/conductance, possibly also altered plasma sheath edge behavior) → edge over-etch
5. Edge over-etch → low-k thickness -2.4 nm at edge, CD bias +3.9-4.2 nm at edge
6. CD excursion at edge → bridging (spaces too narrow / lines too thin) → radial defect signature
7. Chamber-split: W03/W07-W09 on C1 unaffected — same resist lot, same ambient, same recipe → isolates C2 hardware
8. Seal reseat → C2 returns to baseline — confirms seal seating as root cause

Supporting evidence:
- Radial gradient in defects (0.11/0.49/1.62) matches radial gradient in thickness (-2.4 nm edge) and CD bias (+3.9 to +4.2 nm at edge) — all three co-localized at edge
- Chamber specificity: C2 affected, C1 not, same lot, same recipe, same resist lot, same time window (W03 vs W01/W02; W07-W09 after reassignment)
- Temporal correlation: PM with edge seal replacement immediately before excursion
- Seal reseat fixes it (cross-lot table)
- Gradient exceeds metrology noise (±0.25 nm, ±0.4 nm)
- No pattern density dependence — contradicts yield engineering's initial suspicion
- MFC calibration normal — not a bulk gas flow issue; consistent with a local mechanical/geometric disturbance at the edge ring
- No alarms — a mechanical mis-seat wouldn't trigger MFC/endpoint alarms, and center endpoint check would pass

Contradicting/complicating evidence (against the seal hypothesis, or things that need explaining):
- Post-PM qual passed — but it only measured center-point endpoint and blanket etch rate; a proud edge seal would not be caught by these. So not a true contradiction, but explains why it escaped.
- MFC calibration normal — someone might argue flow is fine; but the seal affects local edge conductance/injection geometry, not MFC output. Not a contradiction.
- The chilled-water fluctuation at 09:12 during the run — could be an alternative, but telemetry shows no chuck temp excursion; also would not produce chamber-specific signature (C1 wafers processed later on the same day are fine; and presumably the water system is shared). Also a transient event wouldn't give a stable radial gradient across all C2 wafers.
- New photoresist lot — common to both chambers, so cannot explain chamber split.
- Humidity — other lots in same window unaffected.
- The "proud seal" note is informal, not quantified — weak link; the discriminating experiment is to measure seating height.

Alternative hypotheses ranked:
1. Mis-seated/proud edge gas-ring seal on C2 → edge over-etch (most likely; all evidence fits; reseat is the natural experiment and the cross-lot table essentially shows it works)
2. Edge ring hardware damage/contamination or wrong seal part installed during PM (same family; would be distinguished by inspection/measurement)
3. C2-specific MFC or gas distribution fault (contradicted by normal MFC calibration and no alarms; also less likely to produce pure edge-only signature) — but a partially clogged/changed edge injector could do it
4. Chuck temperature edge nonuniformity (e.g., ESC edge zone) — chilled water fluctuation is a red herring; no telemetry excursion; would likely show time-dependence and possibly affect other tools
5. Photoresist lot / incoming material / CMP / litho — all ruled out by commonality (used on both splits) or in-control data
6. Pattern density / layout — ruled out by density-split data
7. Ambient humidity — ruled out by other lots unaffected
8. Edge bead removal / wafer edge die issue — defects observed to 145 mm (wafer edge region 120-145 mm), but thickness gradient extends from mid radius; a pure edge-bead issue wouldn't give mid-radius elevation and thickness gradient.

Wait, actually let me reconsider: edge 120-145 mm on a 300 mm wafer — radius 150 mm. So edge band is near wafer edge, consistent with edge gas ring influence.

Minimum discriminating experiments:

The cross-lot table already shows "C2 after seal reseat" returning to baseline — that is essentially the decisive experiment and it's already done. But the prompt asks to propose minimum experiments needed to distinguish hypotheses. So:

1. **Quantitative seal seating-height measurement / leak and conductance check on C2** vs C1 (e.g., feeler gauge/height metrology of installed seal, or helium leak rate around edge ring, or pressure rise test). This directly tests the "proud seal" mechanism. Minimum: measure seal seat height at multiple azimuths on C2; compare to C1 and to spec.

2. **Chamber swap / wafer split confirmation**: Run patterned monitor wafers (same lot/resist) split across C2 and C2-post-reseat and C1, with full-radius radial metrology (thickness + CD at ≥3 radial bands, multiple azimuths). Actually already partially done in Table 5. The minimal new experiment: a single radially-characterized wafer per condition.

3. **Radial etch-rate mapping on blanket low-k wafers** in C2 vs C1: blanket etch rate at center/mid/edge. Post-PM qual only did center — repeat with radial mapping. If edge rate elevated in C2 pre-reseat and normal post-reseat, confirms local edge over-etch. This distinguishes seal/gas-distribution cause from e.g., plasma/chuck causes? Partially.

4. **Azimuthal mapping**: a mis-seated seal is often localized azimuthally (seated proud at one azimuth). Mapping defects/CD/thickness vs azimuth would show asymmetry correlated with seal installation orientation; a uniform gas-flow/MFC issue would be azimuthally symmetric, chuck thermal might have its own pattern (e.g., ESC zone boundary). This is a cheap discriminator: reuse existing wafers W01/W02 for azimuthal defect maps — no new runs.

5. **Check chuck edge-zone temperature / ESC** to rule out thermal: in-situ or post-hoc thermal map; review telemetry (already shows none). Minimal: interrogation of existing telemetry is free.

6. **Inspect the replaced seal** (if retained): dimensional check vs new part; check for wrong part number or damage.

7. Gas flow visualization / pressure gauge at edge ring plenum if available.

The truly minimal set:
- (a) Measure the installed seal height/profile on C2 (and compare to C1) — direct test of root cause; zero wafer cost.
- (b) Azimuthal analysis of existing W01/W02 defect and CD maps — free, distinguishes mechanical mis-seat (asymmetric) from uniform flow/hardware drift.
- (c) Radial blanket etch-rate qual on C2 vs C1 (or repeat post-PM qual with radial sampling) — one wafer per chamber, directly reproduces the gradient and tests fix effectiveness.
- (d) If needed, controlled seal-reseat A/B: process monitor wafers pre/post reseat — but Table 5 shows this was done, confirming the fix.

Also the CD bias sign: positive CD bias — if the etched features are trenches in low-k (damascene), a positive bias (wider trenches) narrows the remaining low-k line, which could then have bridges after metal fill? Hmm, bridging is usually metal lines shorting. If trenches widen, the space between adjacent metal lines shrinks → metal bridge risk increases. Or if positive bias means etched line CD larger (for a line-etch), spaces shrink → bridge. Either interpretation supports "edge over-etch → geometry at edge → bridges." The thickness loss indicates more aggressive etch at edge (higher rate or longer effective time). Over-etch into/low-k thinning with CD growth is consistent with a local plasma/flow enhancement at the edge.

Physical mechanism detail: A proud edge gas-ring seal can (i) partially obstruct or redirect edge gas injection, changing local pressure and radical flux; (ii) alter the sheath/plasma boundary at the wafer edge, increasing local ion/radical flux; (iii) create a leak path introducing local O2/air or changing conductance — air leak would also increase etch rate of organic low-k (oxygen ashing) → thinning + CD growth. Actually a leak at the edge seal introducing air (O2/N2) would preferentially ash the low-k (organic) → thickness loss + CD bias at edge — that's a plausible micro-mechanism too. Helium leak check would discriminate "leak" vs "flow redistribution." But either way root cause is the seal.

Note that "edge gas-ring seal" — the seal on the gas ring; proud seating could change the gap/injection angle. Both leak and flow-redistribution sub-mechanisms are consistent. Discriminating between them: helium leak test or residual gas analyzer (RGA) during process (O2/N2 signature), vs. flow modeling.

Evidence contradicting the main hypothesis — to be honest:
- The post-PM qualification passed (weakens but doesn't contradict, since it only measured center).
- MFC normal, no alarms (consistent — a mechanical seal issue wouldn't alarm).
- The engineer's note is informal/subjective; no quantitative measurement existed — the "proud" observation is hearsay until measured.
- Table 5's "C2 after seal reseat" row strongly supports, but one might ask whether other interventions happened at reseat (confound). Minimal control: confirm no other PM actions during reseat; or reseat alone as single variable.
- Time order: chilled-water event 09:12 during C2 run W01-W06 (07:20 start). But W03/W07-09 on C1 at 13:40 are fine; if the water event were causal, only wafers in the chamber at 09:12 would show a spike, not all six C2 wafers uniformly — and Table 5 shows the signature persisted "immediately post-PM" as a stable state, not a transient. So stable hardware cause > transient facilities cause.

Also address: why did defects appear "more often in outer die" leading yield to suspect pattern density? Because outer die = edge radial band; radial over-etch. Density split shows no dependence. Good to explicitly falsify the initial suspicion.

Ranking alternatives (with likelihood):
1. Proud/mis-seated C2 edge gas-ring seal → edge over-etch (local flow/conductance/leak perturbation). Likelihood: high.
2. Wrong/damaged seal part or edge-ring hardware issue from same PM. Medium-low (same family; distinguished by part inspection/dimensional check).
3. C2 gas delivery nonuniformity independent of PM (e.g., partially clogged edge injector, MFC drift). Low: MFC cal normal; no alarm; and fixed by seal reseat (Table 5) — unless reseat coincidentally cleared a clog.
4. Edge thermal nonuniformity (chuck/ESC edge zone, chilled water). Low: no telemetry excursion; transient event vs stable signature; shared facility but C1 unaffected.
5. Resist lot/material/CMP/litho/humidity. Very low: common across splits or in-control or other lots unaffected.
6. Pattern density. Ruled out.

Minimum experiment set — let me define "minimum" carefully: the smallest number of cheap, decisive tests that discriminate hypothesis 1 from 2-4:

Experiment 1 (no wafers): Dimensional metrology of installed edge seal on C2 vs C1 (seating height at multiple azimuths; e.g., dial indicator/feeler/laser profilometer), plus visual/part-number check of the removed seal. Discriminates 1 vs 2,3,4. If a localized high spot is found at a specific azimuth → 1 confirmed; if uniform wrong part → 2.

Experiment 2 (no wafers): Helium leak check / chamber pressure-rise (rate-of-rise) test on C2 vs C1 baseline. Discriminates leak sub-mechanism (1a) from flow-redistribution sub-mechanism (1b) and from generic hardware fault. RGA during a dummy run could substitute.

Experiment 3 (existing data, no cost): Azimuthal defect/CD/thickness mapping on W01/W02 from existing defect maps (defect inspection tools store coordinates). Localized azimuthal asymmetry → mechanical seat; azimuthally uniform ring → gas delivery/thermal ring. This is nearly free and highly discriminating.

Experiment 4 (1-2 wafers): Radial blanket etch-rate requalification on C2 vs C1 (and post-reseat C2) with ≥9-point radial thickness maps. Confirms edge-rate elevation and validates fix; upgrades the PM qual procedure (center-only → radial).

Optionally Experiment 5: review high-resolution chuck-temperature telemetry during 07:20-13:40 for C2 to formally close the thermal path (free data pull).

If Table 5 is taken as already executed (it is in the package), then the residual experiments are 1-3 to nail the micro-mechanism and rule out confounds at reseat.

Also propose corrective/preventive action as a side note? The prompt asks for mechanism, evidence, alternatives, minimum experiments. Maybe a brief note on qual gap (add radial uniformity + seal seating-height check to PM checklist). Could include briefly.

Also note detection gap: bridge defects correlate quantitatively with CD bias: edge +3.9-4.2 nm → 1.55-1.71; mid +1.2-1.5 → 0.46-0.52; center +0.4 → 0.10. Rough monotonic relation — consistent with CD-driven bridging. Thickness also co-varies. Which is causal? Bridges likely from space CD shrink (positive line bias or trench widening). Thickness loss could contribute (e.g., low-k mechanical integrity, dishing) but CD bias is the proximal geometric cause. The two are coupled via edge over-etch. Could mention: discrimination between "CD bias causes bridges" vs "thickness loss causes bridges" — a designed experiment: vary edge CD at fixed thickness (trim etch time) — but that's beyond minimum; the fix (reseat) removes both. For minimum discrimination of root cause, not needed. But I can note that the proximal defect driver is edge CD bias/space narrowing, with thickness loss as a co-symptom of the same over-etch.

One more check on numbers: W01: center 62.1, edge 59.7 → Δ = -2.4 nm. W02: 62.0 → 59.4 → -2.6 nm. Table 5 says -2.5 nm for C2 post-PM. Consistent. C1 W03: 62.2 → 61.7 → -0.5 nm; Table 5 says -0.4 for C1 — close enough (average). Post-reseat -0.5. Good.

CD bias: W01 edge +3.9, W02 +4.2 → Table 5 +4.0 (avg ~4.05). Fine.

Defects: W01 edge 1.55, W02 1.71 → Table 5 edge 1.66. Fine.

Ellipsometry repeatability ±0.25 nm vs gradient 2.4-2.5 nm → ~10σ. CD-SEM ±0.4 vs +3.9 → ~10σ. Real.

Structure of my answer:
1. Executive conclusion (mechanism)
2. Mechanism chain
3. Supporting evidence (mapped: each datum → implication)
4. Contradicting/weak evidence and how it's reconciled
5. Ranked alternatives with reasons
6. Minimum discriminating experiments (with what each discriminates and expected outcomes)
7. Optional: gaps in containment/qual that allowed escape (brief)

Formatting: Use headers, concise bullets, a table for alternatives and experiments maybe. This is a technical review; markdown is appropriate. Length: substantial but not bloated. The user asked for thorough analysis with citations. I'll aim for a well-structured report, maybe 800-1200 words. Use tables where helpful. Avoid excessive nesting.

Let me draft.

**Most likely mechanism:** A mis-seated ("proud") edge gas-ring seal installed during the Sep 14 18:10 PM on ETCH-7 chamber C2 perturbed gas delivery/conductance at the wafer edge, producing locally elevated etch rate at radii >~70 mm, especially 120-145 mm. This edge over-etch thinned the low-k film (~-2.5 nm edge-to-center) and drove positive CD bias (~+4 nm) at the edge, narrowing spaces between features until bridge defects formed. Radial, chamber-specific, PM-time-locked, and reversed by seal reseat.

Supporting evidence — enumerate:
1. Chamber split: W01/W02 (C2) show gradient; W03 (C1) flat. Same lot, same recipe window, same resist lot (used on both splits), same ambient. Isolates C2 hardware, not materials/environment.
2. Radial co-localization: defects 0.10→1.55 track thickness 62.1→59.7 and CD +0.4→+3.9. All three share edge signature; mid-radius partially affected (0.46-0.52 defects, +1.2-1.5 CD) consistent with a falloff from an edge source.
3. Time lock: first PM after which seal replaced (Sep 14 18:10) precedes first affected lot (Sep 15 07:20). No other C2 change recorded.
4. Reversibility: Table 5 — C2 post-reseat matches C1 (-0.5 vs -0.4 nm; +0.9 vs +0.8 nm; 0.18 vs 0.14/cm²). Single strongest causal datum.
5. Signal >> noise: gradients ~10x repeatability.
6. Negative controls: MFC cal normal + no alarms (mechanical seat wouldn't alarm); litho/CMP/incoming material in control; humidity refuted by unaffected contemporaneous lots; resist lot refuted by C1 splits; density hypothesis refuted by density split (0.73 vs 0.69).
7. Escape route explained: post-PM qual measured only center endpoint + blanket rate — a proud edge seal is invisible to both.

Contradicting / to reconcile:
- Post-PM qual pass — explained by center-only sampling; recommend radial sampling. Not contradictory.
- "Proud" note is informal, unquantified — root cause rests on reseat reversibility + chamber split, not the anecdote; measurement needed (Experiment 1).
- Chilled-water fluctuation 09:12 — telemetry shows no chuck temp excursion; facility-level transient would not produce a stable, chamber-specific radial state across all six C2 wafers and persist as a stable signature (Table 5); C1 wafers later same day unaffected.
- Resist lot — common to both affected and unaffected splits; cannot produce chamber-specific effect.
- MFC normal — addresses bulk flow only; seal affects local edge conductance/injection geometry or leak, invisible to MFC calibration.

Ranked alternatives:
1. (Primary) Proud/mis-seated edge gas-ring seal → edge over-etch. 
2. Wrong/damaged seal part or edge-ring hardware defect introduced at same PM (e.g., wrong P/N, nicked seal → leak). Same signature; distinguished by part inspection/leak test.
3. C2 gas-distribution fault unrelated to seal (partially clogged edge injector; MFC nonlinearity at recipe flow). MFC cal normal and reseat fix argue against; a clog coincidentally cleared at reseat is possible — check reseat work record for single-variable confirmation.
4. Edge thermal nonuniformity (ESC edge zone / chilled water). No telemetry excursion; transient vs stable; shared facility. Low.
5. Materials/litho/CMP/humidity/resist. Effectively excluded by commonality/controls.
6. Pattern density (initial suspicion). Excluded by density split.

Minimum discriminating experiments (minimum = smallest decisive set):
E0 (free): Azimuthal re-analysis of W01/W02 existing defect/CD/thickness maps. Expectation: if proud seal, asymmetry localized to seat azimuth (or ring with azimuthal weighting); uniform ring → symmetric flow/thermal cause. Distinguishes 1 vs 3/4.
E1 (free, no wafers): Dimensional metrology of installed seal seat height at multiple azimuths on C2 vs C1/spec; inspect removed seal for damage/wrong P/N. Direct test of 1 vs 2.
E2 (free): Helium leak/rate-of-rise test on C2 vs C1; or RGA during process for N2/O2. Distinguishes leak vs flow-redistribution micro-mechanism; further separates 1/2 from 3.
E3 (2-3 wafers): Radial blanket etch-rate maps (≥9 point, center/mid/edge) on C2 and C1; confirm elevated edge rate reproduces only in affected state. Upgrades post-PM qual.
E4 (free): Pull C2 chuck/edge-zone temperature telemetry for Sep 15 07:20-13:40 to formally close hypothesis 4.
Optional single-variable confirmation: ensure the reseat intervention changed only the seal seat (review maintenance log); if not, one A/B monitor wafer before/after.

If Table 5 already reflects the post-reseat run (it does), E3's "after" leg is done; remaining minimum is E0+E1+E2 (all wafer-free) to pin micro-mechanism and exclude confounds, plus E4 data pull.

Also note proximal cause of bridges: positive CD bias shrinking spaces; thickness loss co-symptom. The quantitative monotone relation defects vs CD bias across radial bands supports CD as proximal driver. Not necessary to distinguish for containment, since fix removes both.

Add brief "why not pattern density" explanation: outer-die clustering misread as density; actually radial.

Maybe include a one-line containment note: C2 should remain down for patterned product until radial qual added; W01-W06 disposition. Keep short.

I'll write the final answer with headers, evidence table maybe. Keep it rigorous and cite specific numbers. Aim ~1000-1300 words. Use markdown.
Let me work through this carefully. This is a synthetic case study asking me to act as a board-level R&D / Trade Secret Invention Strategy team. The task is to analyze a confidential package about HelioForge Systems' liquid-cooled AI compute modules.

Let me extract the key facts:

**Existing patents:**
- HF-101: Cold plate with parallel microchannels beneath accelerator package (hardware structure)
- HF-118: Rack manifold with redundant pumps and pressure sensors (rack-level hardware)
- HF-133: Controller that adjusts coolant flow as a function of measured package temperature (temperature-feedback control)

**Confidential R&D findings:**
- Exp 26-041: Derivative of GPU power demand predicts hotspot 180-260 ms before package temp sensor responds. This is a feed-forward signal discovery.
- Exp 26-052: Controller using workload power derivative + inlet-fluid compressibility estimate reduced 99.9th-percentile hotspot overshoot by 37% vs temperature-feedback control. Effect disappears when accumulator pre-charge changed by >12%. So there's a coupling between the estimator and accumulator pre-charge.
- Exp 26-061: Manufacturing found intentionally varying microchannel width across cold plate by ±6% counteracts package-level heat-flux asymmetry. CAD variation generated from each package's measured post-assembly thermal map, not fixed geometry. Per-unit adaptive geometry.

**Manufacturing:**
- Adaptive channel-width map generated after package attach. Proprietary script converts thermal-map coordinates into laser-machining offsets. Operators see only job ID; transformation coefficients on access-controlled build server. Yield improved 91.2% → 97.8%.
- Supplier receives final machining coordinates but not thermal map, algorithm, or coefficients. Contract prohibits reverse engineering and reuse of job files. Coordinates unique per unit, retained 30 days.

**Controls/firmware:**
- Temperature feedback: PID using package temp — public (HF-133)
- Power-derivative feed-forward: predictive term from accelerator telemetry — not public
- Compressibility estimator: online estimate from pressure-wave response — not public
- Accumulator pre-charge coupling: calibration table links estimator to pre-charge — not public
- Per-unit channel geometry: thermal-map-derived machining transformation — not public

**Commercial/reverse engineering:**
- Customers can measure external coolant pressure, inlet/outlet temps, app performance. Cannot inspect firmware (secure boot, signed updates). Destructive teardown reveals physical channel geometry, but inferring per-unit generation algorithm requires comparing multiple units with pre-machining thermal maps, which customers don't have.
- Competitor advertised "predictive liquid cooling using workload telemetry" — no algorithm disclosed. No known competitor advertises per-package adaptive cold-plate geometry.

**Internal status:**
- No invention disclosures filed for: feed-forward controller, compressibility estimator, accumulator interaction, per-unit geometry-generation method. Engineering treated as implementation details.
- Sales presentations disclose "predictive cooling" and "factory-calibrated cold plates" but not mechanisms.

Now let me structure the analysis per the six questions:

**1. Technical concepts materially distinct from existing patent claims:**

Existing claims cover:
- HF-101: static cold plate structure with parallel microchannels
- HF-118: rack manifold with redundant pumps/pressure sensors
- HF-133: temperature-feedback control (flow as function of measured package temperature)

New concepts:
a) Power-derivative feed-forward control — using derivative of GPU power demand as predictive signal (26-041). This is distinct from HF-133 because HF-133 claims flow adjustment as a function of *measured package temperature*; feed-forward from workload telemetry derivative is a different input signal and control architecture. The 180-260 ms prediction lead is the technical effect.

b) Compressibility estimator — online estimate of inlet-fluid compressibility from pressure-wave response. Not covered by any existing claim. HF-118 has pressure sensors but claims manifold redundancy, not estimation algorithms. Note: the pressure sensors claimed in HF-118 might be usable as the sensing hardware for the compressibility estimator — potential synergy but also potential argument that the sensor hardware alone is known.

c) Accumulator pre-charge coupling — calibration table linking estimator to pre-charge; effect disappears if pre-charge varies >12% (26-052). This is a dependency/interaction concept.

d) Per-unit adaptive microchannel geometry — varying channel width ±6% based on each package's measured post-assembly thermal map (26-061). Distinct from HF-101 which claims a fixed parallel-microchannel cold plate; this is a manufacturing method + per-unit customization. Note: careful — HF-101 claims "parallel microchannels" — does the ±6% width variation still fall within HF-101's claims? Possibly, but the claimed concept is the structure, not the method of generating per-unit geometry. The new concept is the method: measure thermal map post-assembly → transform → machine. Also the product: a cold plate whose channel geometry is a function of a specific package's thermal map.

Caveat per instruction: do not assume patentability merely because not covered. Need prior-art searches. E.g., model-predictive control with power feed-forward is known in other domains (HVAC, datacenter cooling literature); graded/variable microchannel widths are known in literature for hotspot mitigation; per-unit adaptive manufacturing exists in other fields. I must flag these.

**2. Patentable inventions vs. trade-secret know-how:**

Framework considerations:
- Detectability: can a competitor detect the invention from the sold product? If yes, patent (or at least defensive publication) is more valuable. If no, trade secret may be stronger.
- Reverse-engineering resistance: firmware secured (secure boot, signed updates); transformation coefficients on access-controlled server; supplier gets only coordinates.
- Patent trade-offs: disclosure vs 20-year exclusivity; enforcement ability (can you detect infringement?). For control algorithms running inside a competitor's product, infringement detection is hard unless the effect is observable (e.g., 37% overshoot reduction claim, teardown correlation).

Candidates for patenting:
- The combined control method (feed-forward + compressibility estimator + pre-charge coupling) with quantified technical effect (37% reduction in 99.9th percentile overshoot) — this is a concrete technical effect that could support patent eligibility/enablement. Also observable: a competitor advertising "predictive liquid cooling using workload telemetry" suggests market value and possible independent invention — first-to-file urgency. But infringement detection against a competitor's closed firmware is difficult — mitigate by drafting claims with externally observable correlates? Possibly claim the measurable overshoot reduction under defined transient workload? That's tricky. Flag for counsel.
- Per-unit adaptive geometry method: partially detectable via destructive teardown (geometry visible), but the algorithm isn't inferable without multiple units + pre-machining thermal maps. The *method* (measure map → transform → machine) may be patentable as a manufacturing method; product claims may be detectable in teardown. However per-unit uniqueness makes each unit different; a claim to "cold plate with channel widths varying as a function of the specific package's measured thermal map" could be detectable by comparing geometry to... hmm, customers don't have thermal maps. Detection of infringement would require comparing competitor units across units (geometry varies unit-to-unit) — that itself is observable: if every unit's channel pattern differs and correlates with package, that suggests the method. Actually unit-to-unit variation in channel geometry is observable by comparing two units. So infringement detection is possible (buy two units, compare). Interesting point.

Candidates for trade secret:
- Transformation coefficients and the proprietary script (know-how; kept on access-controlled server; operators see only job IDs) — classic trade secret treatment, already operationalized.
- The calibration table linking estimator to pre-charge — a lookup table/data is weak patent subject matter anyway; better as trade secret. But the *method of coupling* might be patentable.
- The specific pre-charge tolerance window (effect disappears beyond 12% change) — this is empirical know-how; could be claimed as a range in a patent (ranges are commonly claimed), but disclosing it teaches competitors. Strategic choice: claim broad ranges in patent, keep optimal narrow window and calibration data as secret. Classic "claim broad, keep the sweet spot secret" strategy — but note enablement/best-mode considerations (in the US, best mode requirement still exists though unenforceable post-AIA as a defense; flag for counsel).
- The 180-260 ms prediction lead window — could be claimed as a range.

Also consider defensive publication for things they don't want to patent but want to prevent competitors from patenting (e.g., if they decide the feed-forward alone is too broad/obvious, publish to create prior art). But careful: publication destroys trade secret and must be weighed.

**3. Combinations more defensible than individual features:**

a) Feed-forward + compressibility estimator + accumulator pre-charge coupling as a single control system: Experiment 26-052 shows the combination produces the 37% overshoot reduction, and the effect disappears when pre-charge deviates >12% — i.e., the combination, not the individual features, produces the technical effect. This interdependence is evidence against obviousness (synergy/unexpected results) — the whole is greater than the sum; the >12% sensitivity is evidence that the elements are interdependent. This supports both patentability (non-obviousness via unexpected results) and trade-secret value (harder to replicate without all elements).

b) Compressibility estimator + existing HF-118 pressure sensors: HF-118 already claims pressure sensors in the manifold; the estimator uses "pressure-wave response" — the existing hardware may support the new algorithm. A combination claim (manifold with pressure sensors + estimator) could tie into existing portfolio, or could be at risk of obviousness (using known sensors for their known outputs? but pressure-wave response for compressibility is a non-traditional use). Flag.

c) Per-unit geometry + feed-forward control: the thermal map used for machining could also characterize the package's heat-flux asymmetry to tune control per-unit. Not in document — flag as a potential unrecognized invention (question 5) but be careful not to invent facts. The document says the map is generated post-assembly; the control system could use per-unit calibration. This is a hypothesis for R&D, not documented fact — recommend an experiment.

d) Per-unit geometry method + supplier data-handling protocol: the protection of the algorithm relies on the split (supplier gets coordinates only). The combination of technical method + information-architecture is the trade-secret system.

e) The full stack: per-unit thermal characterization (26-061) → geometry → expected hotspot behavior → feed-forward tuning. The thermal map data generated per unit is itself an asset (dataset of thermal maps correlating to geometry and yield) — potential data trade secret / know-how.

**4. Exposure assessment:**

Reverse engineering:
- Physical channel geometry exposed via destructive teardown (Section 5). But per-unit generation algorithm protected because inferring requires multiple units + pre-machining thermal maps which customers lack. Residual risk: a competitor could buy many units, image packages pre-teardown? No — they lack pre-machining maps; but they could correlate measured geometry with package X-ray or thermal testing of the same unit before teardown? Actually they could thermally test the unit themselves (measure its hotspot map with external instrumentation?) — the document says customers can measure inlet/outlet temps and performance; a sophisticated competitor could run thermal tests, then teardown, and correlate geometry with their own measured thermal map of that unit. That partially reconstructs the mapping (though their thermal map is post-machining system-level, not the pre-machining package map). Flag as a residual risk; also note the ±6% variation pattern alone reveals the concept ("channels vary ±6%") even if the algorithm isn't inferable — the *idea* of adaptive per-unit geometry could be deduced from teardown of 2+ units showing differing geometries. Once the concept is known, a competitor could independently develop their own transformation — independent invention is a risk trade secrets don't protect against; that argues for patenting the method if patentable. This is a key strategic point: teardown reveals the *concept* (non-uniform, unit-varying geometry), so secrecy has a shelf life; patent protection is the durable moat for that one.

- Firmware: protected by secure boot + signed updates (Section 5). Residual risks: side-channel observation of behavior (pressure measurements at the rack could reveal control dynamics — customers CAN measure external coolant pressure; a competitor could characterize the controller's response to workload transients and infer the presence of feed-forward and possibly estimate parameters like lead time). The 180-260 ms lead could be observable externally by correlating accelerator telemetry (visible in software) with pressure/flow response. So the *existence and rough parameters* of the feed-forward may be behaviorally observable even if code isn't. That cuts both ways: (a) partial loss of secrecy over time; (b) helps detect infringement by others.

Disclosure risk:
- Sales presentations already disclose "predictive cooling" and "factory-calibrated cold plates" (Section 6). These are probably enabling-level? No — marketing statements of results, not mechanisms, likely non-enabling, but: (a) they may constitute prior art against later-filed patents if they disclose the invention; currently they seem non-enabling, but counsel should review exact wording; (b) they put competitors on notice of what to develop (competitor already advertising predictive cooling using workload telemetry — Section 5 — suggests the direction is known in the market). Risk of independent invention/discovery is real.
- Public disclosure/grace period: need to check whether any disclosure was enabling and when — US has 1-year grace period (AIA 102(b)(1)); most foreign jurisdictions have absolute novelty. If sales presentations predate filing and are enabling, foreign rights may be lost. Counsel must review dates and content.

Supplier access:
- Supplier receives final machining coordinates, retained 30 days; contract prohibits reverse engineering and reuse (Section 3). Risks: coordinate streams over many units could be statistically analyzed to infer the transformation structure (supplier sees many jobs; even without thermal maps, pattern analysis across coordinates + knowledge of cold-plate blank could reveal ±6% variation scheme and correlations). 30-day retention limits accumulation? Actually retention by supplier for 30 days means they have rolling window; contract should also address derived data, aggregation, subcontractors. Also: does supplier do the laser machining? Presumably yes ("laser-machining offsets"). Supplier is a potential leak vector; mitigations in place (no algorithm, no coefficients, job IDs only, contract terms). Additional mitigations: audit rights, data-deletion certification after 30 days, prohibitions on derivative works, limit retention, encrypt job files, possibly bring machining in-house for highest-value units, or split suppliers. Also ensure NDA/trade-secret notices; marking.
- Reasonable-measures requirement: trade-secret protection requires reasonable secrecy measures (DTSA/UTSA). Document existing measures: access-controlled build server, job IDs, operator need-to-know, secure boot, signed updates, supplier contract terms. Gaps: no mention of employee NDAs, invention assignment agreements, exit procedures, lab notebook policies, visitor controls, code access logging, firmware threat model. Recommend formalizing — flag for trade-secret counsel.

Accidental disclosure:
- Engineering treats the four items as "implementation details" (Section 6) — cultural risk: engineers may present at conferences, publish, or discuss with suppliers/customers. Need invention-disclosure process and publication review.
- No invention disclosures filed at all — process gap.
- The competitor's announcement increases urgency: first-to-file system; also risk competitor patents the feed-forward concept and blocks HelioForge (their own prior use may give some defense in US — prior-user rights under AIA 35 U.S.C. 273 for commercial use — but limited; flag for counsel). Document prior invention dates now (notebooks 26-041, 26-052, 26-061) — these dated lab notebooks are critical evidence; ensure they're witnessed/signed, preserved.

**5. Inventions management may not realize:**

a) The coupling/discovery that the compressibility estimator's effectiveness is contingent on accumulator pre-charge within ~12% — this interdependency itself is an invention (a control system that co-manages estimator and pre-charge, or a calibration method). Engineers treated as implementation detail; the sensitivity finding (26-052) is exactly the kind of non-obvious interaction that supports patentability.

b) The transformation from thermal-map coordinates to laser-machining offsets (the script + coefficients) — a manufacturing method invention, not just a script. Possibly patentable as a method of manufacturing a cold plate customized to a specific package; also the "closed-loop manufacturing" concept: post-assembly measurement drives subtractive machining. 

c) The per-unit data asset: database of thermal maps → geometries → yield/performance outcomes; potential future ML training data; trade secret as a dataset.

d) The online compressibility estimation via pressure-wave response — using existing pressure sensors (HF-118 hardware) for a new measurement function (virtual sensor). "Virtual sensor" inventions are commonly patentable; also synergizes with portfolio.

e) The use of power-derivative as a leading indicator with a defined 180-260 ms lead window — claiming the predictive window and the derivative-based trigger.

f) Possibly: combination of per-unit geometry + per-unit control calibration (hypothesis, needs experiment — recommend as diligence).

g) The behavior of the control system as a diagnostic: e.g., using pressure-wave response for anomaly detection (pump degradation, air ingestion). Not in document — only mention as R&D direction, clearly labeled as new idea beyond the package? The prompt says "using only this confidential package" — I should be careful. I can recommend it as a diligence experiment direction but must label it as beyond the documented evidence. Better to stay conservative: mention as optional exploratory item derived from the estimator's capability (pressure-wave response is documented), framed as a question for engineering.

**6. Recommendations:**

Immediate (30 days):
- File invention disclosures for the four identified items + the coupling discovery; institute disclosure triage.
- Preserve and witness lab notebooks (26-041, 26-052, 26-061); establish dated, countersigned records; export control check? Not needed.
- Confidentiality audit of sales/marketing materials; freeze external technical disclosures pending review; publication/presentation review board.
- Prior-art searches (outside counsel/patent searcher) on: feed-forward/model-predictive cooling using workload/derivative telemetry; fluid compressibility estimation; accumulator pre-charge control coupling; variable/adaptive microchannel geometries; per-unit thermal-map-driven machining. Also search the competitor's filings and publications.
- Patent counsel: freedom-to-operate is a separate question (not asked directly but relevant before financing); also assess whether HF-133 claims dominate the new controller (does practicing feed-forward still infringe/fall under HF-133? own patents don't give right to practice improvement... it's their own patent so fine; but whether HF-133's claims read on the new system matters for claim drafting to avoid obviousness-type double patenting? Flag for counsel.)
- Trade-secret counsel: reasonable-measures audit; NDA/assignment audit; supplier contract gap review (aggregation, derived data, retention enforcement, audit rights).
- Accelerate decision on patent vs. secret for each item before any further marketing disclosure; consider provisional applications to lock dates cheaply, especially given competitor activity and financing (investors like pending applications).
- Consider defensive publication for deprioritized items.

Diligence experiments:
- Quantify robustness: repeat 26-052 across units/coolant batches/temperature ranges; document the 12% pre-charge boundary precisely (claim range drafting; trade-secret boundary).
- Determine which elements of the control stack are behaviorally observable from customer-accessible signals (pressure, temps, telemetry) — an internal "red-team reverse engineering" exercise to measure information leakage; informs patent-vs-secret decision.
- Teardown red-team: buy own units, have an independent team attempt to infer the transformation from multiple units' geometries; quantify how many units needed; informs secrecy shelf-life.
- Test per-unit control calibration using the machining thermal map (hypothesis from 3c).
- Reproducibility/statistics for the 37% claim (n, confidence intervals) — needed both for patent enablement and for marketing substantiation (avoid false-advertising risk on the 37% and yield claims; flag).
- Prior-art-informed design-around documentation: keep records of independent development.
- Check whether supplier coordinates + publicly known blank geometry allow reconstruction; experiment: give a cleared third party only the supplier-visible data and see what they can infer (supplier-leak red team).

Documentation steps:
- Invention disclosure forms; inventor interviews (L. Chen et al.); inventorship determination by counsel (inventorship is legal, not managerial).
- Trade-secret inventory with classification, access logs, marking ("Confidential"), need-to-know matrix.
- Data-retention/deletion policy for supplier; contractual audit.
- Board-level IP strategy memo for financing: map of patents/pending/secrets, risk register.
- Employment agreements: invention assignment, confidentiality, exit interviews; ensure L. Chen and team signed (flag to counsel).
- Establish grace-period clock tracking: log all external disclosures with dates and content.

Also: the financing context — board wants integrated strategy before next financing; recommend an IP audit deliverable, chain-of-title check (assignments from employees/contractors), and confirm no open-source contamination in firmware? Not in document — skip or briefly mention chain-of-title.

Structure of the answer: The user asked for six numbered deliverables with citations and explicit flags for outside analysis. I should use headers matching the six questions, plus an executive summary and a consolidated "outside expertise required" section. Use tables where helpful (e.g., concept-by-concept disposition matrix). Cite evidence by section/experiment number (e.g., "Exp. 26-052"; "§3 Manufacturing notes"; "§5").

Format considerations: This is a complex, professional deliverable. Long-form with headers, tables, and bullets is appropriate. Should be thorough but not bloated. I'll aim for a well-organized report: maybe 1,500-2,200 words. Tables for the disposition matrix. Every material conclusion needs a citation to the package. Explicit flags where outside analysis is needed. Also the standing caveat: not assuming patentability from non-coverage; note that "materially distinct from claims" ≠ patentable; prior art may anticipate (competitor already in field with "predictive liquid cooling using workload telemetry" — their advertising could be prior art if enabling, or evidence the field is active).

Also note: HF-101's claim to "parallel microchannels" — does ±6% width variation keep them "parallel"? Width variation across channels (channel-to-channel widths differ by ±6%) — channels could still be geometrically parallel but with varying widths. The document says "varying microchannel width across the cold plate by ±6%". So HF-101 might literally still read on the plate (parallel channels), meaning the new plates may be within HF-101's claims — which is fine (own patent) but means HF-101 doesn't capture the *adaptive* aspect. The novel concept is per-unit map-driven variation. Also possible design consideration: HF-101 doesn't help stop competitors from doing adaptive geometry; need new claims.

One more nuance: Experiment 26-052 says effect disappeared when accumulator pre-charge changed by more than 12% — evidence of interdependence (synergy), useful for non-obviousness (unexpected results) and for arguing the combination claim is not mere aggregation. Also useful for trade secret: without knowing the coupling, a competitor copying feed-forward may fail to reproduce results — a natural moat. But if patented, the patent disclosure must enable; can claim the coupling while keeping exact calibration table secret (best-mode considerations — flag to counsel).

Also the accumulator itself: is it claimed in HF-118? HF-118 claims manifold with redundant pumps and pressure sensors — no accumulator mentioned. So accumulator + coupling is open space (subject to prior art).

Potential patentable subject matter list (with detectability and flags):
1. Feed-forward control using dP/dt of accelerator workload with predictive lead window (claims: method; system; controller configured to...). Detectability: behavioral (partially), not code-observable in competitor products — enforcement challenge. Decision: patent if prior art allows, because competitor already in the space and independent invention risk high; combine with 2 & 3.
2. Online compressibility estimation from pressure-wave response (virtual sensor). Detectability: low externally; but pressure sensors are customer-measurable... the estimation happens in firmware. Keep as patent candidate combined with 1; some elements secret.
3. Pre-charge coupling/calibration method (estimator ↔ accumulator pre-charge). Detectability: very low. Could be kept secret entirely OR claimed as combination. Strategy: claim the combination broadly; keep calibration table/tolerances secret. Counsel to advise on enablement/best mode.
4. Per-unit adaptive geometry manufacturing method (measure post-assembly thermal map → transform → machine). Detectability: concept detectable via teardown of multiple units; algorithm not. Decision: patent the method + product-by-process aspects if prior art allows, because secrecy shelf-life limited and no competitor in space yet (first-mover patent opportunity; §5: "No known competitor advertises per-package adaptive cold-plate geometry").
5. The transformation script/coefficients — trade secret (pure know-how; copyright weak; keep secret).
6. The thermal-map→geometry dataset — trade secret (data asset).
7. The 37%/12%/180-260 ms/±6% parameters — some claimed as ranges (defensive breadth), optimal values secret.

Trade-secret program gaps to flag: no mention of NDAs, access logging beyond build server, notebook witnessing, disclosure triage — recommend.

Risk register:
- Competitor independent invention of feed-forward (already advertising) → risk to secrecy AND risk they file first → urgency to file provisional + document prior use dates (26-041, 26-052).
- Supplier aggregation risk.
- Teardown reveals adaptive geometry concept.
- Marketing disclosures could become enabling prior art or tip off competitors; also the "predictive cooling" claims substantiation.
- Key-person risk (L. Chen) — knowledge concentration; capture know-how; retention.
- Grace-period/absolute novelty risk if any enabling disclosure occurred (dates unknown → counsel to investigate).

Outside expertise flags (explicit):
- Patentability requires prior-art search (outside patent searcher/counsel): MPC/feed-forward cooling, dP/dt telemetry control, compressibility estimation, accumulator coupling, adaptive microchannels, per-unit laser machining from thermal maps.
- Patent counsel: claim drafting to capture observable correlates for enforceability; continuation strategy; foreign filing decisions given absolute novelty; inventorship determination; best-mode/enablement strategy for keeping calibration data secret; possible design-around of HF-101 (does HF-101 read on varied-width plates? double-check claim scope vs. new products — actually it's their own patent so no infringement issue, but scope matters for whether HF-101 protects the new plates at all).
- Trade-secret counsel: reasonable-measures audit; supplier contract terms; DTSA/UTSA compliance; employee agreements; litigation readiness (identification of secrets with particularity).
- Prior-user rights analysis (AIA §273) — commercial use defense; document dates.
- Marketing counsel: substantiation for 37%/yield claims; review of sales decks for enabling disclosure.
- Possibly export control if selling internationally? Out of scope; skip or one line.

I should also flag the financing relevance: investors will diligence IP; having invention disclosures filed, provisionals pending, and a documented trade-secret program materially de-risks.

Now, the output format. I'll produce a structured report:

- Header / framing line (one or two sentences, noting it's based solely on the package and that nothing here is legal advice / patentability conclusions require outside work).
- Executive summary (short bullet list of the big calls).
- 1. Concepts materially distinct from existing claims (table or list with evidence citations and the caveat).
- 2. Patent vs. trade secret disposition (table: concept | evidence | detectability | proposed disposition | rationale).
- 3. Combinations (with evidence; the 26-052 interdependence is the centerpiece).
- 4. Exposure assessment (reverse engineering, disclosure, supplier, program gaps) with citations.
- 5. Unrecognized inventions (list with evidence; label hypotheses clearly).
- 6. Recommendations: phased (immediate 0-30 days; 30-90; ongoing), covering experiments, documentation, disclosures, protection decisions.
- Outside analysis required (consolidated list).
- Caveats/limitations (using only the package; no prior-art conclusions; etc.)

Keep citations tight: (Exp. 26-041), (Exp. 26-052), (Exp. 26-061), (§1 HF-133), (§3), (§4 table), (§5), (§6).

Length: aim for comprehensive but readable. Given six explicit questions with evidence-citation requirements, this will be long. I'll use tables to compress. Maybe ~1,800-2,400 words. Markdown is appropriate (this will render). Headers, tables, bullets. Avoid excessive nesting.

Let me draft.

Key analytical points to make sure I include:

Q1 distinctions:
- Feed-forward from power derivative vs. HF-133's temperature feedback: different measured variable (workload telemetry derivative, not package temperature), anticipatory rather than reactive; evidence: 26-041 (180-260 ms lead), §4 (feed-forward row "No" public).
- Compressibility estimator: no existing claim covers estimation of fluid properties from pressure-wave response; HF-118 claims sensors as hardware in redundant-pump manifold, not estimation methods (§1).
- Accumulator/pre-charge coupling: no existing claim mentions accumulator at all (§1, §4).
- Per-unit adaptive geometry: HF-101 claims a cold plate with parallel microchannels — a fixed structure; the new concept is a manufacturing method driven by each unit's measured thermal map, and a product whose geometry is unit-specific (26-061, §3). Caveat: HF-101 may literally read on the new plates (parallel channels of varying width) — scope question for counsel.
- Caveat paragraph: distinctness ≠ patentability; competitor already advertises predictive cooling with workload telemetry (§5) → field active; prior-art search required for each.

Q2 dispositions:
Patent candidates (if search supports):
- Combined predictive control system (feed-forward + estimator + coupling) — rationale: quantified effect (37%, 26-052), competitor activity (§5) creates independent-invention and first-to-file risk, behavioral observability aids enforcement but also erodes secrecy → time-limited secrecy anyway.
- Per-unit adaptive geometry method (+ product claims) — rationale: concept exposed by teardown (§5), no competitor in space (§5), yield/performance benefits documented (26-061, §3 91.2→97.8%).
Trade-secret candidates:
- Transformation script + coefficients (§3) — already access-controlled, low detectability, no enforceability upside from patent since undetectable in competitor products... actually manufacturing method patents are hard to police; keep secret. But note supplier exposure → strengthen contracts.
- Calibration table (estimator↔pre-charge) (§4) — data/table; low detectability.
- Optimal parameter values within claimed ranges (12% window, 180-260 ms, ±6%) — claim ranges, keep optima secret; counsel flag re enablement/best mode.
- Thermal-map/geometry/yield dataset (§3) — data asset.
Also mention defensive publication option for leftovers.

Q3 combinations:
- The control triad (26-052): interdependence proven by >12% sensitivity — synergy evidence; combination claim stronger against obviousness; also a natural trade-secret moat.
- Estimator + HF-118 pressure sensors: reuse of claimed hardware; portfolio layering; also obviousness risk flag.
- Geometry method + control calibration per unit (hypothesis — label as such; recommend experiment).
- Method + information architecture (supplier split) as protection system (§3).
- Whole-stack "per-unit digital thread": thermal map → geometry → control tuning → fleet data (partly hypothesis).

Q4 exposure:
- Teardown exposes geometry; multi-unit comparison exposes the *concept* of per-unit variation though not the algorithm (§5: inferring algorithm requires multiple units + pre-machining maps customers lack). Shelf-life argument → patent.
- Behavioral observability of control: customers can measure pressure/temps/performance (§5) → control dynamics partially inferable; 180-260 ms lead observable via telemetry correlation (26-041 + §5). Firmware itself protected (secure boot/signed updates, §5).
- Sales disclosures (§6): "predictive cooling" + "factory-calibrated cold plates" — non-enabling on their face but flag for counsel to confirm; market signaling already triggered competitor response (§5).
- Supplier: coordinates-only, 30-day retention, contract prohibitions (§3) — good, but aggregation/statistical inference risk; no audit/deletion certification mentioned; subcontractor flow-downs unknown → counsel.
- Program gaps: no invention disclosures at all (§6); engineering culture treats as implementation details (§6); no mention of NDAs, access logs beyond build server, notebook witnessing, exit procedures → trade-secret counsel; reasonable-measures documentation needed for DTSA/UTSA.
- Key-person concentration: L. Chen (26-041).
- Competitor first-to-file risk on feed-forward (§5); document invention dates (26-041/26-052) for prior-user rights / derivation — counsel.

Q5 unrecognized:
- The coupling/sensitivity discovery itself (26-052 >12%) as an independent invention (co-control of pre-charge; calibration method; possibly a method of operating accumulator within tolerance band).
- The transformation method as a patentable manufacturing method, not just a script (§3; 26-061).
- Virtual-sensor framing of compressibility estimator (§4).
- The per-unit data asset (§3).
- The predictive lead window as a claimed range (26-041).
- Hypothesis: per-unit control calibration from machining map (recommend experiment; no evidence in package).
- Possibly yield improvement itself as evidence of technical effect for the manufacturing method claims (§3).

Q6 recommendations — phase them:
Immediate (0–30 days):
1. File internal invention disclosures for the five items (feed-forward, estimator, coupling, geometry method, transformation/coefficients as secret asset) (§6 gap).
2. Commission prior-art searches across five domains; include competitor patent filings and publications (§5).
3. Patent counsel engagement: provisional strategy for the two patent candidates; foreign-filing/absolute-novelty review of past disclosures (§6 sales decks — dates needed).
4. Freeze/ga control external technical disclosures; review sales decks (§6).
5. Preserve evidence: witness/date notebooks 26-041/052/061; secure version history of script and calibration tables (§3).
6. Trade-secret counsel: reasonable-measures audit; supplier contract review (aggregation, deletion certification, audit rights, flow-downs) (§3).
30–90 days:
7. Red-team experiments: behavioral inference from customer-accessible signals (§5); multi-unit teardown inference (§5); supplier-data inference from coordinates-only (§3).
8. Robustness experiments: replicate 26-052 across units/coolants/ambient; map the 12% boundary precisely; statistics for 37% claim (26-052) — supports enablement and marketing substantiation.
9. Test the per-unit control-calibration hypothesis (26-061 map reuse).
10. Decide patent vs. secret per item using detectability findings; file provisionals/PCT decisions; defensive publications for rejects.
11. Implement IP hygiene: NDAs/assignments check (incl. L. Chen), access logs, marking, invention-disclosure process, publication review board, exit interviews.
Ongoing/board level:
12. IP audit package for financing: portfolio map (patents/pending/secrets), risk register, chain-of-title verification.
13. Quarterly disclosure triage; competitive monitoring of the "predictive cooling" competitor (§5).
14. Key-person risk mitigation (26-041 L. Chen).

Outside analysis required (consolidated):
- Patentability/prior art: everything in Q1 — state clearly we cannot conclude patentability; searches needed in listed domains; competitor's advertisement (§5) may or may not be prior art depending on enablement/date — counsel.
- Claim scope of HF-101 vs. varied-width plates (counsel claim construction).
- Enforceability/detectability claim drafting (counsel).
- Best mode/enablement if keeping calibration table secret while patenting combination (counsel).
- Absolute novelty/foreign rights re sales disclosures (counsel; need dates + full text of decks).
- Trade-secret reasonable measures, DTSA/UTSA, supplier contract, NDAs/assignments (trade-secret counsel).
- Prior-user rights documentation (counsel).
- Marketing substantiation (37%, yield) — advertising counsel (26-052, §3).
- Inventorship determination (counsel) — esp. L. Chen and manufacturing authors.

Also add the standing instruction compliance: "Do not assume patentable merely because not covered" — I'll include explicit caveats in Q1 and the disposition table (column: "distinct from claims ≠ patentable; search required").

I want to keep every material conclusion tied to a citation. Let me now write the final response. I'll aim for clarity and board-appropriate tone. Use a few tables. Keep headers aligned to the six questions plus summary and outside-expertise section.

One more check on details I should not get wrong:
- 99.9th-percentile hotspot overshoot reduced by 37% (26-052). ✓
- Effect disappears when accumulator pre-charge changed by >12% (26-052). ✓
- ±6% width variation (26-061). ✓
- Yield 91.2% → 97.8% (§3). ✓
- 180–260 ms lead (26-041). ✓
- Supplier retains coordinates 30 days; contract prohibits reverse engineering/reuse (§3). ✓
- Secure boot + signed updates (§5). ✓
- Teardown reveals geometry; algorithm inference needs multiple units + pre-machining maps (§5). ✓
- Competitor advertises "predictive liquid cooling using workload telemetry," no algorithm (§5). ✓
- No competitor advertises per-package adaptive geometry (§5). ✓
- Sales decks: "predictive cooling," "factory-calibrated cold plates" (§6). ✓
- No invention disclosures filed; treated as implementation details (§6). ✓
- HF-101/118/133 claims as listed (§1). ✓
- Operators see only job identifier; coefficients on access-controlled build server (§3). ✓
- Customers can measure external coolant pressure, inlet/outlet temp, app performance (§5). ✓

Also note §5: "pressure sensors" in HF-118 — the compressibility estimator uses "pressure-wave response" (§4) — likely using existing sensors; synergy point stands, but I shouldn't overstate that it uses HF-118's sensors specifically; say "may leverage the pressure-sensor hardware already claimed in HF-118 (§1, §4) — confirm with engineering."

I'll add a short "limits of this analysis" section: based solely on the package; no legal advice; patentability, FTO, and trade-secret enforceability require licensed counsel; facts like dates of disclosures, employment agreements, and supplier identity unknown.

Format: Markdown with headers, tables where they compress information, bold key terms sparingly. Length target ~2,000 words; it's a big deliverable but the prompt demands six sections with citations, so it's justified. I'll be disciplined: tight bullets, no repetition between sections.
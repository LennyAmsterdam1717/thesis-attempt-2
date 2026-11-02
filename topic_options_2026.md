# Thesis topic options (2026 literature): ML on stock and company networks

Anchor paper: **Krstev, Rigoni, Mishkovski & Pasa (2026)**, "A Temporal Graph Learning Framework for Lead-Lag Detection in
Financial Markets", *Machine Learning* 115:173 (see `NOTES.md` Phase 0). Every option states how it **extends**,
**transfers** or **differs** from the anchor. Literature and crowding evidence: `lit_review_2026.md`. Citations use the
entry codes there (A1…D7, C0 = anchor). Legend: [FT] / [ABS] / [UNVERIFIED] / [BACKGROUND].

<!-- PHASE 4 SUMMARY GOES HERE -->

---

## Part 1: Raw ideas and kill tests

Every idea got at least one kill-test search (queries logged in `NOTES.md` under "Phase 3 kill tests").
Outcome: **survived** (no 2026 work does it) · **weakened** (close work exists, a specific angle remains) · **killed**.

| # | Idea | Dir. | Kill-test outcome | Closest work found (and why it does / doesn't kill) |
|---|---|---|---|---|
| 1 | **Signed temporal lead-lag link prediction**: predict existence *and* sign (same vs opposite direction) of residualised lead-lag edges | C | **Weakened → survives** | MSGNN / FiLL (He et al. 2022, [BACKGROUND]) built *static yearly* signed directed lead-lag graphs for link/sign prediction. Anchor (C0) is temporal but unsigned; Regier et al. 2026 (C1) is temporal signed but on Bitcoin trust networks. No 2026 temporal + signed + stock work found. |
| 2 | **Do economic links explain price lead-lag?** Typed edges (supply chain, common ownership, sector, co-mention) as context for lead-lag LP | C/A | **Survives** | Yılkı 2026 (C3) uses a 51-edge supply KG for return factors (no LP); Köse et al. 2026 (C2) build typed firm networks (no LP); Cartea et al. 2026 JEF [UNVERIFIED] is statistical lead-lag; a non-academic GitHub study (QuantIntellect, Oct 2026) finds customer-momentum drift *not* supported. A useful prior: "supply links ≠ automatic lead-lag". |
| 3 | **Re-evaluating the anchor**: is GraphMixer learning lead-lag or "who is volatile"? EdgeBank, time-decay PPR, degree/volatility baselines, all-entity ranking, larger universe | D/C | **Survives** | Generic 2026 critiques (C4 Yu & Ha; C5 Fei et al.; [UNVERIFIED] 2609.35364, 2610.33171) on Wiki/Reddit/MOOC; none on financial graphs. Search `EdgeBank … stock lead-lag re-evaluation` found nothing financial. |
| 4 | **Surprise edges**: edge-level anomaly = lead-lag edges a calibrated link predictor did not expect; validated on public event calendars | B | **Survives** | B1 (Caputi & Meadows) and B2 (ORCA) are graph-level; BAG (B3) and SimpleCount (B4) have no finance data; 2024 TGN financial anomaly paper (Kim et al., arXiv 2404.00060, [BACKGROUND]) is transaction fraud. |
| 5 | **Who buys next?** Classify institutional entry / exit / increase / decrease events on a temporal heterogeneous 13F manager–stock graph, with price co-movement edges as a second relation | A | **Weakened → survives** | A2 (Izadifar & Rahmati 2026) predicts *weights* (levels), and persistence nearly wins; Frontiers in AI 2026 (doi:10.3389/frai.2026.1722121, [ABS via search extract]) forecasts *aggregate* fund–stock network indicators (Chinese data) with LLMs. Entry/exit *event* classification on the graph was not found. |
| 6 | **Crowding anomalies in 13F ownership graphs**: coordinated institutional moves as dynamic-graph anomalies, validated against later crash risk | B | **Weakened** | GeomHerd (arXiv 2605.11645, 2026, [ABS via search extract]) has a 13F fund-overlap graph in an appendix that recovers stress quarters qualitatively; graph crowding signals exist (arXiv 2306.08105, 2023, background). An ML anomaly-detection + rigorous evaluation angle remains. |
| 7 | **Contagion exposure after credit events**: which linked listed firms (customers, suppliers, co-owned peers) suffer abnormal returns after an 8-K Item 1.03 bankruptcy filing; heterogeneous temporal graph + counterfactual explanations | A (+credit) | **Survives** | CausalGraphX (A5) is synthetic; DefaultGNN (D5) and guarantee-network HGNN [UNVERIFIED] use private data; 2026 TGNN counterfactual explainers (TemGX, ICLR 2026; MemExplainer, ICML 2026, [ABS]) are not applied to finance. Search `8-K bankruptcy … contagion … graph EDGAR 2026` found no ML paper. |
| 8 | **Tariff/sanction windows as stress tests** for learned company graphs (does a learned lead-lag/relation graph break during the April 2025 tariff shock?) | D | **Survives (thin)** | D6 (RMT complexity gap) and D7 (mediation) study the 2025 tariff episode without graph ML. Best used as an evaluation chapter inside idea 1 or 4, not as a standalone thesis. |
| 9 | **Point-in-time LLM-built company graphs**: how extraction choices and LLM look-ahead bias change downstream graph-ML results | D | **Weakened** | Look-ahead work for LLM *signals* exists (D1–D3, DatedGPT [UNVERIFIED]); A3 and C2 build graphs without point-in-time checks. Risky: needs heavy LLM extraction work. |
| 10 | **Explainable lead-lag for analysts**: apply TemGX / CoDy-style counterfactual explanations to the anchor's predictions | D/C | **Weakened** | General TGNN XAI methods exist in 2026 (TemGX, MemExplainer, QIEA-TGX AAAI 2026, [ABS]); none on finance. Too thin alone; **folded into ideas 2 and 7** as the XAI component. |
| 11 | Plain heterogeneous distress / fraud node classification on company graphs | A | **Killed** | Crowded: ≥ 10 2026 papers seen (A4, A8, unverified ESWA/Information Sciences items). Matches the supervisor's "classification is done". |
| 12 | New TGNN architecture for stock return prediction | – | **Killed** | Outside the gates (plain return prediction); crowded (A6, BAREKENG 2026, IEEE Access 2026). |
| 13 | Crypto pump-and-dump detection with ST-GNNs | B | **Killed (scope)** | B6 already does it; crypto-only and out of scope. Equity manipulation labels are not public. |

Ideas 1, 2, 4, 5, 6 and 7 become the six worked cards (two per direction A/B/C). Ideas 3, 8 and 9 become the short runner-up
cards in Part 3. Idea 3 is recommended as **work package 1** of the C cards.

---

## Part 2: Worked proposals (card template from the briefing, §11; scores per §9)

Scoring weights: novelty ×3, supervisor fit ×2, data ×2, feasibility ×2, evaluation clarity ×1, AI relevance ×1,
artefact ×1, industry ×1, Frank's interests ×1. **Maximum 70.** All gates (public data; not plain classification/return
prediction; ≥ 1 concrete 2026 closest work; ~30-week feasible) are checked per card.

---

### Card A-1: "Who buys next?" Classifying institutional position changes on a temporal heterogeneous ownership graph

**1. Pitch.** Instead of forecasting portfolio *levels* (nearly trivial, because portfolios persist), predict the *events*
(entry, exit, large increase, large decrease) on a quarterly manager–stock graph from SEC 13F filings, with stock–stock
co-movement/lead-lag edges as a second relation.

**2. Research question.** Can a temporal heterogeneous graph model of institutional ownership predict next-quarter
position-change events better than persistence, memorisation and tabular baselines?
- SQ1: Does adding a stock–stock relation (rolling co-movement or anchor-style lead-lag edges) improve prediction, i.e. is the heterogeneity worth it?
- SQ2: Which managers and stocks are predictable (by manager size, turnover, style; stock size, sector)?
- SQ3 (light economic check): do predicted entries/exits relate to next-quarter abnormal returns or volatility?

**3. Supervisor directions hit.** Heterogeneous graph (non-standard classification: edge *events*) · temporal · link prediction
(entry = edge birth, exit = edge death) · time series + graph.

**4. Closest 2026 work and the difference.**
- A2 Izadifar & Rahmati 2026 [FT]: same 13F bipartite graph, but predicts weights (node affinity); persistence/EMA nearly win.
  → We predict **changes** (classification of edge events), which their result shows is where the open problem is.
- Frontiers in AI 2026 (fund–stock graphs, LLM forecasting) [ABS]: aggregate network indicators, Chinese funds. → We predict at the manager–stock level, on US public data.
- GeomHerd 2026 [ABS]: 13F overlap graph used qualitatively for herding. → We use a supervised, evaluated prediction task.
- **Anchor**: *transfer*. Reuses the anchor's TGNN link-prediction framing and model zoo (TGN, TGAT, GraphMixer) on ownership edges,
  and tests the anchor's own lead-lag edges as an extra relation. It *differs* in being heterogeneous and event-typed.

**5. Kill test.** Queries: `13F institutional holdings entry exit prediction graph neural network temporal bipartite 2026`;
`institutional herding crowding detection anomaly 13F ownership network graph machine learning 2026`. Found only A2, the
Frontiers paper and GeomHerd (above). No entry/exit event classification on a graph. **Weakened → survives.**

**6. Data and access risk.** SEC EDGAR 13F-HR information tables (free; quarterly since 2013 in XML; SEC's structured "Form 13F
data sets" exist per secondary sources, **[UNVERIFIED]** exact coverage). CUSIP → ticker mapping via SEC's quarterly
"Official List of Section 13(f) Securities" **[UNVERIFIED]** plus fuzzy matching as in A2. Prices from Stooq/yfinance.
**Risk: medium** (identifier mapping and filing amendments are the main effort; A2's code shows it is doable).

**7. Method sketch.** Quarterly snapshots 2013–2026 with manager→stock edges weighted by portfolio weight; stock→stock edges from
rolling residual co-movement. Event labels per (manager, stock, quarter): entry / exit / up > 25 % / down > 25 % / no change.
Models: (i) TGB/DyGLib temporal models (TGN, DyGFormer) on the bipartite stream; (ii) heterogeneous snapshot model
(PyG `HeteroConv` + GRU, or HGT); (iii) LightGBM on hand-made graph features (strong tabular baseline). Code base: A2's repo
(<https://github.com/e-izdfr/portfolio-holdings-prediction>), TGB, PyG.

**8. Baselines and evaluation.** Persistence, EMA, EdgeBank, stock popularity (entry rate), manager-history frequency, LightGBM.
Metrics: AUPRC per event class (imbalanced), Recall@k / MRR for "which stocks will manager *m* enter". **Chronological split by
filing date** (13F is filed up to 45 days after quarter end; features may only use filings already public at prediction time);
the last 8 quarters are the test set.

**9. Artefact.** "Manager radar": per manager, predicted entries/exits with confidence; per stock, predicted net institutional flow;
attention or attribution showing which co-holders and correlated stocks drove the prediction.

**10. Feasibility (30 weeks), risks, fallback.** Weeks 1–6 review + proposal; 7–12 13F pipeline and mapping (reuse A2); 13–19
baselines + temporal models; 20–24 heterogeneity ablation (SQ1) + SQ2/SQ3; 25–30 writing. Risks: identifier mapping; few
time steps (~50 quarters); strong persistence. **Fallback:** binary entry prediction on S&P 500 stocks for the top-500 managers by AUM.

**11. Ethics.** Most 13F filers are firms (legal persons). Some are individuals or family offices, i.e. natural persons, so GDPR
applies to those: exclude or pseudonymise filers that are natural persons. No investment advice; aggregate reporting.

**12. Scores (§9).** Novelty 4 (×3 = 12) · fit 4 (8) · data 4 (8) · feasibility 4 (8) · evaluation 4 · AI relevance 4 · artefact 4 ·
industry 4 · Frank's interests 3 → **55 / 70**. Gates: all pass.

**13. Questions for the supervisor.** Is an *ownership* network acceptable as "stock network" for this thesis? Is classifying edge
events (entry/exit) "non-standard" enough? Does UU give MSc students WRDS access (Thomson/Refinitiv s34 13F would remove the mapping risk)?

---

### Card A-2: Contagion exposure after credit events, explained counterfactually

**1. Pitch.** When a listed firm files for bankruptcy (SEC 8-K Item 1.03), which of its connected listed firms (customers,
suppliers, co-owned peers, lead-lag followers) will suffer abnormal stock losses, and *why*? This is heterogeneous-graph
classification conditioned on an event, with counterfactual explanations as a core contribution.

**2. Research question.** Can a temporal heterogeneous company graph built from public filings classify which linked firms
are exposed to a credit event, and can counterfactual explanations identify the relation that carries the exposure?
- SQ1: Which relation types (supply chain, common ownership, sector, price lead-lag) carry equity contagion (relation ablation)?
- SQ2: Are counterfactual explanations faithful and stable (fidelity, sparsity, agreement across explainers), and do they match the economic channel?
- SQ3: Does the model beat event-study heuristics (same industry, "largest customer" flag, past correlation)?

**3. Supervisor directions hit.** **Credit default → stock market** (her separate problem) · heterogeneous graph with a non-standard
classification angle (event-conditioned + XAI) · temporal · lead-lag edges as one relation.

**4. Closest 2026 work and the difference.**
- A5 CausalGraphX 2026 [ABS]: counterfactual GNN for cascading defaults, **synthetic data only**. → We use real public data and equity outcomes.
- D5 DefaultGNN 2026 [ABS]: supply-network default prediction from **private** tax invoices. → We predict *equity contagion after* a default, on public data.
- A4 Ind_HGF 2026 [ABS]: distress node classification. → We do not predict the default itself, only who is hit by it.
- TemGX (ICLR 2026) / MemExplainer (ICML 2026) [ABS]: generic TGNN explainers, not applied to finance. → First financial-contagion use.
- **Anchor**: *extends/differs*. The anchor's price lead-lag edges become one typed relation; the task changes from generic edge
  prediction to event-conditioned exposure classification, and it adds the interpretability evaluation the anchor names as future work.

**5. Kill test.** Queries: `8-K bankruptcy filing contagion customers suppliers stock returns machine learning graph EDGAR 2026`;
`arXiv 2026 bankruptcy default contagion equity returns supply chain network peer firms graph`;
`explainable temporal graph neural network counterfactual explanation financial link prediction lead-lag 2026`. No ML paper on
public-data equity contagion after 8-K bankruptcy filings found. **Survives.**

**6. Data and access risk.** Events: EDGAR 8-K Item 1.03 (bankruptcy/receivership) filings, public (2026 examples verified on
sec.gov); optional extensions Item 2.04 (triggering events) and Item 3.01 (delisting notices). Supply links: 10-K "major customer"
disclosures (segment rules require disclosing customers with ≥ 10 % of revenue, but often **not by name**, **[UNVERIFIED]** share
named); extract with an LLM from EDGAR full text. Common ownership: 13F. Sector: SIC codes on EDGAR. Abnormal returns: prices
(Stooq/yfinance) + Kenneth French factors. **Risk: high.** Few events with *listed* counterparties, plus extraction effort.

**7. Method sketch.** For each event date, build the k-hop temporal heterogeneous subgraph around the failing firm from filings
dated *before* the event. Label each linked firm "exposed" if its cumulative abnormal return over [0, +5] (and [0, +20]) is
below a threshold (market-model CAR, estimation window [−250, −30]). Models: RGCN/HGT (PyG) with relation-typed edges + temporal
encodings; LightGBM on relation indicators and network features. Explainers: PyG GNNExplainer/PGExplainer, a CF-GNNExplainer-style
edge-deletion counterfactual, TemGX (code: <https://github.com/nicej1899/TemGX>).

**8. Baselines and evaluation.** Heuristics (same SIC, named major customer, top-k past correlation), logistic regression, LightGBM.
Metrics: AUPRC, precision@k per event, calibration; explanation fidelity (+/−), sparsity, stability across seeds. Chronological
split by event date; graph snapshots strictly pre-event.

**9. Artefact.** "Contagion map": enter a distressed firm → ranked exposed firms with the relation path that explains each (e.g.
"supplier, 18 % of revenue" or "shares 7 of top-10 holders").

**10. Feasibility, risks, fallback.** Weeks 1–6 review; 7–14 event + relation pipeline (the bottleneck); 15–20 models; 21–24 XAI
evaluation; 25–30 writing. Risk: too few events (listed-firm bankruptcies with listed partners may number only in the low hundreds,
**[UNVERIFIED]**). **Fallback:** widen "credit event" to severe negative shocks (e.g. ≥ 30 % one-day drop, large earnings misses
via 8-K Item 2.02 dates) to get thousands of events, keeping the same graph and XAI pipeline.

**11. Ethics.** Firm-level nodes only; do not use executive/board (Form 4, interlocks) data to avoid personal data. Present outputs
as risk analysis, not trading or credit advice; name firms only from public filings.

**12. Scores (§9).** Novelty 5 (15) · fit 5 (10) · data 2 (4) · feasibility 2 (4) · evaluation 3 · AI 4 · artefact 5 · industry 5 ·
Frank 5 → **55 / 70**. Gates: pass. Feasibility is the weak point.

**13. Questions for the supervisor.** Is event-study labelling (abnormal returns) acceptable as ground truth? Would she accept the
broadened "severe negative shock" fallback? Can we get WRDS Compustat Customer Segments (would remove the extraction risk)?
Is the XAI evaluation enough of an AI contribution, or should the model itself be novel?

---

### Card B-1: "Surprise edges": edge-level anomaly detection in signed lead-lag streams

**1. Pitch.** Train a temporal link predictor on daily signed lead-lag edges (as in the anchor, extended with signs), then flag the
edges it did **not** expect (new pairs, sign flips, expected links that vanish) as anomalies, and test whether these cluster around
public information events.

**2. Research question.** Is link-prediction surprise a useful *edge-level* anomaly score for stock networks, beyond count/recency
baselines and graph-level detectors?
- SQ1: Which surprise types (unexpected edge, missing edge, sign flip) concentrate around firm events (8-K filings, earnings releases) and macro/political events (FOMC, the 2025 tariff windows)?
- SQ2: Does edge-level localisation explain graph-level stress signals (as in B1 Caputi & Meadows)?
- SQ3: How do learned detectors compare with SimpleCount-type baselines and BAG detectors on injected anomalies with known ground truth?

**3. Supervisor directions hit.** **Anomaly detection** · temporal · **signed, time-activated edges** · link prediction · time series + graph.

**4. Closest 2026 work and the difference.**
- B1 Caputi & Meadows 2026 [FT]: graph-level anomalies on correlation snapshots, 7 labelled events, negative edges dropped. → Edge-level, signed, event-calendar evaluation.
- B2 ORCA 2026 [FT]: network features for crash/rally classification. → Unsupervised localisation, not index-level classification.
- B3 BAG (AAAI 2026) [ABS], B8 DynGraphAgentBench [ABS]: dynamic-graph anomaly benchmarks with **no equity data**. → Contributes a financial dataset and protocol.
- B4 SimpleCount 2026 [ABS]: one-feature baselines often win. → Mandatory baseline.
- **Anchor**: *transfer*. Turns the anchor's link predictor into a detector (prediction → surprise), and *extends* it with signs and an event-based evaluation.

**5. Kill test.** Queries: `temporal link prediction anomaly score unexpected edges stock market network earnings announcement graph 2026`;
`arXiv 2026 dynamic graph anomaly detection stock market correlation network`; `… temporal graph anomaly detection benchmark …`.
Nothing on edge-level anomalies in market lead-lag graphs; the 2024 TGN financial-anomaly paper (arXiv 2404.00060, background)
is transaction fraud. **Survives.**

**6. Data and access risk.** Daily prices for S&P 500 constituents (Stooq; yfinance is unofficial, so check terms). Event calendars, all
public: EDGAR 8-K filing timestamps (Item 2.02 = earnings results; Item 1.01/2.01 = deals; Item 1.03 = bankruptcy), FOMC dates
(federalreserve.gov), the 2025 tariff announcement dates (public record). Stress index: OFR Financial Stress Index **[UNVERIFIED]**.
**Risk: low–medium** (8-K item-level parsing is routine with EDGAR indices).

**7. Method sketch.** (a) Edge stream: residual returns (market + sector factors removed); edge i → j at day t if both exceed a
quantile threshold at lag τ, sign = sign(e_i(t)·e_j(t+τ)) (shared with C-1). (b) Predictors: GraphMixer/TGN via DyGLib or the anchor's
TGL code; a signed head. (c) Anomaly score = −log p(observed edge) and p(expected-but-absent edge); sign-flip score from the sign head.
(d) Comparators: SimpleCount-style count/recency/novelty features, Isolation Forest on edge features, BAG detectors (code:
<https://github.com/frhuaaa/BAG>), and B1's graph-level GlocalKD.

**8. Baselines and evaluation.** (i) **Injected anomalies** (BAG-style synthetic edges/sign flips) → AUROC/AUPRC with true labels;
(ii) **event enrichment**: precision@k of the top-k anomalies falling within ±1 trading day of an event at either endpoint, compared
with degree- and volatility-matched random edges (important: the volatility control prevents "anomaly = big move");
(iii) graph-level: aggregate edge surprise vs stress dates as in B1. Chronological deployment windows (as in B8); no fitting on test windows.

**9. Artefact.** Daily "surprise board": the top unexpected links (A → B, sign, surprise score), each with the nearest public event
(an 8-K headline, an earnings date) and a link to the filing. A natural market-surveillance analyst view.

**10. Feasibility, risks, fallback.** Weeks 1–6 review; 7–11 edge stream + event calendars; 12–17 predictors + detectors; 18–23
evaluations (i)–(iii); 24–30 writing. Risk: proxy labels are weak and anomalies may just track volatility (hence the matched controls).
**Fallback:** report injected-anomaly results + graph-level comparison with B1; event enrichment as exploratory analysis.

**11. Ethics.** Company-level only. Flagged "anomalies" are not accusations of manipulation; the thesis must say so. No personal data.

**12. Scores (§9).** Novelty 4 (12) · fit 5 (10) · data 4 (8) · feasibility 4 (8) · evaluation 3 · AI 4 · artefact 5 · industry 4 ·
Frank 4 → **58 / 70**. Gates: pass.

**13. Questions for the supervisor.** Is an event-enrichment evaluation (no true per-edge labels) acceptable for an MSc thesis, if
backed by injected anomalies? Should B-1 be combined with C-1 (shared edge stream) as one thesis with two chapters?

---

### Card B-2: Crowding anomalies in the 13F ownership graph

**1. Pitch.** Treat quarterly institutional holdings as a dynamic bipartite graph and use dynamic-graph anomaly detectors to spot
abnormal *coordinated* position changes (crowding into, or herding out of, a stock), then test whether these anomalies precede
stock-level crash risk or volatility spikes.

**2. Research question.** Do dynamic-graph anomaly scores on the 13F manager–stock graph identify crowding episodes that predict
next-quarter tail risk better than standard ownership-based crowding measures?
- SQ1: Which detectors (CTDG vs DTDG; learned vs count-based) give scores that carry forward-looking information?
- SQ2: Are anomalies concentrated in particular manager types (e.g. high-turnover) or sectors?
- SQ3: Do market-wide anomaly counts line up with known stress quarters (as GeomHerd reports qualitatively)?

**3. Supervisor directions hit.** Anomaly detection · temporal · heterogeneous (bipartite) graph · links (position entries/exits).

**4. Closest 2026 work and the difference.**
- GeomHerd 2026 (arXiv 2605.11645) [ABS]: Ricci curvature on a 13F fund-overlap graph; qualitative recovery of stress quarters, quantitative comparison deferred. → We do the quantitative, stock-level evaluation with ML detectors.
- A2 Izadifar & Rahmati 2026 [FT]: same data, prediction of weights. → Anomalies in *changes* rather than forecasts of levels.
- B3 BAG 2026 [ABS] and B4 SimpleCount 2026 [ABS]: detector zoo + mandatory simple baselines, no financial data.
- **Anchor**: *differs*. Ownership edges instead of price edges; it shares the TGNN model family and could add the anchor's lead-lag edges as a covariate graph.

**5. Kill test.** Queries: `institutional herding crowding detection anomaly 13F ownership network graph machine learning 2026`;
`13F institutional holdings … graph neural network temporal bipartite 2026`. Found GeomHerd (appendix only), Frontiers AI 2026
(aggregate indicators, China) and older crowding-signal work (arXiv 2306.08105, 2023, background). **Weakened** (the idea is
known in finance; the ML-detector + evaluation angle is open).

**6. Data and access risk.** SEC 13F (free), prices and volumes (Stooq/yfinance) for outcomes and a days-of-volume crowding proxy.
**Risk: medium** (same 13F mapping effort as A-1; the two can share a pipeline).

**7. Method sketch.** Quarterly DTDG snapshots and a CTDG stream of position-change events; detectors from BAG (TGN-, TGAT-,
DyGFormer-based, plus classical), SimpleCount-style features, Isolation Forest; node-level (stock) anomaly score = aggregated
edge-event anomaly. Outcomes: next-quarter max drawdown, realised volatility, crash indicator (bottom-decile return).

**8. Baselines and evaluation.** Ownership-breadth change, institutional ownership concentration (HHI), days-ADV proxy, GeomHerd-style
curvature. Metrics: AUROC/AUPRC for top-decile drawdown, rank correlation with realised volatility, simple portfolio sorts (secondary).
Strictly chronological; features available only after the filing date (45-day lag).

**9. Artefact.** "Crowding monitor": quarterly heat-map of stocks with anomalous coordinated buying/selling, drill-down to the managers involved.

**10. Feasibility, risks, fallback.** Weeks 1–6 review; 7–12 13F pipeline; 13–19 detectors; 20–24 outcome tests; 25–30 writing.
Risks: only ~50 quarters (few time steps); signal may duplicate known crowding measures. **Fallback:** a descriptive + predictive
comparison of graph-based vs classical crowding measures on S&P 500 stocks.

**11. Ethics.** As A-1: exclude or pseudonymise natural-person filers. Do not present results as allegations of coordination.

**12. Scores (§9).** Novelty 3 (9) · fit 4 (8) · data 4 (8) · feasibility 4 (8) · evaluation 3 · AI 3 · artefact 4 · industry 4 ·
Frank 3 → **50 / 70**. Gates: pass.

**13. Questions for the supervisor.** Is ownership-based anomaly detection close enough to her "stock network" idea? Would she prefer
B-1 (price graphs) over B-2 if only one anomaly topic is pursued?

---

### Card C-1: Signed temporal lead-lag link prediction, done rigorously

**1. Pitch.** The anchor predicts *whether* a stock follows another's big move. This thesis predicts *whether and in which direction*
(same or opposite sign) on market- and sector-neutral returns, and first checks whether the anchor's headline result survives the
2026 evaluation standards (memorisation, structural-proximity and volatility baselines; all-entity ranking).

**2. Research question.** Can temporal graph models predict the existence and sign of residual lead-lag edges between stocks
better than memorisation, structural-proximity and statistical lead-lag baselines, under leakage-safe evaluation?
- SQ1 (work package 1, re-evaluation): On the anchor's public dataset, does GraphMixer's advantage survive EdgeBank, time-decay PPR,
  node-degree/volatility scores, historical/inductive negatives and all-entity ranking?
- SQ2: Does residualising returns and adding opposite-direction (−) edges change what is learnable, and can the sign be predicted?
- SQ3: Do results transfer across universes (S&P 500 vs AEX/Euronext) and regimes (COVID-2020, the 2025 tariff shock)?

**3. Supervisor directions hit.** **Signed, time-activated influence edges + time series** (her capitalised point) · **link prediction** · temporal.

**4. Closest 2026 work and the difference.**
- **Anchor** (C0) [FT]: *extends* it with signs, residual returns, a larger universe and the 2026 evaluation protocol, and directly tests its main claim.
- C1 Regier et al. 2026 [FT]: temporal signed GNN (HCIM), Bitcoin trust networks, sign-only. → Real market data, existence + sign.
- A1 Fanshawe et al. 2026 [FT]: signed correlation *regression* with WRDS data. → Event-level link prediction on free data.
- C4 Yu & Ha 2026, C5 Fei et al. 2026 [ABS]: evaluation critiques on generic datasets. → First application to financial graphs.
- C6 Deng & Zhang 2026 [ABS]; Cartea et al. 2026 *J. Empirical Finance* [UNVERIFIED]: statistical lead-lag. → Serve as non-ML baselines.
- [BACKGROUND] MSGNN/FiLL (2022): static yearly signed lead-lag graphs. → We make it temporal (daily events).

**5. Kill test.** Queries: `signed lead-lag network stock opposite direction edge sign prediction temporal graph`; `2026 signed graph
neural network temporal link sign prediction financial correlation`; `EdgeBank baseline financial temporal graph stock lead-lag
re-evaluation …`. The only signed lead-lag LP work is static (MSGNN, 2022); the only temporal one is unsigned (anchor); no
re-evaluation exists. **Weakened → survives.**

**6. Data and access risk.** WP1: the anchor's processed dataset on GitHub (<https://github.com/ivankrstev7/Temporal_Graphs_Lead_Lag_Data>,
**not opened yet**; its commercial sentiment features can be dropped). Main study: daily prices (Stooq; yfinance unofficial),
Kenneth French factors and SPDR sector-ETF prices for residuals, AEX constituents via Stooq **[UNVERIFIED coverage]**. Survivorship
bias: use point-in-time constituent lists if obtainable, otherwise report the bias explicitly. **Risk: low.**

**7. Method sketch.** Edge definitions (compared, not assumed): (a) the anchor's thresholded co-exceedance (replication);
(b) residual co-exceedance with sign = sign(e_i(t)·e_j(t+τ)); (c) a statistical variant (rolling lagged correlation with a
significance filter). Models: DyGLib (TGN, TGAT, GraphMixer, DyGFormer, with existence + sign heads); the anchor's TGL code; signed
models (C1's HCIM + SE-SGformer: <https://github.com/incoder-mru/Historical-Context-Integration-Module>; PyG signed GNNs).

**8. Baselines and evaluation.** EdgeBank∞/tw; time-decay PPR (C5); popularity/degree; a **volatility-only score** (rank candidates by
recent |return|); ARIMA-pairwise (anchor); lagged-correlation sign. Metrics: AP/AUROC under random, historical and inductive negatives;
**all-entity MRR** (the catalogue is small, as C4 recommends); sign macro-F1/AUROC. Chronological splits with an embargo gap; 5 seeds;
Friedman + post-hoc tests (as the anchor). A key diagnostic is performance *conditional on edge novelty* (repeat vs new pairs).

**9. Artefact.** "Lead-lag explorer": today's large movers → predicted next-day followers with sign and confidence, plus a history
view showing when an edge switched on, off or flipped sign.

**10. Feasibility, risks, fallback.** WP1 (weeks 7–10) reuses the anchor repo and is low-risk and publishable as a short paper on its own.
Weeks 11–20 signed/residual pipeline + models; 21–24 transfer/regime tests; 25–30 writing. Risk: the signal may mostly be volatility;
that is itself a clear, honest finding. **Fallback:** WP1 + sign prediction on the anchor's 37-asset universe.

**11. Ethics.** No personal data. Disclose generative-AI use (coding) per UU rules. Not trading advice.

**12. Scores (§9).** Novelty 4 (12) · fit 5 (10) · data 4 (8) · feasibility 4 (8) · evaluation 5 · AI 4 · artefact 4 · industry 3 ·
Frank 4 → **58 / 70**. Gates: pass.

**13. Questions for the supervisor.** Is a thesis whose first result may be "the 2026 SOTA is mostly memorisation/volatility" welcome,
or must the thesis also propose a model? Which edge definition (thresholded vs statistical) does she find most meaningful? Is a
European universe (AEX) a plus?

---

### Card C-2: Do economic links explain price lead-lag? Heterogeneous temporal link prediction

**1. Pitch.** Lead-lag edges are learned from prices alone. Add the firms' *economic* relations as typed edges (supplier–customer,
common institutional owners, same sector, similar 10-K business text) and ask whether, and which, relations help predict and
explain who follows whom.

**2. Research question.** Do observable economic relations between listed firms improve temporal prediction of price lead-lag
edges, and which relation types does the model rely on?
- SQ1: Relation-type ablation: prices only vs + sector vs + common ownership vs + supply chain vs + text similarity.
- SQ2: Do lead-lag edges along economic links last longer or flip sign less often (edge-lifetime analysis)?
- SQ3: Do 2026 TGNN explainers (MemExplainer, TemGX) attribute predictions to the economically plausible relation types, and are these attributions stable?

**3. Supervisor directions hit.** **Heterogeneous graph** (not node classification) · **link prediction** · temporal · knowledge/text
networks (her text-analysis and knowledge-network expertise) · signed edges optional.

**4. Closest 2026 work and the difference.**
- **Anchor** (C0) [FT]: uses LLM text only as *node* features; most models do best with description embeddings. → We test text and
  economic relations as *typed edges*; *extends* the anchor from a homogeneous to a heterogeneous temporal graph.
- C2 Köse et al. 2026 [ABS]: LLM-reconstructed temporal supply/partner/ownership networks (semiconductors). → We use such links as predictors of market behaviour.
- C3 Yılkı 2026 [ABS]: supply-chain propagation of 10-K text signals for returns (51-edge KG). → Link prediction instead of factor construction; larger relation set.
- A3 Zhang 2026 [ABS]: LLM dynamic KG, simulation-only. → Real-data test.
- A2 Izadifar & Rahmati 2026 [FT]: 13F graph, providing the common-ownership relation.
- Sanity prior: a non-academic Oct-2026 study found **no drift along supply links** (QuantIntellect repo), so a null result is plausible and must be designed for.

**5. Kill test.** Queries: `supply chain customer momentum lead-lag link prediction graph neural network stocks 2026`; `arXiv 2026 supply
chain link prediction companies knowledge graph GNN SEC filings`; `arXiv 2026 LLM-constructed company knowledge graph stock relations
heterogeneous GNN`. No paper predicts lead-lag edges from typed economic relations. **Survives.**

**6. Data and access risk.** Sector: SIC codes (EDGAR, public). Text similarity: 10-K Item 1 business descriptions (EDGAR) embedded
with a sentence-transformer. Common ownership: 13F (as A-1). Supply chain: 10-K customer disclosures extracted with an LLM
(**effort + coverage risk**), or C2's network if released **[UNVERIFIED]**, or Wikidata relations **[UNVERIFIED coverage]**.
Prices as in C-1. **Risk: medium–high** (supply-chain coverage). The thesis still works with sector + ownership + text if supply links are thin.

**7. Method sketch.** Heterogeneous temporal graph: dynamic lead-lag edges (from C-1's pipeline) + slowly changing typed relation
edges (annual/quarterly). Models: TGN/GraphMixer with relation-type edge features (DyGLib); snapshot HGT/RGCN + GRU (PyG); a
pair-feature logistic/LightGBM model (relation indicators + EdgeBank + correlation), which is likely hard to beat. Explainers:
MemExplainer (<https://github.com/yazhengliu/MemExplainer>), TemGX.

**8. Baselines and evaluation.** All of C-1's baselines and protocol, plus the price-only model as the main comparison. Ablations with
paired significance tests across seeds. Explanation evaluation: fidelity, sparsity, rank agreement between explainers, and share of
attribution per relation type.

**9. Artefact.** Analyst "why does B follow A?" view: for a predicted lead-lag link, show the relation path (e.g. "A is B's largest
customer; 6 shared top-10 holders") and the explanation scores.

**10. Feasibility, risks, fallback.** Weeks 1–6 review; 7–10 C-1 edge pipeline (shared); 11–15 relation data (sector, text, 13F first;
supply chain last); 16–21 models + ablations; 22–25 XAI; 26–30 writing. **Fallback:** drop supply chain and keep sector + text +
ownership; the heterogeneous LP question still stands.

**11. Ethics.** Company-level relations only. Exclude board-interlock or insider (Form 4) relations, which involve named natural persons
(GDPR). LLM extraction: disclose model use and keep extracted triples auditable.

**12. Scores (§9).** Novelty 4 (12) · fit 5 (10) · data 3 (6) · feasibility 3 (6) · evaluation 4 · AI 4 · artefact 5 · industry 4 ·
Frank 4 → **55 / 70**. Gates: pass.

**13. Questions for the supervisor.** Which relation sources would she trust most (her knowledge-network work)? Is a possible null
result ("economic links add little beyond prices") acceptable? Could C-2 be the second half of a thesis that starts with C-1's WP1?

---

## Part 3: Runner-up cards (direction D, shorter)

### D-1: "Lead-lag or volatility?" A leakage and baseline audit of temporal stock-graph benchmarks
- **Pitch/RQ:** How much of the reported performance of temporal GNNs on financial graphs (anchor C0; also A1, A6-style setups) is
  explained by memorisation, node volatility, or evaluation choices (negative sampling, candidate-set size, future-informed graph construction)?
- **Builds on:** C4, C5 (2026 TLP evaluation critiques), B4 (SimpleCount), D1 "When Alpha Disappears" (leakage switches incl. future-informed
  graphs), D3 Look-Ahead-Bench (the anchor's GPT-4o descriptions are not point-in-time).
- **vs anchor:** a direct *re-test* of the anchor on its own public data.
- **Data/feasibility:** anchor dataset + public prices; very feasible (8–10 weeks of work). **Too small alone** for a 30-week thesis →
  recommended as **work package 1 of C-1** (already built in there).
- **Kill test:** no financial re-evaluation found (Phase 3 query 5). **Survives.**
- **Scores:** novelty 4 (12) · fit 4 (8) · data 5 (10) · feasibility 5 (10) · eval 5 · AI 4 · artefact 2 · industry 2 · Frank 3 → **56 / 70**.

### D-2: Tariff and sanction windows as out-of-distribution stress tests for learned company graphs
- **Pitch/RQ:** Do learned lead-lag/relation graphs (C-1, C-2) break, rewire or keep working during political shocks (the April-2025 US
  tariff announcements, sanction rounds), and can the graph change be localised to exposed sectors and supply links?
- **Builds on:** D6 (RMT complexity gap; tariff shock synchronised G5 markets), D7 (sector responses to the tariff windows), C2 (geopolitical realignment of firm ties).
- **vs anchor:** the anchor checked COVID only (Appendix H). This adds a political-shock regime.
- **Fit:** strong for Frank's interest in politics; **thin as a standalone thesis** → best as an evaluation chapter of C-1 or B-1.
- **Scores:** novelty 3 (9) · fit 3 (6) · data 5 (10) · feasibility 4 (8) · eval 3 · AI 3 · artefact 4 · industry 4 · Frank 5 → **52 / 70**.

### D-3: Point-in-time LLM-built company graphs
- **Pitch/RQ:** When an LLM extracts company relations from filings (A3, C2), how much of the downstream gain comes from look-ahead knowledge
  inside the LLM? Compare graphs extracted by point-in-time vs current LLMs, and with vs without the text.
- **Builds on:** D1–D3, DatedGPT [UNVERIFIED], C2, A3.
- **Risk:** heavy LLM extraction and compute; point-in-time models may not be available in the needed sizes. **Weakened.**
- **Scores:** novelty 4 (12) · fit 4 (8) · data 3 (6) · feasibility 2 (4) · eval 3 · AI 5 · artefact 2 · industry 4 · Frank 4 → **48 / 70**.

### Score overview (all cards)

| Card | Direction | Score / 70 | Main risk |
|---|---|---|---|
| **C-1** Signed temporal lead-lag LP (with WP1 re-evaluation) | C | **58** | Signal may be mostly volatility (still a finding) |
| **B-1** Surprise edges (edge-level anomalies) | B | **58** | Weak proxy labels |
| D-1 Lead-lag or volatility? (audit) | D | 56 | Too small alone → folded into C-1 |
| A-1 13F position-change events | A | 55 | Identifier mapping; few quarters |
| A-2 Credit-event contagion + counterfactual XAI | A | 55 | Few events; supply-link extraction |
| C-2 Economic links explain lead-lag | C | 55 | Supply-chain coverage; possible null result |
| D-2 Tariff/sanction stress tests | D | 52 | Thin alone → evaluation chapter |
| B-2 13F crowding anomalies | B | 50 | Overlaps known crowding measures |
| D-3 Point-in-time LLM company graphs | D | 48 | Compute and extraction effort |

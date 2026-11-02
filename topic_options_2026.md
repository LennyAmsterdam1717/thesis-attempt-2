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

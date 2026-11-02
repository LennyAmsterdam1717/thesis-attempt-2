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

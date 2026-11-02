# NOTES.md: decision log (thesis-attempt-2)

Run started 2026-11-02 (Europe/Amsterdam). Scope: thesis options for ML on stock/company networks,
directions **A** (heterogeneous-graph classification, non-standard angle), **B** (anomaly detection in
dynamic market/company graphs), **C** (link prediction: signed / lead-lag / supply chain / ownership /
co-mention), **D** (creative wildcards). Literature rule for this run: **first published or first posted in
2026 only**; dataset/library origin papers allowed as [BACKGROUND].

Verification legend: **[FT]** full text opened · **[ABS]** abstract/record page only · **[UNVERIFIED]** not opened.

---

## Phase 0: Anchor paper

**Krstev, I., Rigoni, D., Mishkovski, I., & Pasa, L. (2026). A Temporal Graph Learning Framework for Lead-Lag
Detection in Financial Markets. *Machine Learning* 115(7), article 173. doi:10.1007/s10994-026-07113-y.**
Published online 8 July 2026, peer-reviewed journal, open access (CC BY 4.0). **[FT]**: full PDF read via
web fetch of `link.springer.com/content/pdf/...` (the HTML article page was rate-limited twice; the PDF route
worked). Bibliographic data checked against the Crossref record.
Note: the PDF was **not** attached to the chat (the uploads folder was empty) and the title in the prompt was a
placeholder, so I identified the paper from its DOI.

### Task
Lead-lag detection framed as **temporal link prediction** on a dynamic graph: given the history up to day *t*,
predict whether a directed edge *i → j* (asset *i* moves on day *t*, asset *j* follows on day *t+τ*) will appear.
The authors also present the dataset as a real-world benchmark for temporal GNNs (TGNNs).

### Graph type
- **Temporal (event-based, continuous-time style), directed, homogeneous** (one node type "asset", one edge type).
- **Not signed in the signed-network sense.** Two scenarios: (i) "positive and negative": leader and follower move
  more than ε **in the same direction**, either up or down; (ii) "positive only": both rise. Opposite-direction
  (A up → B down) influence is **not modelled**. Edge sign/magnitude is not a prediction target.
- Edge rule: |r_i(t)| ≥ ε and |r_j(t+τ)| ≥ ε, same sign; ε = 5 %, τ = 1 trading day. So an edge means
  *co-exceedance of large moves one day apart*, not a statistically tested lead-lag (no Granger test or
  cross-correlation significance).
- Sparse: average degree ≈ 3.0, ≈ 14 edges/day with very high variance (SD ≈ 35), so edges cluster on
  volatile days.

### Data
- 37 nodes: 29 US-listed companies (e.g. Tesla, NVIDIA, Ford) + 8 commodities (e.g. gold, crude oil),
  5 sectors (energy, technology, materials, automotive, industrials). Daily, 2019-06-17 to 2024-06-12,
  1,257 trading days.
- Node features: OHLC, returns, VWAP, technical indicators (ADX, RSI, Williams %R, DEMA, TEMA, WMA, SD);
  **daily sentiment from the CityFalcon API** (commercial); short asset descriptions written by GPT-4o and
  embedded with `paraphrase-MiniLM-L6-v2` (384-d).
- Prices from City Falcon APIs (NYSE, CME); the code repo also has an `Extract_FMP_Data.ipynb` notebook
  (Financial Modeling Prep, a freemium API). **The processed dataset is public on GitHub**, but rebuilding it or
  extending it to new assets needs commercial sentiment data → for a thesis, swap in free sentiment (e.g. GDELT tone
  or a FinBERT model run on public headlines) or drop sentiment.

### Method
- Benchmarks nine deep models: LSTM (no graph), JODIE, DySAT, TGAT, TGN, APAN, GraphMixer (GM), GM-TNF (the authors'
  GM variant with temporal node features), CGLR (adapted from Xiao et al. 2025). Non-deep baselines: ARIMA,
  ARIMA-pairwise (AR(5) on each pair's binary edge series), CGLR-LLED (FFT cross-correlation at lag τ).
- Protocol: **strictly chronological split** (train to 2023-06-19, validation and test 130 days each);
  5 seeds; BCE loss, Adam lr 1e-4, 90 epochs; Colab T4 GPU. Metrics: AP, AUC, Recall@{1,5,10}, MRR.
  Friedman + Conover post-hoc with critical-difference diagrams.
- **Negative sampling: one random destination** (for AP/AUC); all other destinations for R@k/MRR. The
  authors note that true positives can be sampled as negatives.

### Main results
- **GraphMixer wins** (AP ≈ 0.79, AUC ≈ 0.85, MRR ≈ 0.47 in the positive-and-negative setting; similar in
  positive-only). JODIE/TGN AP ≈ 0.73–0.74; LSTM ≈ 0.52 (chance level); ARIMA-pairwise 0.675.
- Ablation: most models do best with *only* the text-description embeddings; adding price and other features
  usually hurts, except for GM. Adding temporal node features (GM-TNF) does not help → the edge history carries
  most of the signal.
- Sensitivity over ε ∈ {4,5,6}, τ ∈ {1,…,21}: best AP at ε = 6, τ = 21, but they keep ε = 5, τ = 1.
- Stable under the COVID regime (Appendix H), with higher variance.

### Code
- Code: <https://github.com/ivankrstev7/temporal_graph_lead_lag_detection> (exists, checked 2026-11-02: 2
  commits, notebooks `TGL_lead_lag.ipynb`, `GraphMixer_lead_lag.ipynb`, `build_dataset.ipynb`,
  `Extract_FMP_Data.ipynb`, `Extract_Sentiment_Data.ipynb`, plus vendored `tgl/` and `GraphMixer/`; no
  license shown on the page). Builds on **TGL** (Zhou et al. 2022) and the GraphMixer reference code.
- Data: <https://github.com/ivankrstev7/Temporal_Graphs_Lead_Lag_Data/> (link from the paper; not opened).

### What the anchor leaves open (my reading; these drive the topic ideas)
1. **Evaluation weakness.** Random one-destination negatives on a 37-node graph with hub-like volatile nodes
   make AP easy to inflate. There is no memorisation baseline (EdgeBank) and no node-degree or volatility
   baseline, and no historical or inductive negative sampling (TGB-style). The authors' own warning that
   results may reflect the volatility of specific stocks points the same way. → Is GM learning *lead-lag* or *who is
   volatile*? (rigour angle, D/C)
2. **No real signed or opposite-direction influence.** A↑ → B↓ (hedges, substitutes, competitors) is excluded.
   Signed temporal link-sign prediction is open. (C, supervisor's capitalised point)
3. **Homogeneous, tiny universe.** 37 nodes; no sector, supply-chain, ownership or news-co-mention relations
   as typed edges. → heterogeneous temporal graph: does structural company knowledge help predict lead-lag
   edges? (A/C)
4. **Edge semantics.** Thresholded co-exceedance confuses common market/sector shocks with pairwise
   lead-lag. Removing market/sector factors (residual returns) before building edges is an obvious and
   untested improvement.
5. **No anomaly view.** Which lead-lag edges are *unexpected* given the learned dynamics (e.g. around
   earnings, sanctions, defaults)? (B)
6. **No economic or interpretability evaluation.** Their future work names interpretability, which fits
   Frank's XAI background.
7. Future work they name themselves: larger datasets, other domains (crypto), interpretability.

---

## Access tests (this run)

| Resource | Result | Date |
| --- | --- | --- |
| link.springer.com article HTML (web fetch) | rate-limited twice (`RATE_LIMIT_EXCEEDED`) → not retried | 2026-11-02 |
| link.springer.com `/content/pdf/` (web fetch) | **works**, full text extracted | 2026-11-02 |
| api.crossref.org (web fetch) | works (metadata + reference list) | 2026-11-02 |
| GitHub web page (web fetch) | works | 2026-11-02 |
| GitHub API from shell (`gh api`) for repos not attached to session | 403 (session-scoped) → use web fetch instead | 2026-11-02 |

## Search log
(Queries per direction appended below as the search proceeds.)

- [anchor] web search `"10.1007/s10994-026-07113-y"`: only other MLJ papers; Crossref fetch found the title.
- [anchor] web search `"Temporal Graph Learning Framework for Lead-Lag Detection in Financial Markets"`: found
  DeltaLag (arXiv 2511.00390, **Nov 2025, so out of scope as evidence; background only**) and a Padova MSc thesis
  "GraphMixer in Financial Networks" (thesis.unipd.it, likely the anchor's precursor; [ABS], undated on the snippet).

### Direction A queries (2026-11-02)
1. `heterogeneous temporal graph neural network stock 2026 arXiv` (extended): found TH-GNN (shilling, off-topic), Fanshawe 2601.04602, Bukhori 2026 (BAREKENG), older THGNN (2022, background only).
2. `heterogeneous graph company classification financial distress supply chain 2026` (extended): many 2026 distress/fraud/supply-chain-finance HGNN papers (mostly ScienceDirect/ResearchGate, Chinese markets); Ind_HGF (Comput Econ 2026).
3. `arXiv 2026 LLM-constructed company knowledge graph stock relations heterogeneous GNN` (extended): Zhang 2607.10932; FinLLM-KG 2610.36083 (sentiment, KG at inference; noted for D); FinReflectKG (10-K triples, 700+ firms; a possible public data source, origin paper not opened).
4. `2026 graph neural network industry classification companies embeddings 10-K peers` (extended): nothing 2026 on stocks; off-topic results.
5. `arxiv 2026 "heterogeneous graph" stocks funds holdings mutual fund graph` (extended): **Izadifar & Rahmati 2607.12067 (13F node affinity)**; "Revisiting Simple Baselines for RDL" 2610.20534.
6. `arxiv 2026 temporal heterogeneous graph benchmark financial node classification firms events` (extended): FinFraudBench 2608.15177; TTGBench 2609.08226 (text-attributed temporal graph benchmark; noted); BenchTGC 2601.12903.
7. `arXiv 2026 stock heterogeneous graph explainability counterfactual explanation GNN market` (standard): CausalGraphX 2607.14416; XPlore 2603.04209 (general counterfactual GNN XAI); ICExplainer AAAI 2026.
8. `2026 arXiv stock "heterogeneous graph" US equities S&P 500 relational sector supply chain GNN new task` (standard): Wade 2605.19278 (code, yfinance); IEEE Access 2603.05917 (20 stocks, price prediction; crowded type); OmniGNN 2510.10775 (**Oct 2025, out of scope**).

Access: **ScienceDirect blocks web fetch (client_error)**; logged once, not retried. Those papers are kept as [UNVERIFIED] leads.
Decision A-1: plain distress/bankruptcy/fraud node classification on heterogeneous graphs = **crowded** (dozens of 2026 hits, mostly paywalled, Chinese data). Only angles that change the *task or evaluation* go forward.

### Direction B queries (2026-11-02)
1. `arXiv 2026 dynamic graph anomaly detection stock market correlation network` (extended): Caputi & Meadows 2604.02549; Schlinge et al. 2609.02965; AIR systematic review 2026; Springer bubble paper (10.1007/s10791-026-10424-2); 2510.17088 (Oct 2025, out).
2. `2026 market manipulation detection graph neural network stocks pump and dump coordinated trading` (extended): almost all **crypto** (Losavio 2604.24590, PumpWatcher, PumpSense); equity manipulation work is order-book/high-frequency (not public) or social-media (AIMM, Dec 2025).
3. `arXiv 2026 temporal graph anomaly detection benchmark continuous-time dynamic graph edge anomaly` (extended): DynGraphAgentBench 2609.33980; BAG (AAAI 2026); SimpleCount 2608.15965; TEMG-TTA (blockchain).
4. `"BAG" "Benchmarking Anomaly Detection on Dynamic Graphs" Hua 2026` (standard): AAAI 2026 record; IJCAI 2026 BAD paper (lead).
5. `arXiv 2026 systemic risk early warning graph neural network stock market crash network` (extended): Uddin 2604.14232 (FDIC); Umeorah & Fadina 2608.27295 (CDS); SRR 2512.17185 (Dec 2025, out).
6. `arXiv 2026 regime change detection stock correlation network change point graph` (extended): ORCA 2604.17251; Geodesics 2606.07151 (lead); Treasury text regime 2605.30363 (noted for D).

Access: **AAAI OJS (ojs.aaai.org) blocks web fetch** (bot protection); **Springer article pages are inconsistent** (Comput Econ page worked; AIR review failed). Logged, not retried.
Decision B-1: finance-specific dynamic-graph anomaly work in 2026 is **thin** (a handful of preprints, mostly graph-level stress detection on small universes with very few labelled events). Generic DyGAD benchmarks (BAG, DynGraphAgentBench) contain **no equity data**. → B is relatively open, but the main risk is *ground truth* (few labelled events).

### Direction C queries (2026-11-02)
1. `arXiv 2026 lead-lag stock network learning graph neural network` (extended): the anchor's own Springer record (received 2026-01-15); news/supply-chain "lead-lag" GNN predictors (ResearchGate, unverified); DeltaLag (Nov 2025, out).
2. `2026 signed graph neural network temporal link sign prediction financial correlation` (extended): **Regier et al. 2605.26290**; news co-occurrence TGNN (ScienceDirect, blocked); many price-prediction GNNs (crowded).
3. `arXiv 2026 supply chain link prediction companies knowledge graph GNN SEC filings` (extended): Yılkı 2606.29290; supply chain KG completion (MDPI Sustainability 2026, non-equity); older Brintrup/Kosasih line (background).
4. `2026 temporal graph link prediction evaluation negative sampling EdgeBank benchmark critique TGB` (extended): **Yu & Ha 2607.27861; Fei et al. 2610.38385; 2609.35364; 2610.33171**. A strong 2026 wave of "simple baselines beat TGNNs".
5. `arXiv 2026 lead-lag relationships equities network detection q-fin` (extended): Deng & Zhang 2608.24703; 2602.07048 (prediction markets); **Cartea et al. 2026 JEF (unverified, important)**; Bennett et al. 2022 (MLJ, background).
6. `arXiv 2026 news co-mention company network temporal link prediction firms GDELT` (extended): **Köse et al. 2605.15842 (LLM firm networks)**; Papadopoulos 2604.23714 (temporal connection probabilities, physics); no 2026 paper on *predicting* news co-mention links between listed firms found.

Access: arXiv abs pages rate-limit after several quick fetches (2610.33171 failed twice → [UNVERIFIED]).
Decision C-1: **No 2026 paper does signed (same- vs opposite-direction) temporal link prediction on stock lead-lag graphs**. The anchor (unsigned) and Regier et al. (signed, non-financial) bracket the gap.
Decision C-2: The 2026 temporal-link-prediction evaluation literature (C4, C5, 2609.35364, 2610.33171) gives a ready-made rigour toolkit that the anchor did not use. A re-evaluation is a credible, low-risk thesis component.
Decision C-3: Firm-relation link prediction (supply chain / partnership) has a new public-ish data route (LLM extraction from web/10-K; C2) but no 2026 *temporal link prediction* study on listed firms was found.

### Direction D queries (2026-11-02)
1. `arXiv 2026 bankruptcy default contagion equity returns supply chain network peer firms graph` (extended): DefaultGNN 2609.25542 (private data); guarantee-network HGNN (Springer, unverified); older Hungarian supply-chain contagion work (2023, background).
2. `arXiv 2026 tariff sanctions shock propagation firm network stock returns graph neural network` (extended): RIETI 26-E-045 (Arata); G5 RMT tariff paper 2604.19107; tariff mediation 2609.33881; no graph-ML study of the 2025 tariff episode on listed-firm networks found.
3. `2026 look-ahead bias leakage stock prediction graph neural network benchmark reproducibility re-evaluation` (standard): "When Alpha Disappears" 2605.23959; Zenodo leakage-controlled benchmark; Look-Ahead-Bench 2601.13770; DatedGPT 2603.11838.
Decision D-1: **Evaluation rigour** is a hot 2026 theme in both temporal-graph ML (C4, C5, B4) and financial ML (D1–D3), but **nobody has joined the two for stock graphs**. This is a strong wildcard.
Decision D-2: Credit-default contagion into equities via *public* data (EDGAR 8-K Item 1.03 + 10-K customer disclosures) shows no 2026 graph-ML paper; all 2026 default-GNN papers use private transaction/guarantee/tax data.

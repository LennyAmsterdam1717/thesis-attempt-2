# Literature review: ML on stock and company networks (2026 only)

Scope: papers **first published or first posted in 2026**. Dataset/library origin papers appear only as
**[BACKGROUND]**. Legend: **[FT]** full text opened (via web fetch, which extracts and summarises; numbers
must be re-checked in the PDF before thesis use) · **[ABS]** abstract or record page only · **[UNVERIFIED]**
not opened, listed only as a lead.

Reference point throughout: **Krstev, Rigoni, Mishkovski & Pasa (2026)**, *Machine Learning* 115:173,
"A Temporal Graph Learning Framework for Lead-Lag Detection in Financial Markets" (the **anchor**; summary in
`NOTES.md`). Each entry ends with a line saying how it relates to the anchor.

<!-- CROWDING MAP (Phase 2) GOES HERE -->

---

## Direction A: classification and node-level learning on heterogeneous (temporal) company graphs

### A1. Fanshawe, Masih & Cameron (2026). Forecasting Equity Correlations with Hybrid Transformer Graph Neural Network. arXiv:2601.04602
- Link: <https://arxiv.org/abs/2601.04602> · First posted 2026-01-08 (arXiv v1, q-fin.CP) · preprint · **[FT]**
- Note: the PDF carries an internal date "01/07/2025" (possibly a July-2025 thesis/working paper). By arXiv v1 it is 2026.
- Summary: Predicts 10-day-ahead stock-pair correlations for S&P 500 constituents as a residual in Fisher-z space on
  top of a rolling baseline. Transformer temporal encoder per stock + edge-aware GAT over a daily graph; edges get
  a 3-class relation type (negative / neutral / positive) with expert heads. Strong gains over persistence
  (Pearson 0.31 → 0.78 per the paper) and a better basket-trading Sharpe ratio.
- Graph: temporal (daily snapshots), **signed** weighted edges, "heterogeneous" only in edge types (one node type).
- Task: edge regression (correlation forecasting), with clustering and trading downstream.
- Data: CRSP/Compustat via WRDS (**paid**), FRED (public). 2006–2018 train, 2019–2024 out-of-sample with 60-day buffer.
- Code: none listed.
- Leaves open: no ML/GNN baselines (authors admit it); no transaction costs; a single hyperparameter configuration;
  interpretability only via attention + Gradient×Input; internal inconsistencies (20 vs 30-day baseline).
- vs anchor: same "edges between stocks over time" idea, but it **regresses signed correlation strength** where the
  anchor **classifies thresholded lead-lag events**. Its signed relation-type routing is exactly what the anchor lacks.

### A2. Izadifar & Rahmati (2026). Institutional Equity Holdings Prediction Using Node Affinities of Dynamic Graphs. arXiv:2607.12067
- Link: <https://arxiv.org/abs/2607.12067> · First posted 2026-07-13 · preprint (marked for an Elsevier journal) · **[FT]**
- Summary: The first temporal-graph benchmark for SEC **Form 13F** holdings. A quarterly bipartite manager → security graph
  (99 sampled managers, 503 S&P 500 securities, 48 quarters 2013–2025); predicts next-quarter portfolio weights as
  **TGB node-affinity prediction** (NDCG@10). NAVIS is best (≈0.913), but the persistence forecast (≈0.889) and EMA
  (≈0.888) beat every other learned dynamic-graph model; node features add < 1.2 %.
- Graph: **discrete-time temporal, heterogeneous (2 node types), directed, weighted bipartite.**
- Task: node property / affinity prediction (closest to link-weight prediction).
- Data: **SEC EDGAR 13F (public, free)**; CUSIP → S&P 500 fuzzy matching. Code: <https://github.com/e-izdfr/portfolio-holdings-prediction>.
- Leaves open: random 99-manager sample; heuristics are near-unbeatable (high portfolio persistence); no economic
  validation in the shown text; changes in holdings (entries/exits) are the interesting signal, not levels.
- vs anchor: same TGB/TGL tooling family and also uses temporal-graph benchmarks, but on **ownership** edges, not price
  co-movement. Combining both (does common ownership predict lead-lag edges?) is untested.

### A3. Zhang (2026). LLM-Enhanced Dynamic Financial Knowledge Graphs for Cross-Entity Signal Propagation and Alpha Discovery. arXiv:2607.10932
- Link: <https://arxiv.org/abs/2607.10932> · First posted 2026-07-12 (stat.AP, working paper) · preprint · **[ABS]** (abstract + intro snippet)
- Summary: An LLM turns documents into state-change events and extracts explicit and implicit firm relations (supplier,
  customer, competitor, technology, capex exposure, industry) into a monthly dynamic knowledge graph; community-aware
  propagation of "information surprise" signals. **Evaluated on simulations only** (one calibrated to the Russell 1000).
- Graph: temporal, multi-relational (typed edges). Task: signal propagation / cross-sectional return ranking.
- Data: simulated; no real backtest shown. Code: not listed.
- Leaves open: real-data validation; quality of LLM-extracted edges; how construction choices change results.
- vs anchor: the anchor uses LLM text only as **node** features (GPT-4o descriptions). Using LLM-extracted **typed edges**
  as structural priors for lead-lag link prediction is the natural bridge.

### A4. Zhao, Zheng, Sun, Yu, Ma, Wang & Zhang (2026). A Novel Industry-aware Hierarchical Graph Fusion Model for Multi-modal Financial Distress Prediction. *Computational Economics*.
- Link: <https://doi.org/10.1007/s10614-026-11387-8> · Published online 2026-06-29 · peer-reviewed · **[ABS]** (paywalled)
- Summary: Ind_HGF fuses financial and textual modalities and builds company-level and industry-level graphs joined by
  hierarchical message passing for distress classification.
- Graph: two-level (company / industry) hierarchical graph; temporal aspect not stated. Task: **node classification**.
- Data: "real-world datasets" (market not stated in abstract; this literature is mostly Chinese A-shares, unverified).
- Leaves open: typical of a crowded line (distress/bankruptcy node classification with a new fusion module).
- vs anchor: different task (static node labels vs temporal edges). It illustrates the "plain heterogeneous
  classification" the supervisor calls done.

### A5. Karanjai, Madhavarao, Xu & Shi (2026). CausalGraphX: A Counterfactual Graph Neural Network Framework for Explainable Systemic Risk Assessment. arXiv:2607.14416
- Link: <https://arxiv.org/abs/2607.14416> · First posted 2026-07-15; AAAI'26 workshop (Agentic AI in Financial Services) · workshop paper · **[ABS]**
- Summary: GAT with adversarial regularisation predicts cascading defaults of institutions; an optimisation procedure gives
  sparse counterfactuals ("what minimum capital injection would have prevented the default").
- Graph: institution network (type not stated); **synthetic data**. Task: node classification + counterfactual XAI.
- Code: none. Leaves open: real public data; temporal dynamics; equity-market transmission.
- vs anchor: none directly. It shows that **counterfactual explanations on financial graphs** are wanted but only
  tested on synthetic networks. A counterfactual explainer for the anchor's lead-lag predictions would be new.

### A6. Wade (2026). Do Better Volatility Forecasts Lead to Better Portfolios? Evidence from Graph Neural Networks. arXiv:2605.19278
- Link: <https://arxiv.org/abs/2605.19278> · v2 dated 2026-05-20 (v1 date not shown; id implies May 2026) · preprint · **[FT]**
- Summary: Controlled comparison of HAR, LSTM and GraphSAGE over three graph constructions (rolling |ρ| ≥ 0.3, GICS
  sector, Bonferroni-corrected Granger), with and without macro features, for weekly realised volatility of 465 S&P 500
  stocks. Correlation graphs help point forecasts; only minimum-variance portfolios clearly benefit; long-short
  is negative for all models.
- Graph: snapshot graphs of several relation types (compared, not fused). Task: node regression + portfolio evaluation.
- Data: **yfinance (public)**, 2015–2025, chronological split; survivorship bias admitted. Code: <https://github.com/waderylan/sp500-gnn>.
- Leaves open: forecast accuracy ≠ economic value; static Granger graph; graph choice matters more than architecture.
- vs anchor: a useful **public-data, reproducible template** (yfinance + PyG). Its message that "graph construction
  matters" supports re-examining the anchor's edge definition.

### A7. Nguyen, Dwivedi & Gust (2026). Revisiting Simple Baselines for Relational Deep Learning. arXiv:2610.20534
- Link: <https://arxiv.org/abs/2610.20534> · First posted 2026-10-14 · preprint · **[ABS]**
- Summary: On RelBench, an MLP over fixed low-pass-filtered neighbourhood features recovers much of heterogeneous-GNN
  performance; Correct-and-Smooth over task-specific meta-graphs adds more.
- Graph: relational entity graphs (heterogeneous). Task: node classification/regression. Not finance-specific.
- vs anchor: methodological. Simple-baseline rigour for heterogeneous graphs is exactly what the anchor's benchmark
  lacks (no EdgeBank, no degree baseline).

### A8. Chen et al. (2026). FinFraudBench: A Heterogeneous Graph Benchmark for Financial Fraud Detection. arXiv:2608.15177
- Link: <https://arxiv.org/abs/2608.15177> · First posted 2026-08-15 · preprint · **[ABS]** (abstract + HTML snippet)
- Summary: Two public heterogeneous graphs (6 node types, 14 directed edge types) built from public transaction
  tables; binary node classification of transactions; heterogeneous designs give the strongest signal.
- Not stock-market data (transactions). Included to show that heterogeneous **classification benchmarks** are being
  consolidated in 2026, so the classification niche is filling up.
- vs anchor: none directly.

### Direction A: unverified leads (not opened; seen only in search snippets; not used as evidence)
- [UNVERIFIED] "Multi-modal bankruptcy risk prediction for listed companies via large language models", *Information Sciences* 2026 (ScienceDirect S002002552600318X; site blocked web fetch).
- [UNVERIFIED] "Credit risk assessment in supply chain finance using dynamic temporal evolution hypergraph neural network" (DySC-HGNN), *Expert Systems with Applications* 2026 (S0957417426027065; blocked).
- [UNVERIFIED] Bukhori et al., "Heterogeneous Graph Neural Networks for Stock Price Prediction…", *BAREKENG* 20(2), 2026.
- [UNVERIFIED] ResearchGate listings (2026): CrossHGAT, FFD-DHG, SCALE, "Multi-Perspective Fusion Graph Model for Financial
  Distress Prediction of Listed Companies", "Temporal Attentive Graph Networks for Financial Surveillance" (TAGN).
  These indicate a very crowded Chinese-market distress/fraud classification literature.

---

## Direction B: anomaly detection in dynamic market and company graphs

### B1. Caputi & Meadows (2026). Financial Anomaly Detection for the Canadian Market. arXiv:2604.02549
- Link: <https://arxiv.org/abs/2604.02549> · First posted 2026-04 (page dated March 2026; id = April 2026) · preprint · **[FT]**
- Summary: Graph-level anomaly detection on daily correlation graphs (Convergent Cross Mapping, 25-day window, negatives
  zeroed) of 39 TSX-60 stocks, 2005–2021. Compares TDA (persistent homology on directed flag complexes), PCA and
  unsupervised GNNs (One-Shot GIN(E), GlocalKD). Ground truth = seven Canadian Financial Stress Index peaks; an event
  counts as detected if an anomalous graph (score > 97.5th percentile) appears within 50 days before it. GlocalKD is best
  (F1 0.68; DJIA appendix 0.85); raw/PCA 0.28–0.45.
- Graph: temporal snapshot sequence, homogeneous, weighted (negative edges dropped). Task: **graph-level anomaly / early warning**.
- Data: **Yahoo Finance via yfinance (public)**. Code: <https://github.com/njmead811/Financial-Anomaly-Detection-for-the-Canadian-Market> (PyG, pyflagser).
- Leaves open: only 7 events (very small evaluation sample); fixed thresholds; **negative correlations discarded**; no
  edge- or node-level localisation (which stocks or links are anomalous?).
- vs anchor: both use daily equity graphs from public prices, but B1 scores **whole snapshots** where the anchor predicts
  **individual edges**. Using a lead-lag link predictor's surprise (low predicted probability for edges that do appear)
  as an edge-level anomaly score would join the two.

### B2. Kriuk & Kriuk (2026). ORCA: Online Regime Correlation Analyzer. arXiv:2604.17251
- Link: <https://arxiv.org/abs/2604.17251> · First posted 2026-04 (from id; no date on the HTML) · preprint · **[FT]**
- Summary: 127 spectral and graph features of rolling correlation networks over 24 ETFs (absorption ratio,
  eigenvalue entropy, Marchenko–Pastur excess, density and clustering at thresholds) + 79 price features → Random
  Forest classifies 10-day rally/crash events. Walk-forward (8 folds, 10-day gap). Crash AUC 0.71 vs 0.61 without
  network features, but HAR-RV gets crash AUC 0.76. SHAP shows topology features dominate crash predictions.
- Graph: rolling correlation snapshots (threshold graphs). **No GNN.** Task: event classification / early warning.
- Data: EODHD API (commercial, but ETF prices are also on Stooq/yfinance). Code: none (live demo only).
- Leaves open: inconsistent ablation numbers; a simple volatility model beats it on crashes; portfolio backtest tuned in-sample.
- vs anchor: a **non-deep, interpretable** counterpoint. Network-structure summaries carry crash signal, which
  suggests aggregate lead-lag graph statistics (from the anchor's edge stream) as features.

### B3. Hua, Qi, Wei, Tian, Xu, Wu, Li & Guo (2026). BAG: Benchmarking Anomaly Detection on Dynamic Graphs. *AAAI 2026* 40(17):14892–14900
- Link: <https://doi.org/10.1609/aaai.v40i17.38510> · Published 2026-03-14 (conference 2026-01) · peer-reviewed · **[ABS]**
  (abstract + first page seen only via the search-engine extract; the AAAI page itself blocked web fetch, so not opened directly)
- Summary: First comprehensive dynamic-graph anomaly benchmark: 25 models (classical, DTDG and CTDG DGNNs), 10
  datasets with synthetic and organic anomalies, node- and edge-level. Main finding: continuous-time (CTDG) models do
  best on edge anomalies. Code: <https://github.com/frhuaaa/BAG>.
- vs anchor: BAG wraps the same CTDG model family as the anchor (TGN, TGAT, JODIE, GraphMixer-type) for anomaly scoring,
  so a **lead-lag edge stream could be dropped into BAG** with little engineering. No financial-market dataset is evident.

### B4. Ahmed & Shafi (2026). Beat the Counter First: A Baseline for Temporal-Graph Anomaly Detectors. arXiv:2608.15965
- Link: <https://arxiv.org/abs/2608.15965> · First posted 2026-08-16 · preprint · **[ABS]**
- Summary: SimpleCount, a parameter-free one-feature (count/recency/novelty) baseline, matches or beats the SLADE
  detector on 3 of 6 datasets and beats IsoForest on all, at 23–133× less compute; on synthetic data with planted
  structure all detectors are near random.
- vs anchor: same lesson as for link prediction (EdgeBank). **Any lead-lag anomaly or link-prediction claim must beat
  count/recency baselines**, which the anchor never tests.

### B5. Schlinge, Schnipper & Atzmueller (2026). Statistical Feature Augmentation for Anomaly Detection in Dynamic Graphs. arXiv:2609.02965
- Link: <https://arxiv.org/abs/2609.02965> · First posted 2026-09-02 · preprint · **[ABS]**
- Summary: Adds classical network-statistic features to temporal GNNs; consistently improves anomaly detection on
  Reddit/Wikipedia/MOOC across 7 continuous- and discrete-time models.
- vs anchor: the anchor found extra node features mostly *hurt*; this paper finds structural statistics *help* for
  anomalies. Contrasting the two on market graphs would be a cheap experiment.

### B6. Losavio, Persia, Sathe & Pasadakis (2026). Fraud Detection in Cryptocurrency Markets with Spatio-Temporal Graph Neural Networks. arXiv:2604.24590
- Link: <https://arxiv.org/abs/2604.24590> · First posted 2026-04-27 · preprint · **[ABS]**
- Summary: Three graph constructions from aggregated hourly market data, then an ST-GNN (spatial attention + temporal
  Transformer) to detect pump-and-dump events over 3+ years; graph models beat standard ML baselines.
- **Crypto-only. Transfer label: the method (graphs built from aggregated price/volume only) transfers to equities;
  labelled manipulation data for equities does not exist publicly** (except SEC litigation releases, unverified).
- vs anchor: same "market-data-only graph construction" philosophy; supervised event labels instead of lead-lag edges.

### B7. Uddin (2026). Explainable Graph Neural Networks for Interbank Contagion Surveillance: A Regulatory-Aligned Framework for the U.S. Banking Sector. arXiv:2604.14232
- Link: <https://arxiv.org/abs/2604.14232> · First posted 2026-04-14 · preprint · **[ABS]**
- Summary: ST-GAT (GAT + BiLSTM) over 8,103 FDIC banks, 58 quarters (2010Q1–2024Q2); bilateral exposures reconstructed
  from **public FDIC Call Reports** by maximum entropy; distress early warning. AUPRC 0.939, just below XGBoost
  (0.944); attention + permutation importance (ROA, NPL ratio dominate).
- vs anchor: different entities (banks, balance-sheet exposures). Relevant to the credit/contagion side, and it
  shows again that **gradient boosting matches the GNN**.

### B8. Han, Wei, Yang, Huang, Fang, Ma & Yu (2026). DynGraphAgentBench: A Benchmark for Agentic Lifecycle Control in Dynamic Graph Anomaly Detection. arXiv:2609.33980
- Link: <https://arxiv.org/abs/2609.33980> · First posted 2026-09-27 · preprint · **[ABS]**
- Summary: 7 temporal graph datasets with node/edge anomaly tasks, 11 detectors (built on BAG), 8 chronological deployment
  windows with **delayed labels**; a controller picks/retrains detectors using only time-causal information.
- vs anchor: offers a deployment-realistic protocol (chronological windows, delayed feedback) that market surveillance
  needs and the anchor's single split lacks.

### Direction B: out of scope by date (noted, not used as evidence)
- Neela, "Systemic Risk Radar" arXiv:2512.17185 (**Dec 2025**).
- "Explainable Heterogeneous Anomaly Detection in Financial Networks via Adaptive Expert Routing" arXiv:2510.17088 (**v1 Oct 2025**, v2 Mar 2026).

### Direction B: unverified leads
- [UNVERIFIED] Ares-Robledo, Rifà-Pous & Clarisó (2026), "Graph neural networks for anomaly detection: a systematic review of dynamic temporal approaches", *Artif. Intell. Rev.* 59:129, doi:10.1007/s10462-026-11532-7 (open access; Springer page failed to fetch). **Worth reading in phase 1.**
- [UNVERIFIED] "Dynamic relational network embedding-based anomaly detection for predicting financial risks" (stock bubble detection, transformer dynamic network embedding + autoencoder), Springer, doi:10.1007/s10791-026-10424-2.
- [UNVERIFIED] Luo et al. (2026), "Unsupervised Anomaly Detection in Dynamic Graphs via Compatibility Modeling and Boundary Learning" (BAD), IJCAI 2026 (ijcai.org/proceedings/2026/0322.pdf).
- [UNVERIFIED] Umeorah & Fadina (2026), "A Temporal Multiplex Graph Neural Network for Systemic Risk Transmission in Global Banking", arXiv:2608.27295 (CDS spreads are paid data).
- [UNVERIFIED] Cappelletti, Voutaz & Frossard (2026), "Geodesics of Dynamic Graphs for Regime Change Detection", arXiv:2606.07151.

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

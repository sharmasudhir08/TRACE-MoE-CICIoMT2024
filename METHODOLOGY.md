# Methodology: TRACE-MoE

This file describes what the code does, in the order the paper's Method section should present it.
Each part names the file that implements it. (The folder and Python package keep the name `trace-iomt` /
`trace_iomt`; the model is TRACE-MoE.)

## 0. Research questions, objectives and hypotheses

### Aim
Build and test TRACE-MoE, a protocol-routed mixture-of-experts intrusion detector for CICIoMT2024 (Wi-Fi/MQTT).
It should match published accuracy on clean traffic, stay robust under realistic feature attacks, and abstain
with an "unknown attack" alert when it is unsure.

### Research questions
| RQ | Question | Special-issue keyword |
|---|---|---|
| RQ1 | Does protocol-routed MoE improve 19-class detection, especially rare and look-alike DoS/DDoS classes, over published winning models under the same split? | Architectures |
| RQ2 | Do the experts specialise by protocol and attack family, and does routing add value over a single dense model of the same size? | Architectures |
| RQ3 | How much do the models degrade under feature-constrained white-box and transfer attacks at 19 classes, and does adversarial fine-tuning of TRACE-MoE lower the attack success rate (ASR)? | Robustness |
| RQ4 | Can calibrated abstention flag unseen attack types and lower the error on the traffic the model accepts? | Robustness, Applications |
| RQ5 | Is TRACE-MoE light and fast enough for an IoMT edge gateway compared with the baselines? | Applications |

### Objectives
| Objective | What we do | Answers | Delivered by |
|---|---|---|---|
| O1 | Reproducible pipeline: official train/test files, 5 seeds, duplicate/overlap/shortcut checks as a sanity step | all | `01_prepare.py`, `02_audit.py` |
| O2 | Design TRACE-MoE: flow window + protocol token, shared encoder, router with 4 experts, 2/6/19-class heads, abstain layer | RQ1, RQ2 | `models/networks.py`, `abstain.py`, `04_train_deep.py` |
| O3 | Benchmark against RF, BiLSTM-DNN, Transformer and TF-SAGE-lite under one protocol; macro-F1, balanced accuracy, minimum class recall, ECE | RQ1 | `03_train_rf.py`, `04_train_deep.py`, `09_compare.py` |
| O4 | Ablate the router (dense expert of equal size, protocol-only routing, no experts) and measure expert specialisation | RQ2 | `05_ablation.py`, routing report in every TRACE-MoE run |
| O5 | Feature-constrained PGD (18 mutable features) over an ε grid plus a transfer matrix; adversarial fine-tuning; ASR | RQ3 | `06_robustness.py` |
| O6 | Leave one of 5 attack types out of training; unknown-attack AUROC, FPR@95%TPR, risk-coverage (AURC) | RQ4 | `07_unknown_attacks.py`, abstain report in every main run |
| O7 | Parameters, latency, memory; comparison with 5 published papers by their reported numbers | RQ5, positioning | `08_efficiency.py`, `configs/published_results.yaml` |

### Hypotheses, tests and decision rules
Pairwise tests use the Wilcoxon signed-rank test over (class, seed) pairs (about 19 × 5 = 95 pairs, so p < 0.05
is reachable; 5 seeds alone could never give p < 0.05) with Holm correction at α = 0.05. Thresholds are in
`configs/default.yaml` → `hypotheses`. `10_technical_report.py` computes every verdict (`reporting.hypothesis_verdicts`):
supported, partly supported, not supported, or not run, with the evidence numbers next to it.

| H | RQ | Hypothesis | Test | Supported if | Output file |
|---|---|---|---|---|---|
| H1 | RQ1 | TRACE-MoE has higher 19-class macro-F1 than each baseline | Wilcoxon on per-class F1, Holm | significant and positive vs all 4 baselines | `tables/wilcoxon_trace_moe_vs_baselines_official` |
| H2 | RQ1 | TRACE-MoE raises the recall of the weakest class | minimum class recall, mean over seeds | ≥ 5 points above the best baseline | `tables/main_official_label19` |
| H3 | RQ1 | On binary detection TRACE-MoE is non-inferior to RF | per-seed difference of binary macro-F1, 95% t-interval | lower bound above −0.5 points | `tables/noninferiority_binary` |
| H4 | RQ2 | Routing helps: one dense expert of the same size lowers 19-class macro-F1 | ablation, Wilcoxon (full vs dense_expert) | significant drop | `tables/ablation_tests` |
| H5 | RQ2 | The experts specialise by attack family | NMI between chosen expert and 6-class category | NMI ≥ 0.5 | `tables/expert_routing` |
| H6 | RQ3 | Adversarially fine-tuned TRACE-MoE has lower ASR than every (equally fine-tuned) baseline at ε = 0.25 | Wilcoxon on per-class ASR, Holm | significant vs all neural baselines | `tables/robustness_asr_tests` |
| H7 | RQ3 | Adversarial fine-tuning costs little clean accuracy | clean 19-class macro-F1 before vs after | drop ≤ 1 point | `tables/robustness_asr`, main runs |
| H8 | RQ4 | The abstain score detects held-out attack types | mean AUROC over 5 leave-one-attack-out classes | AUROC ≥ 0.90 and above softmax confidence | `tables/zero_day_overall` |
| H9 | RQ4 | Abstaining lowers the error on accepted traffic | AURC of the risk-coverage curve | lowest AURC of all models | `tables/abstain_official` |
| H10 | RQ5 | TRACE-MoE is faster than the Transformer baseline and fits an edge budget | median CPU latency per window (batch 1024), parameter count | faster than Transformer and < 5 M parameters | `efficiency/efficiency` |

Report each hypothesis as the results show. A rejected hypothesis is still a finding.

## 1. Problem and data

* **Dataset.** CICIoMT2024, Wi-Fi/MQTT part: 45 flow features per row, official train and test CSV files.
  The CSVs have no label column; the label is in the file name (`TCP_IP-DDoS-ICMP1_train.pcap.csv` →
  `TCP_IP-DDoS-ICMP`), see `data/labels.py`.
* **Three label levels** (`Hierarchy` in `data/labels.py`): `label19` (18 attacks + Benign), `label6`
  (Benign, Spoofing, Recon, MQTT, DoS, DDoS), `label2` (Benign vs Attack).
* **Splits** (`data/windows.py`): `official` (main result: official files; validation = last 15 % of each training
  file), `random` (pooled rows split at random, the protocol most published papers use; reported for context),
  `loao` (official split with one attack class removed from training, Section 6).

## 2. Data sanity checks (`audit.py`, `scripts/02_audit.py`)

A supporting step, not a contribution. Before any model is trained:
1. exact duplicates within train, within test and across the two are counted; duplicate windows are removed
   from training and test windows identical to a training window are removed (`dedup_windows`);
2. a depth-6 tree on one feature at a time flags possible shortcut features (macro-F1 ≥ 0.95);
   `data.drop_features: auto` removes them in all later steps;
3. scaling statistics (median/IQR, clip ±10) are fitted on training rows only.

## 3. Window encoder (`models/networks.py`, class `TRACE`)

A sample is a window of T = 16 consecutive flows inside one file segment (no window crosses a split boundary);
its label is that of its last flow. Training stride 8, test stride 1. Each flow gets a protocol token from its
`Protocol Type` value only (never from the file name).

1. **Feature tokenizer.** Feature j becomes w_j·x_j + b_j ∈ R^16; attention pooling over features gives one
   vector per flow, concatenated with the raw flow, projected to d = 128, plus the protocol embedding.
2. **BiGRU** (2 × 64) for local dynamics, then a **Transformer encoder** (2 layers, 4 heads) for longer context.
3. **Readout.** Attention pooling over time + the last step → window summary h.

## 4. Protocol-routed mixture of experts (C1; class `TRACEMoE`, `train.py`)

* **Router.** g(h) = softmax(W[h ; e_proto]), where e_proto embeds the last flow's protocol token.
* **Top-1 routing** (Switch style): k = argmax g(h); z = h + g_k(h)·E_k(h). Only expert k runs for the window.
  The gate value g_k scales the output so the router receives gradients.
* **Experts.** 4 residual feed-forward networks (LayerNorm → d → 2d → d). The intended split is MQTT,
  TCP/IP DoS-DDoS, Recon and ARP/benign traffic, but it is learned, not forced; H5 measures what is learned.
* **Load-balancing loss** L_bal = n·Σ_k f_k·P_k (f_k = share of windows sent to expert k, P_k = mean router
  probability), weight `model.moe.balance_weight` = 0.01, so no expert is left unused.
* **Hierarchical heads** on z: three linear heads for label2, label6, label19.
* **Loss** (`models/heads.py`): L = Σ_l w_l·FocalLA_l + λ_c·[KL(p6‖agg p19) + KL(p2‖agg p6)] + λ_b·L_bal,
  with a logit-adjusted focal loss (τ = 1, γ = 1), head weights (0.5, 0.5, 1.0), λ_c = 0.1.
  The focal exponent γ = 1 (not 2) avoids triple-weighting rare classes — logit adjustment,
  focal weighting, and class-balanced sampling all upweight small classes; stacking γ = 2 on
  top caused rare-class memorisation and overfitting by epoch 5 in a pilot run.
* **Training.** AdamW (1e-3), cosine schedule, class-balanced sampling with **replacement for
  classes smaller than 60k windows** (so Recon-Ping_Sweep, with ~45 windows at T = 16, still
  contributes 60k gradient updates per epoch instead of 45), **max 50 epochs, patience 10 on
  validation 19-class macro-F1**, dropout 0.2, mixed precision. The replacement sampling is
  essential: without it, rare classes receive 10³× fewer gradient updates per epoch than DDoS
  giants, which under-trains the fine 19-class head.
* **Ablation variants** (`ablation.variants`): `dense_expert` (one expert with 4× hidden size, same parameter
  count, no routing; H4), `protocol_routing` (fixed routing by protocol token), `no_experts` (encoder only),
  `no_balance_loss`, `no_hierarchy`, `plain_ce_loss`, `no_protocol_token`; plus T ∈ {1, 4, 8, 16, 32}.
* **Routing report** (`abstain.routing_report`), saved in every TRACE-MoE run: NMI between the chosen expert and
  the category/class, share of windows per expert, and expert counts per category (figure `expert_by_category`).

## 5. Robustness at 19 classes (C2; `attacks.py`, `scripts/06_robustness.py`)

* **Threat model.** The attacker wants attack traffic to be missed and can change only 18 timing and size
  statistics (`robustness.mutable_features`); flags, protocol indicators and counts stay fixed.
* **Attack.** PGD (20 steps) in scaled space with L∞ budget ε ∈ {0.05, 0.1, 0.25, 0.5}; every value is projected
  so its raw value stays ≥ 0. It maximises the 19-class loss and pushes the binary head to Benign. Only attack
  windows are perturbed. ASR = share of attack windows labelled Benign, overall and per attack class.
* **Transfer.** Black-box transfer matrix: windows crafted on each model are given to every other model;
  RF is attacked with these transfer examples.
* **Defence.** Every neural model (not only ours) is adversarially fine-tuned for 3 epochs on a 50/50 mix of clean
  and PGD windows; the curves are measured again. H6 compares the fine-tuned models at ε = 0.25.

## 6. Calibrated abstain layer and unknown attacks (C3; `abstain.py`, `scripts/07_unknown_attacks.py`)

* **Calibration.** A temperature is fitted on the validation windows (NLL, grid search), for every model incl. RF.
* **Reject rule.** The threshold is the validation confidence that keeps 95 % of the windows
  (`abstain.target_coverage`); below it the model abstains and raises an "unknown / uncertain" alert.
* **Reported per main run** (`metrics.json` → `abstain`): coverage, error on all windows, error on accepted
  windows (selective risk), macro-F1 on accepted windows, AURC, ECE before and after calibration, and the
  risk-coverage curve (figure `risk_coverage`).
* **Unknown attacks.** One class at a time (ARP_Spoofing, Recon-VulScan, MQTT-Malformed_Data, TCP_IP-DoS-UDP,
  MQTT-DDoS-Publish_Flood) is removed from training. The abstain score is the energy score of the 19-class head;
  the threshold is again set on validation (95 % of known windows accepted). Reported: AUROC and FPR@95%TPR of
  the energy score and of plain softmax confidence, share of unseen windows flagged unknown, false-alarm rate on
  known windows, and the share caught at all (flagged unknown or classified as Attack).

## 7. Baselines (`models/networks.py`, `models/classical.py`)

One per model family, each the winner in published CICIoMT2024 or special-issue work:

| Family | Model | Input | Why |
|---|---|---|---|
| Tree ensemble | RF (`rf`) | one flow | wins most often in the reviewed papers (e.g. Naeem 2024, Doménech 2026, Alkhodaidi 2026) |
| Recurrent | BiLSTM-DNN (`bilstm_dnn`) | features of one flow read as a sequence | best recurrent model at 19 classes (Benahmed 2025) |
| Transformer | Transformer (`transformer`) | the flow window | the family of the best leakage-aware model (Jodayree 2026) |
| Graph | TF-SAGE-lite (`tfsage_lite`) | the flows of a window as a k-NN graph with trust filtering | simplified re-implementation of TF-SAGE (Shieh et al., this special issue) |

All neural models share the heads, loss, sampler, early stopping, abstain layer and adversarial fine-tuning, so
the comparison isolates the architecture. RF uses `class_weight="balanced_subsample"` and the same abstain rule.

### Positioning against the special issue
| Paper | What it does | What we add |
|---|---|---|
| Alkhodaidi et al., CMC 2026;88(2):87 (CICIoMT2024) | data balancing with LR/RF/AdaBoost/DNN; 2/6/19 classes; one seed; accuracy, weighted F1 | a new architecture, 5 seeds, macro-F1, robustness, abstention, unknown attacks; compared by its reported numbers |
| Shieh et al., TF-SAGE, CMC 2026;89(2):73 (NF-ToN-IoT-v2, CICIIoT2025; not CICIoMT2024) | trust-filtered graph IDS; attack success rate and ECE | its stated future work: multiclass robustness, unseen attacks, multi-seed runs, adversarially trained baselines; TF-SAGE-lite is one of our baselines; compared on design only |

## 8. Efficiency (`efficiency.py`, `scripts/08_efficiency.py`)

Parameters, fp32 size, and median latency per window at batch 1 and 1024 on CPU and GPU. TRACE-MoE is timed in
eval mode, where only the selected expert runs. RF prediction time per 1k flows is in its `metrics.json`.

## 9. Comparison with published results (`configs/published_results.yaml`, `scripts/09_compare.py`)

The five closest CICIoMT2024 papers are compared by the numbers they report, without re-implementing them:
Jodayree et al. 2026 (Transformer, zero-day), Benahmed et al. 2025 (BiLSTM-DNN), Alkhodaidi et al. 2026 (this
special issue), Alkhattabi et al. 2026 (adversarial test) and Söğüt et al. 2026 (window-based, edge). The table
`published_comparison` shows each value with the paper's split and metric, labelled "reported in paper", next to
TRACE-MoE under the official and the random split. All values were checked against the
publisher pages on 2 Oct 2026. Alkhattabi et al. report their FGSM test on the SmartWard dataset, not CICIoMT2024,
so no number from that paper is tabulated; it is discussed in the text only (reserve: Al-Sharo et al. 2026, FedMamba).

## 10. Metrics and statistics (`metrics.py`, `compare.py`)

* **Primary metric:** 19-class macro-F1.
* **Also:** accuracy, balanced accuracy, weighted F1, macro PR-AUC, minimum class recall, ECE; at the binary level
  ROC-AUC, PR-AUC and recall at FPR ≤ 1 %; ASR; coverage, selective risk, AURC; AUROC and FPR@95%TPR for unknown attacks.
* **Seeds:** 42–46, mean ± std.
* **Tests:** Wilcoxon over (class, seed) pairs with Holm correction; Friedman + Nemenyi critical difference
  (Demšar, 2006); paired t-interval over seeds for H3.

## 11. Threats to validity (for the Discussion)

* Each CICIoMT2024 file holds one class, so every window holds one class.
* Published results come from different splits and metrics; they give context, not head-to-head evidence.
* The feature-space attack respects mutability and non-negativity but not full packet-level realisability.
* The BiGRU + Transformer encoder may make TRACE-MoE slower than the plain Transformer (H10 can fail on latency).

## 12. Traceability: from results to report to paper

1. Steps 01–09 write every number to `results/` and every action to `logs/`.
   * Every per-run `metrics.json` carries a reproducibility fingerprint: `run_id`, `started_at`/`finished_at`,
     `host`, `session`, `git_commit`, `git_dirty`, `code_sha256`, `data_sha256`, environment versions.
   * Every rerun is append-only: `trace_moe_official_T16_seed42/` is never overwritten; a second attempt lands
     in `trace_moe_official_T16_seed42__v2/`, a third in `__v3`, so no previous experiment is lost.
   * `results/registry.jsonl` is appended to once per run — one JSON line listing the run_id, dir, git commit,
     training time and headline macro-F1. It is the single ledger of every run ever done on the project.
   * `logs/<step>.log` is overwritten each invocation (current), but `logs/sessions/<session>_<step>_<host>.log`
     is appended — so the training history of every pipeline run is preserved.
   * Every aggregated table in `results/tables/` has a sibling `<table>.provenance.json` listing, for each row,
     the exact run folders, git commits and sessions that fed into it. The CSV also carries `source_runs` and
     `source_git_commits` columns for human-readable inspection.
2. **Technical report** (`10_technical_report.py` → `documents/report/technical_report.docx`). Every table (T1 …)
   and figure (F1 …) is read from a file in `results/`; its caption names that file, the script that produced it,
   that script's log, the git commit at report time, and (for aggregated tables) the provenance sidecar.
   `manifest.json` stores the list with a SHA-256 checksum of each source file; `hypotheses.json` freezes the
   H1–H10 verdicts with their criteria and evidence; `environment.json` freezes the versions used.
3. **Paper plan and draft** (`11_write_paper.py` → `documents/paper/`). The script stops if any result file
   changed after the report was written. It writes `paper_plan.docx` (claims allowed by the verdicts, storyline,
   which report item goes where, CMC checklist), then `paper.docx` and `paper.tex`, whose captions carry the
   report IDs and whose appendix lists every paper table/figure next to its report ID, source file, log and git
   commit. `traceability.json` is written alongside: one JSON entry per paper item linking it to its report ID,
   source, SHA-256 and git commit.
4. **Audit tool** (`scripts/verify_traceability.py`). Run any time (and at least after step 11). It confirms
   every table and figure in the report+paper resolves to a source file whose SHA-256 still matches, every
   aggregated table has its provenance sidecar, and every `results/*/*/metrics.json` is listed in
   `results/registry.jsonl`. `--strict` makes it exit non-zero on any failure.

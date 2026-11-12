# Hypotheses: project protocol and paper labels

The project's protocol fixed hypotheses P-H1 to P-H10 before the experiments (thresholds in `configs/default.yaml`,
decision rules in `src/trace_iomt/reporting.py`, `hypothesis_verdicts`). P-H11 was added with the Digital Twin study
(TWIN_PROTOCOL.md §8) and is post hoc. The code, configurations and result files (`hypotheses.json`,
`twin_h11.json`) use the protocol labels.

Each paper numbers its own hypotheses from H1 and reports only those it analyses (decided 2026-11-11, option A).

| Protocol | Short statement | Paper 1 (DT-TRACE-MoE, IEEE IoT-J) | Paper 2 (robustness) | Paper 3 (unseen attacks) |
|---|---|---|---|---|
| P-H1 | TRACE-MoE has higher 19-class macro-F1 than each baseline | H1 | | |
| P-H2 | Worst-class recall at least 5 points over the best baseline | H2 | | |
| P-H3 | Binary detection non-inferior to random forest (margin −0.5 points) | H3 | | |
| P-H4 | Routing helps: a dense expert of equal size is worse | H4 | | |
| P-H5 | Experts specialise by attack family: NMI(expert, category) ≥ 0.5 | H5 | | |
| P-H6 | Lower attack success rate than equally fine-tuned baselines at ε = 0.25 | | H1 | |
| P-H7 | Adversarial fine-tuning costs at most 1 point of clean macro-F1 | | H2 | |
| P-H8 | Unseen-attack AUROC ≥ 0.90 and above softmax confidence (leave-one-attack-out) | | | H1 |
| P-H9 | Abstention: TRACE-MoE has the lowest AURC | H6 | | |
| P-H10 | Edge budget: < 5 M parameters and faster than the Transformer at batch 1024 | H7 | | |
| P-H11 (post hoc) | The twin context raises 19-class macro-F1 over No-Twin on the device rows | H8 | | |

Papers 2 and 3 add their own new hypotheses after these; their labels are listed here when they are fixed.

# 5. Experiments

## 5.1 Experimental setup

We evaluated ReCoMER on multimodal conversational emotion recognition using four complementary evidence sources: a frozen formal test of the complete system, module-isolation studies for the three proposed components, corrected abstention experiments, and an exact-feature replication of the cRBEF expert. This separation was maintained throughout the analysis because the experiments use different data splits, numbers of random seeds and fusion scopes.

ReCoMER contains four functional parts. MHnoU is the history-aware main predictor. EvidenceRouter estimates sample-specific modality contributions from local modality evidence, cross-modal context and prediction statistics; it is evaluated in a frozen-backbone setting when used for module isolation. The history-abstention module lets real historical candidates compete with an explicit empty candidate and, when history is rejected, suppresses both the historical residual path and the raw historical-token path. cRBEF is an external class-probability expert. Its output is combined by the outer ReCoMER fusion module and is not fed back into the MHnoU modality router.

We used MELD, M3ED and IEMOCAP for classification and CMU-MOSEI for regression. Weighted F1 (WF1) was used for classification, with larger values indicating better performance; mean absolute error (MAE) was used for MOSEI, with smaller values indicating better performance. Innovation 1 and Innovation 2 used ten random seeds (7, 13, 17, 23, 29, 37, 43, 53, 71 and 101). Innovation 3 used three seeds under its corrected protocol. The frozen formal ReCoMER test consisted of three paired runs, with the cRBEF checkpoint fixed at seed 17. The exact-feature M3ED queue used ten seeds for the MHnoU controls and five seeds for the standalone cRBEF replication.

For the formal M3ED runs, the model used a 192-dimensional internal representation, dropout 0.15, AdamW with learning rate 2 × 10−4 and weight decay 0.01, batch size 256, warm-up followed by cosine scheduling, a maximum of 20 epochs and early stopping based on the selection split. The exact M3ED HuBERT pack contained 17,427 training samples, 2,821 validation samples and 4,201 test samples, with feature dimensions 768, 2,048 and 342 for text, audio and video, respectively. The remaining innovation-specific configurations followed their recorded run manifests. Test data were not used for module selection; the formal test was opened only after the available model and fusion controls had been frozen.

The ten-seed confidence intervals in the module-isolation studies are paired seed-level intervals on fixed validation protocols. The confidence intervals in the formal ReCoMER table are dialogue-bootstrap intervals over test conversations. These uncertainty definitions are not interchangeable.

## 5.2 Main comparison on the frozen M3ED test

We first compared the complete ReCoMER system with its available fusion controls under the frozen M3ED test protocol. The results are shown in Table 1.

**Table 1. Frozen M3ED formal test comparison.** WF1 is reported in percent; values are higher-is-better. The cRBEF checkpoint was fixed at seed 17, whereas the ReCoMER and equal-weight values summarize the three paired runs.

| Model or fusion rule | Test WF1 (%) |
|---|---:|
| AV | 48.71 |
| TAV | 57.03 |
| Equal-weight fusion | 57.39 ± 0.37 |
| **ReCoMER** | **58.00 ± 0.34** |
| cRBEF (fixed seed 17) | 58.60 ± 0.00 |

ReCoMER improved over the matched equal-weight control by 0.607 percentage points (dialogue-bootstrap 95% CI, 0.250–1.005). The learned outer fusion therefore provided a measurable gain over uniform combination. However, the fixed cRBEF expert remained slightly higher than the complete ReCoMER result. Thus, the present test supports the usefulness of external-expert fusion, but does not support the stronger claim that the learned fusion is better than every individual expert.

For context, the original M3ED paper reported a full-modality DialogueRNN WF1 of 51.66%. Because the current ReCoMER result uses different feature extraction and a different implementation protocol, this comparison is contextual rather than a strict same-protocol SOTA claim.

## 5.3 Innovation 1: historical feedback as conditional context compensation

We next asked whether historical feedback improves the current utterance prediction. To isolate this question, we compared history-enabled and no-history models under uniform modality weights. Table 2 reports the paired ten-seed effects.

**Table 2. Historical feedback under the ten-seed validation protocols.** Positive values indicate an improvement in WF1 or a reduction in MAE, according to the metric direction.

| Dataset and comparison | Metric | Mean effect | 95% CI | Same-direction seeds |
|---|---|---:|---:|---:|
| M3ED: uniform_h − uniform_nh | WF1 ↑ | +0.004774 | [+0.001105, +0.008444] | 7/10 |
| MELD: uniform_h − uniform_nh | WF1 ↑ | −0.001534 | [−0.008234, +0.005165] | 5/10 |
| MOSEI: mosei_nh − mosei_h | MAE ↓ | +0.003361 | [+0.000021, +0.006701] | 8/10 |

The effect of history was therefore conditional on the data regime. The clearest positive result occurred on MOSEI, where historical feedback reduced MAE. The M3ED effect was positive but did not reach the pre-specified 8/10 same-direction criterion, whereas the aggregate MELD interval crossed zero. The available front/late/closed-loop controls also did not establish an independent gain from the ordering “feedback followed by reassessment”: FULL − ONE_GATE was +0.000042, FULL − UNIFORM_QUERY was +0.000114 and FULL − GLOBAL_LATE was −0.000765, with all corresponding intervals overlapping zero. We therefore interpret the current evidence as conditional historical-feedback fusion rather than proof that a particular closed-loop order is universally superior.

## 5.4 Innovation 2: Shapley-anchored contribution routing

We then isolated the contribution-routing component. The encoder, history module, solo heads and joint head were frozen, while only the EvidenceRouter utility head was optimized on a fixed uniform backbone. The router received exact per-sample Shapley supervision together with the task objective and a pairwise ranking term. These results measure the independent routing effect and should not be confused with a final full-SABER test.

**Table 3. EvidenceRouter under the fixed-backbone ten-seed protocol.**

| Dataset | Uniform | EvidenceRouter | Change | 95% CI | Positive seeds |
|---|---:|---:|---:|---:|---:|
| MOSEI MAE ↓ | 0.566134 | 0.559393 | −0.006741 | [−0.008536, −0.004946] | 10/10 |
| MELD WF1 ↑ | 0.577705 | 0.581131 | +0.003426 | [+0.000938, +0.005914] | 8/10 |
| M3ED WF1 ↑ | 0.602071 | 0.598774 | −0.003298 | [−0.005441, −0.001154] | 2/10 |

EvidenceRouter consistently improved MOSEI MAE and produced a smaller positive effect on MELD WF1. In contrast, the M3ED E2 representation did not benefit from the router. The MOSEI result was also metric-dependent: MAE decreased, whereas RMSE and Pearson correlation did not improve in the same experiment. These observations indicate that Shapley-anchored routing can be useful when the modalities provide complementary evidence, but its benefit is not independent of the feature regime.

## 5.5 Innovation 3: sentence-level history abstention

The third study tested whether the model could compete between real historical candidates and an explicit empty candidate. B1 denotes the original empty-candidate design. E5 applies utility calibration before softmax normalization, E6 additionally uses semantic history features, and E7 replaces softmax with sparsemax. All four variants retain the same candidate set and are evaluated under the corrected history-path implementation.

**Table 4. Corrected history-abstention variants.** Values are mean ± population standard deviation over three seeds.

| Dataset | Metric | B1 | E5 | E6 | E7 |
|---|---|---:|---:|---:|---:|
| MELD | inner-dev WF1 ↑ | 0.5967 ± 0.0069 | 0.5970 ± 0.0090 | 0.5968 ± 0.0088 | 0.5957 ± 0.0091 |
| M3ED | valid WF1 ↑ | 0.561040 ± 0.0012 | 0.558256 ± 0.0008 | 0.558256 ± 0.0008 | 0.558650 ± 0.0023 |
| IEMOCAP | valid WF1 ↑ | 0.724835 ± 0.0022 | 0.730898 ± 0.0065 | 0.730898 ± 0.0065 | 0.730378 ± 0.0058 |
| MOSEI | valid MAE ↓ | 0.563965 ± 0.0026 | 0.562650 ± 0.0028 | 0.562651 ± 0.0028 | 0.563299 ± 0.0028 |

Relative to B1, E5 improved IEMOCAP WF1 by 0.006063 and reduced MOSEI MAE by 0.001315. It remained essentially unchanged on MELD (+0.0003 WF1) and decreased M3ED WF1 by 0.002784. E6 was numerically indistinguishable from E5, and E7 showed no stable advantage over softmax. The corrected results therefore support a dataset-dependent abstention effect rather than universal improvement.

## 5.6 Integrated ablations and information-flow checks

The frozen ReCoMER test also enabled paired ablations of the integrated model. Table 5 reports the difference between the complete model and each control.

**Table 5. Integrated ReCoMER ablations.** Values are Full − control in percentage points of test WF1.

| Control | Full − control | Dialogue-bootstrap 95% CI |
|---|---:|---:|
| Equal-weight fusion | +0.607 | [0.250, 1.005] |
| No relative contribution C | +0.013 | [−0.043, 0.069] |
| No Shapley supervision | −0.050 | [−0.281, 0.158] |
| Unbounded softmax weights | −0.036 | [−0.272, 0.172] |
| No history abstention | −0.487 | [−0.911, −0.091] |
| No contribution-token feedback | −0.015 | [−0.290, 0.248] |
| No history-residual feedback | −0.056 | [−0.160, 0.047] |

The complete model was better than equal weighting, and removing history abstention was harmful under this test protocol. The intervals for relative-contribution, Shapley-supervision and bounded-weight controls overlapped zero; these comparisons are therefore reported as mechanism indications rather than independent significance claims.

The cRBEF/MHnoU splice passed the available identity and information-flow checks. Replacing cRBEF changed the expert and final fused probabilities while leaving MHnoU probabilities and its modality-relative contribution unchanged. This supports the interpretation of cRBEF as an external probability expert rather than an accidental input to the MHnoU router. The CR17 adapter qualification also produced zero hidden replay error on 4,201 test rows.

## 5.7 Exact-feature replication and mechanism diagnostics

To separate model behavior from feature-protocol effects, we ran an additional exact M3ED queue using the official train/validation/test split and the exact HuBERT feature pack. The queue was deliberately labeled as a supplementary protocol because its five arms were MHnoU controls with the three integrated innovation flags disabled.

**Table 6. Exact-feature M3ED ten-seed test controls.**

| Arm | Test WF1 |
|---|---:|
| uniform_nohist (B0) | 0.535269 ± 0.005824 |
| uniform_history | 0.534562 ± 0.004858 |
| evidence_closed | 0.537279 ± 0.004118 |
| evidence_mpath | 0.539917 ± 0.006380 |
| uniform_abstain | 0.533945 ± 0.004854 |

Under the same exact-feature protocol, the best arm was evidence_mpath, whereas B0 reached 0.535269. These values are useful for controlled exact-feature comparisons but should not be merged with the 58.00% formal ReCoMER result, which uses the complete model and a different feature protocol.

We also retrained the cRBEF-style evidence heads on the exact T/A/V features. Gate parameters were selected on the validation split and frozen before test evaluation. The five-seed test WF1 values were 0.421594 ± 0.002423 for T-only, 0.531980 ± 0.001816 for AV-only, 0.528735 ± 0.005345 for ungated TAV, 0.526567 ± 0.005946 for log-evidence fusion and 0.548599 ± 0.004454 for gated CRBEF. This replication confirms that the expert-fusion arithmetic remains functional under the exact feature pack; it is not a replacement for the frozen formal ReCoMER result.

Finally, the ten-seed mechanism diagnostic on the evidence_mpath checkpoints measured history availability, keep probability, hard keep rate, abstention mass and history override sensitivity. On the exact test split, history was available for 0.952868 of samples, the mean keep probability was 0.950279, the hard keep rate was 0.952868, the mean abstention mass was 0.202451 and forcing the history decision changed deployed logits by 0.046409 on average. These values demonstrate that the history path participates in deployment decisions, but they do not by themselves establish a performance gain.

## 5.8 Sensitivity, failure boundaries and reproducibility

The results revealed three scope conditions. First, M3ED is strong at the utterance level under the tested representation, so history feedback and contribution routing can have small or negative marginal effects. Second, MELD exposes a continuation-versus-transition trade-off: historical context can help continuation utterances but can also propagate an incorrect prior state across an emotional transition. Third, MOSEI benefits most consistently from historical compensation, but the direction is metric-specific and should not be generalized from MAE to every regression or classification measure.

The complete run package records model hashes, checkpoints, per-seed histories, Shapley diagnostics, modality weights, history decisions, token masks and split guards. The exact-feature queue provides a separate B0 and mechanism reference. The following analyses remain supplementary or pending rather than being silently inferred: a complete four-dataset B0 matrix, same-label versus different-label subgroup results, a full three-seed diagnostic matrix for every abstention variant, a train-only utility-cache comparison and a complete same-protocol external SOTA table.

## 5.9 Summary of experimental claims

The experiments support four bounded conclusions. First, ReCoMER improved over equal-weight fusion on the frozen M3ED formal test, although the fixed cRBEF expert remained slightly stronger. Second, historical feedback can compensate for weak utterance-level evidence, most clearly on MOSEI, but its effect is conditional on context. Third, Shapley-anchored contribution routing and sentence-level history abstention showed positive effects on selected datasets but did not transfer uniformly to M3ED. Fourth, the information-flow audit supports cRBEF as an external expert in the outer fusion. Together, these results establish a reproducible and interpretable fusion framework while making its data-dependent failure boundaries explicit.

## Data and audit sources

- Formal ReCoMER results and paired test ablations: `05_Experiments_ReCoMER_整合稿_20261004.md` and `最终代码/实验结果/ReCoMER_正式测试与消融_20261004/`.
- Innovation 1: `文档/创新点1_闭环实验数据整理_20261004.md`.
- Innovation 2: `文档/创新点2_实测贡献门控_交接版.md`.
- Innovation 3: `文档/创新点3_实验数据整理_20261004.md` and `文档/创新点3_论文Experiments数据包_20261004.md`.
- Exact-feature M3ED and mechanism diagnostics: `/root/autodl-tmp/paper_experiments_exact_20261005/`.
- Exact-feature cRBEF replication: `/root/autodl-tmp/crbef_unified_20261005/summary.json`.

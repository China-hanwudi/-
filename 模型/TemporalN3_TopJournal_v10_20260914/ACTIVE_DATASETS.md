# Active Dataset Contract

| Dataset | Task | Train/valid source | Test policy | Selection |
|---|---|---|---|---|
| M3ED | 7-class emotion | `M3ED/packed/{train,valid}.pt` | sealed | valid Weighted-F1, Macro-F1, Accuracy, loss |
| CMU-MOSEI | scalar sentiment | `MOSEI/packed/{train,valid}.pt` | sealed | valid MAE |
| CH-SIMS_v2 | scalar sentiment | `CH-SIMS_v2/v7_packed/{train,valid}.pt` | sealed | valid MAE |

No legacy dataset is used as a baseline or comparison source in v10. Any historical legacy code is outside the active package and must not be imported by training scripts. Published SOTA rows are admitted only after split, labels, modalities, and citation are independently verified.

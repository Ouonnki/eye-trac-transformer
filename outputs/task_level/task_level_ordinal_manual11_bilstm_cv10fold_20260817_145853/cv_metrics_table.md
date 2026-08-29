# Task-level 10-fold CV metrics

| Split | ACC | Macro-F1 | G-Mean | AUC-PR | QWK |
|---|---:|---:|---:|---:|---:|
| Val (同分布) | 0.6405 ± 0.0317 | 0.5676 ± 0.0349 | 0.6399 ± 0.0403 | 0.5748 ± 0.0417 | 0.4774 ± 0.0533 |
| Test1 (新被试+旧题) | 0.5953 ± 0.0279 | 0.5064 ± 0.0313 | 0.5687 ± 0.0408 | 0.5060 ± 0.0396 | 0.3961 ± 0.0522 |
| Test2 (旧被试+新题) | 0.5592 ± 0.0898 | 0.4892 ± 0.0824 | 0.5106 ± 0.0917 | 0.5732 ± 0.0459 | 0.3808 ± 0.0792 |
| Test3 (新被试+新题) | 0.5314 ± 0.0625 | 0.4678 ± 0.0652 | 0.4725 ± 0.0845 | 0.5360 ± 0.0279 | 0.3621 ± 0.0613 |

Definitions: G-Mean = geometric mean of per-class recall; AUC-PR = macro one-vs-rest average precision; QWK = quadratic weighted Cohen kappa.

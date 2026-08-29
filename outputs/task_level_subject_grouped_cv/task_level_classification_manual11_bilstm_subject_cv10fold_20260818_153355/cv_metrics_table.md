# Task-level 10-fold CV metrics

| Split | ACC | Macro-F1 | G-Mean | AUC-PR | QWK |
|---|---:|---:|---:|---:|---:|
| Val (同分布) | 0.6270 ± 0.0844 | 0.4956 ± 0.1035 | 0.5080 ± 0.1314 | 0.4956 ± 0.0885 | 0.3303 ± 0.1662 |
| Test1 (新被试+旧题) | 0.5997 ± 0.0282 | 0.4838 ± 0.0432 | 0.5084 ± 0.0696 | 0.4832 ± 0.0499 | 0.3476 ± 0.0868 |
| Test2 (旧被试+新题) | 0.4822 ± 0.0388 | 0.4034 ± 0.0430 | 0.4054 ± 0.0683 | 0.5224 ± 0.0596 | 0.2694 ± 0.0657 |
| Test3 (新被试+新题) | 0.4723 ± 0.0292 | 0.3973 ± 0.0337 | 0.3776 ± 0.0601 | 0.4917 ± 0.0330 | 0.2879 ± 0.0531 |

Definitions: G-Mean = geometric mean of per-class recall; AUC-PR = macro one-vs-rest average precision; QWK = quadratic weighted Cohen kappa.

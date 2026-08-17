# Uniform random baseline (10-fold)

| Split | ACC | Macro-F1 | G-Mean | AUC-PR | QWK |
|---|---:|---:|---:|---:|---:|
| Val (同分布) | 0.3385 ± 0.0288 | 0.2860 ± 0.0257 | 0.3352 ± 0.0381 | 0.3374 ± 0.0081 | 0.0083 ± 0.0463 |
| Test1 (新被试+旧题) | 0.3432 ± 0.0095 | 0.2884 ± 0.0082 | 0.3481 ± 0.0132 | 0.3370 ± 0.0031 | 0.0092 ± 0.0217 |
| Test2 (旧被试+新题) | 0.3346 ± 0.0139 | 0.2853 ± 0.0137 | 0.3304 ± 0.0210 | 0.3339 ± 0.0040 | -0.0076 ± 0.0313 |
| Test3 (新被试+新题) | 0.3342 ± 0.0220 | 0.2903 ± 0.0215 | 0.3353 ± 0.0282 | 0.3342 ± 0.0062 | 0.0066 ± 0.0427 |

Random strategy: uniform class sampling, P(class)=1/3, seed=42.
AUC-PR uses one-hot random class scores and macro one-vs-rest average precision.

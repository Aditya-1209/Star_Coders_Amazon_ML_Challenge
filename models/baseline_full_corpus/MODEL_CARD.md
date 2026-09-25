# Frozen CPU baseline

CatBoostClassifier 1.2.10, trained from scratch on challenge data. The CatBoost
library is Apache-2.0 licensed; no external pretrained weights or business lookup
data are used. The classifier has 450 depth-6 trees and 31 numeric features.

- Training/tuning/holdout anchors: 4,198 / 898 / 904, sampled with seed 42.
- Retrieval corpus: all 10,320,219 training Source 2/3 records.
- Holdout macro F0.5: 0.9255490622573501.
- Decision threshold: 0.605; 20 candidates per retrieval channel; posting budget 3,000.
- France has no training labels; its test accuracy is unknown.
- No full-test or leaderboard score is claimed.

The two files required for prediction must remain together:

| File | SHA-256 |
| --- | --- |
| `model.cbm` | `14ce84284c138c7c155e1bd38df5a388a369be903d60d2d7be6a6c8b6b91b52c` |
| `config.json` | `790f7631d8f5ce58fcb75aee85903c88990c3d93c487121e4220b51bf11574e6` |

`metrics.json` records the completed full-corpus development experiment.
The repository includes the code and commands to reproduce training, while
these frozen weights allow inference without repeating it.

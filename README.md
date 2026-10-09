# Fraudster Detection

Identifying fraudulent users of a (fictional) online bank from their profile and transaction history.
Data science take-home assignment from StrataScratch.

The analysis lives in the notebook [`fraud_detection.ipynb`](fraud_detection.ipynb), organised in 6 steps:
data checks → exploratory analysis → feature engineering → modelling → evaluation → improvements.

```
fraud_detection.ipynb        analysis notebook
fraud_detection/
  features.py                user-level feature engineering (profile, transaction mix, sequences)
  evaluation.py              alert stats, cost-based threshold, PSI drift
tests/                       unit tests for both modules (pytest)
```

## Results at a glance

| Model | Test PR-AUC | Recall | Precision |
|---|---|---|---|
| Gradient Boosting — base features, F2 threshold | 0.83 | 0.76 | 0.76 |
| Gradient Boosting — + sequence features, calibrated, cost-based threshold | **0.86** | **0.90** | 0.52 |

Random baseline PR-AUC = 0.03 (fraud rate). More realistic estimates:

- **Time-based split** (train on older accounts, test on newer ones): PR-AUC **≈ 0.76**
- **Early detection**, using only the first 30 days of activity: PR-AUC **≈ 0.68**

## Data

| File | Content |
|---|---|
| `users.csv` | 9,944 users — profile, KYC status, target `IS_FRAUDSTER` (3% fraudsters) |
| `transactions.csv` | 688,651 transactions (type, state, amount, entry method, source, merchant…) |
| `countries.csv`, `currency_details.csv` | Code dictionaries |

The data files are **not included** in this repository. Place the four CSV files at the root of the project before running the notebook.

## Approach

1. **Data checks.** I found a **target leak**: `STATE` is `LOCKED` for every fraudster and `ACTIVE` for everyone else, so it is excluded. I also dropped about 50k transactions from unknown users.
2. **Exploratory analysis.** Fraudsters follow a *cash-in → cash-out* (money mule) pattern. They make more top-ups, ATM withdrawals and bank transfers, have more declined transactions and more manual card entry, and use the `MINOS` source heavily.
3. **Feature engineering.** I aggregated transactions into 47 user-level features: transaction mix (shares), amounts, timing, geography, and cash-out share. Each feature is justified in the notebook. I excluded the account state and the activity span, because both are shaped by the account lock.
4. **Modelling.** I used a stratified 60/20/20 split and 5-fold cross-validation, with PR-AUC as the main metric. I compared Logistic Regression, Random Forest and Histogram Gradient Boosting, all with class weighting. Gradient Boosting was selected.
5. **Evaluation.** I chose the threshold on the validation set (F2), measured permutation importance and checked that the model still works without the `MINOS` feature. The test set is never used to choose anything: it only reports the Step 5 model, then the improved one. I also looked at "top-k" alert lists and analysed the fraudsters the model misses.
6. **Improvements:**
   - sequence features: top-up → cash-out speed, velocity, declined streaks (with explicit "not applicable" flags);
   - early-detection windows and a temporal split;
   - isotonic calibration and a cost-based threshold;
   - SHAP explanations;
   - PSI drift monitoring.

## Main limitations

- Few fraudsters (298 in total, about 60 per evaluation set), so metrics are uncertain by about ±0.05 PR-AUC.
- Labels only cover fraudsters that were *caught*, so the model partly learns past detection rules.
- Several key features drift between older and newer accounts, so a deployed model needs monitoring and retraining.
- The cost ratio used for the threshold is an assumption, not a business figure.
- The kept category levels were chosen during an exploratory analysis that saw all users (a mild leak that only affects which columns exist).

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb   # or open it in VS Code
```

Tests:

```bash
pip install -r requirements-dev.txt
pytest
```

Python 3.12+ recommended. Results are reproducible (`RANDOM_STATE = 42`).

## Author

Soro Amidou

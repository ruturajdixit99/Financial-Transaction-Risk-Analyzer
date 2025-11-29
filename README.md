# Financial Transaction Risk Analyzer

This project simulates card transactions, builds **risk features** on top of them, and trains both a **supervised classifier** and an **unsupervised anomaly detector** to flag high-risk customers.

It is designed as a *sandbox* for experimenting with:
- rules-based fraud signals,
- feature engineering on transaction logs,
- supervised risk scoring,
- unsupervised anomaly detection (Isolation Forest).

---

## 1. Project Structure

Financial_Transaction_Risk_Analyzer/
├── data/
│ ├── customers.csv # simulated customer master data
│ ├── transactions.csv # simulated raw transactions
│ ├── customer_features.csv # per-customer aggregated features
├── src/
│ ├── simulate_transactions.py # customer + transaction simulation
│ ├── rules_engine.py # rule-based risk scoring per txn
│ ├── build_features.py # aggregates transaction features
│ ├── train_risk_model.py # risk model + anomaly detection
└── README.md

markdown
Copy code

---

## 2. Data Simulation

### 2.1 Customer Simulation

`simulate_customers()` creates a population of **1,000 customers** with:
- `customer_id`
- `created_at` (account creation date in 2023)
- `country` (US, UK, IN, DE, SG)
- `risk_segment` (low / medium / high, with ~10% high-risk) :contentReference[oaicite:0]{index=0}  

This gives a simple way to test how risk varies by segment and geography.

### 2.2 Transaction Simulation

For each customer, `simulate_transactions()` generates ~80 transactions on average: :contentReference[oaicite:1]{index=1}  

- **Fields:**
  - `txn_ts`: timestamp (2024)
  - `amount`: transaction amount (gamma-distributed, long-tailed)
  - `merchant_cat`: grocery, electronics, entertainment, gambling, crypto, travel
  - `channel`: online, POS, ATM
  - `country`: includes extra high-risk countries (NG, RU)
  - `is_chargeback`: ~2% chargeback probability

- **Injected suspicious behavior for `risk_segment = "high"`:**
  - occasional *very large amounts* (3–10× multiplier),
  - increased chance of `merchant_cat = "crypto"`,
  - increased chance of `country = "NG"`.

This design ensures that high-risk customers exhibit **heavier tails in amount**, more **crypto / cross-border** behavior, and more **chargebacks**, which the model can learn to exploit.

**Example snippet from a run:**

```text
   customer_id     txn_ts  amount currency   merchant_cat channel country  is_chargeback
0            1 2024-01-07   30.19      USD  entertainment     atm      SG          False
1            1 2024-01-07  105.66      USD    electronics     pos      NG           True
2            1 2024-01-25  217.86      USD        grocery  online      IN          False
3            1 2024-01-01  227.34      USD         travel     pos      RU          False
4            1 2024-01-03   81.76      USD         crypto  online      RU           True
We already see:

Cross-border activity (NG, RU)

Crypto & entertainment

Chargebacks mixed in → exactly the patterns we want a risk system to pay attention to.

3. Rules Engine (Per-Transaction Risk Signals)
rules_engine.py computes lightweight rule flags for every transaction: 
rules_engine


rule_large_amount → amount > 500

rule_very_large_amount → amount > 2000

rule_suspicious_country → country in {NG, RU, PK}

rule_high_risk_merchant → merchant in {gambling, crypto}

rule_night_txn → transactions between 00:00–05:00

Each transaction gets a rules_score:

python
Copy code
rules_score = (
    1.0 * rule_large_amount +
    2.0 * rule_very_large_amount +
    1.5 * rule_suspicious_country +
    1.5 * rule_high_risk_merchant +
    1.0 * rule_night_txn
)
This is a simple, interpretable, point-based risk score, capturing:

amount risk,

geographic risk,

merchant-category risk,

time-of-day risk.

4. Customer-Level Feature Engineering
build_features.py aggregates transaction-level data into per-customer features: 
build_features


Computed features include:

Volume & value

txn_count – number of transactions

total_amount – total spend

max_amount, mean_amount – severity and average ticket size

Rule-based behavioral stats

rules_score_mean – average rule score across all txns

night_txn_ratio – share of night transactions

high_risk_mcc_ratio – share of high-risk MCCs (gambling/crypto)

suspicious_country_ratio – share of txns to suspicious countries

Loss behavior

chargeback_rate – mean of is_chargeback

Example row from the output:

text
Copy code
   customer_id  txn_count  total_amount  max_amount  mean_amount  rules_score_mean  night_txn_ratio  high_risk_mcc_ratio  suspicious_country_ratio  chargeback_rate
0            1         77       9445.76      454.35   122.672208          1.798701              1.0             0.272727                  0.259740         0.038961
From this one line we can tell:

this customer transacts a lot (77 txns, ~9.4k total),

every transaction is at night (night_txn_ratio = 1.0),

~27% of txns are gambling/crypto,

~26% hit suspicious countries,

~4% chargeback rate.

Even before machine learning, this already looks like a high-risk profile.

5. Risk Labeling Strategy
train_risk_model.py creates a synthetic risk label (risk_label) from the engineered features: 
train_risk_model


Compute percentile ranks:

cb_rank = rank of chargeback_rate

rules_rank = rank of rules_score_mean

Combine into a continuous risk_score:

python
Copy code
risk_score = 0.6 * cb_rank + 0.4 * rules_rank
Chargebacks are weighted slightly more heavily than rules.

Define risk_label = 1 for the top 20% of customers by risk_score.

From the run:

text
Copy code
Label distribution:
0    800
1    200
So 20% of the 1000 customers are labeled as high risk (1) and 80% as low/normal risk (0).

This mimics a realistic risk system:

most users are low risk,

a small, high-risk tail receives enhanced monitoring.

6. Supervised Risk Model
6.1 Model
Algorithm: RandomForestClassifier

class_weight="balanced" to handle 80/20 imbalance

Trained on all engineered features (except customer_id, risk_score) 
train_risk_model


6.2 Performance
From train_risk_model.py run:

text
Copy code
ROC-AUC: 0.9957

              precision    recall  f1-score   support

           0       0.97      0.98      0.97       240
           1       0.91      0.87      0.89        60

    accuracy                           0.96       300
   macro avg       0.94      0.92      0.93       300
weighted avg       0.96      0.96      0.96       300
6.3 Interpretation
AUC ~ 0.996 → the model almost perfectly separates high-risk vs low-risk customers (which makes sense, because labels are derived from the same signals).

Recall for high risk = 0.87 → the model correctly detects 87% of risky customers.

Precision for high risk = 0.91 → when the model flags someone as high risk, it is right ~91% of the time.

Overall accuracy = 96%.

In real life, labels would be noisier and performance would be lower. But this sandbox demonstrates that:

With well-designed features (rules + chargebacks), a risk model can be highly discriminative and prioritize the highest-risk customers effectively.

7. Unsupervised Anomaly Detection (Isolation Forest)
To complement the supervised model, the project also runs an IsolationForest on the same feature space:

python
Copy code
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_train)
anomaly_scores = -iso.score_samples(X_test)
An anomaly_score is added to each customer in the test set, and the top 10 most anomalous customers are printed: 
train_risk_model


Example output:

text
Copy code
Top 10 anomalous customers (by IsolationForest):
     customer_id  txn_count  total_amount  chargeback_rate  rules_score_mean  risk_label  anomaly_score
870          871         75      23468.81         0.026667          2.640000           1       0.664753
539          540         59      13709.73         0.000000          2.601695           0       0.664117
369          370         68      14363.53         0.044118          2.573529           1       0.657109
494          495         90      21855.05         0.033333          2.683333           1       0.656022
...
7.1 What this tells us
High total spend + high rules_score_mean customers float to the top.

Some are already labeled risk_label = 1 (the supervised system caught them).

Interesting: others are risk_label = 0 but still highly anomalous (e.g., customer 540 with very high rules_score but zero chargebacks).

This is exactly why combining supervised risk scoring with unsupervised anomaly detection is powerful:

Supervised model → catches “known” risk patterns (chargebacks + rule-based signals).

Isolation Forest → surfaces unusual but not yet loss-making customers that might become risky later.

In production, those would be great candidates for manual review or tighter limits.

8. How to Run
From src/:

Generate customers & transactions

bash
Copy code
python simulate_transactions.py
This writes customers.csv and transactions.csv into data/ and prints sample rows.

Build per-customer features

bash
Copy code
python build_features.py
This creates customer_features.csv with aggregated features.

Train risk model & run anomaly detection

bash
Copy code
python train_risk_model.py
This:

generates risk_label and risk_score,

trains the Random Forest risk classifier,

prints classification metrics,

fits Isolation Forest,

prints the top anomalous customers.

9. Key Takeaways & Extensions
Key Takeaways

Rules + chargeback rate are enough to build a strong synthetic risk label.

The Random Forest model reproduces the label with near-perfect AUC, showing that:

Transaction behavior,

rules-derived features, and

loss behavior (chargebacks)
carry enough signal to prioritize high-risk customers.

IsolationForest adds a second line of defense, flagging customers who look “weird” even if they have not yet produced losses.

Possible Extensions

Add device fingerprints, IP risk, velocity checks.

Separate domestic vs cross-border transaction behavior.

Use time windows (last 7/30/90 days) for recency-aware risk.

Log and visualize model outputs in a dashboard (e.g., Streamlit).

Calibrate risk scores into interpretable probabilities or risk bands (Low / Medium / High / Critical).

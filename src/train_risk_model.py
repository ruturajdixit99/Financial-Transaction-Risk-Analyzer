import os
import pandas as pd
import numpy as np  # still imported if you later need it
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

if __name__ == "__main__":
    # Use os.path.join for better path handling across operating systems
    data_path = os.path.join(
        r"D:\Projects\Finance\Financial_Transaction_Risk_Analyzer\data\customer_features.csv"
    )
    feats = pd.read_csv(data_path)

    # ==========================
    # 1) LABEL GENERATION
    # ==========================
    # Create a continuous risk score based on chargebacks + rules-based score
    cb_rank = feats["chargeback_rate"].rank(pct=True)          # higher = more risky
    rules_rank = feats["rules_score_mean"].rank(pct=True)      # higher = more risky

    # Weighted combination (tune 0.6 / 0.4 if needed)
    feats["risk_score"] = 0.6 * cb_rank + 0.4 * rules_rank

    # Define "high risk" as top 20% by risk_score
    high_risk_cutoff = feats["risk_score"].quantile(0.80)
    feats["risk_label"] = (feats["risk_score"] >= high_risk_cutoff).astype(int)

    print("Label distribution:\n", feats["risk_label"].value_counts())

    target = "risk_label"
    feature_cols = [c for c in feats.columns if c not in ["customer_id", target, "risk_score"]]

    X = feats[feature_cols]
    y = feats[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # ==========================
    # 2) CLASS BALANCE CHECK
    # ==========================
    unique_classes_train = y_train.unique()
    if len(unique_classes_train) < 2:
        print(f"Error: Training data only contains one class: {unique_classes_train}")
        print("Cannot train a binary classifier. Check data generation logic or input CSV.")
        print(y_train.value_counts())
        raise SystemExit(1)

    # ==========================
    # 3) SUPERVISED RISK CLASSIFIER
    # ==========================
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print(classification_report(y_test, y_pred))

    # ==========================
    # 4) UNSUPERVISED ANOMALY DETECTION (IsolationForest)
    # ==========================
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_train)
    anomaly_scores = -iso.score_samples(X_test)

    feats_test = feats.loc[X_test.index].copy()
    feats_test["anomaly_score"] = anomaly_scores

    print("\nTop 10 anomalous customers (by IsolationForest):")
    print(
        feats_test.sort_values("anomaly_score", ascending=False).head(10)[
            ["customer_id", "txn_count", "total_amount", "chargeback_rate",
             "rules_score_mean", "risk_label", "anomaly_score"]
        ]
    )

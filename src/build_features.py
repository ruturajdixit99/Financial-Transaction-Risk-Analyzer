import pandas as pd
from rules_engine import apply_rules


def build_transaction_features(txn_path=r"D:\Projects\Finance\Financial_Transaction_Risk_Analyzer\data\transactions.csv"):
    df = pd.read_csv(txn_path, parse_dates=["txn_ts"])
    df = apply_rules(df)

    # aggregate per customer + recent window
    df["txn_date"] = df["txn_ts"].dt.date

    agg = df.groupby("customer_id").agg(
        txn_count=("txn_id", "count") if "txn_id" in df.columns else ("amount", "count"),
        total_amount=("amount", "sum"),
        max_amount=("amount", "max"),
        mean_amount=("amount", "mean"),
        rules_score_mean=("rules_score", "mean"),
        night_txn_ratio=("rule_night_txn", "mean"),
        high_risk_mcc_ratio=("rule_high_risk_merchant", "mean"),
        suspicious_country_ratio=("rule_suspicious_country", "mean"),
        chargeback_rate=("is_chargeback", "mean"),
    ).reset_index()

    return agg


if __name__ == "__main__":
    feats = build_transaction_features()
    feats.to_csv("D:\Projects\Finance\Financial_Transaction_Risk_Analyzer\data\customer_features.csv", index=False)
    print(feats.head())

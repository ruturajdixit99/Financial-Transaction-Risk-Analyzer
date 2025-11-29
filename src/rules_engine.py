import pandas as pd


SUSPICIOUS_COUNTRIES = {"NG", "RU", "PK"}
HIGH_RISK_MCC = {"gambling", "crypto"}


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rule_large_amount"] = (df["amount"] > 500).astype(int)
    df["rule_very_large_amount"] = (df["amount"] > 2000).astype(int)

    df["rule_suspicious_country"] = df["country"].isin(SUSPICIOUS_COUNTRIES).astype(int)
    df["rule_high_risk_merchant"] = df["merchant_cat"].isin(HIGH_RISK_MCC).astype(int)

    df["hour"] = df["txn_ts"].dt.hour
    df["rule_night_txn"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)

    df["rules_score"] = (
        1.0 * df["rule_large_amount"] +
        2.0 * df["rule_very_large_amount"] +
        1.5 * df["rule_suspicious_country"] +
        1.5 * df["rule_high_risk_merchant"] +
        1.0 * df["rule_night_txn"]
    )
    return df

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def simulate_customers(n_customers=1000, seed=42):
    np.random.seed(seed)
    customer_id = np.arange(1, n_customers + 1)
    created_at = [datetime(2023, 1, 1) + timedelta(days=int(x)) 
                  for x in np.random.randint(0, 365, size=n_customers)]
    countries = np.random.choice(["US", "UK", "IN", "DE", "SG"], size=n_customers)
    risk_segment = np.random.choice(["low", "medium", "high"], p=[0.6, 0.3, 0.1], size=n_customers)

    return pd.DataFrame({
        "customer_id": customer_id,
        "created_at": created_at,
        "country": countries,
        "risk_segment": risk_segment
    })


def simulate_transactions(customers: pd.DataFrame, avg_txn_per_cust=80, seed=42):
    np.random.seed(seed)

    rows = []
    merchant_cats = ["grocery", "electronics", "entertainment", "gambling", "crypto", "travel"]
    channels = ["online", "pos", "atm"]

    for _, row in customers.iterrows():
        n_txn = np.random.poisson(lam=avg_txn_per_cust)
        base_time = datetime(2024, 1, 1)
        for _ in range(n_txn):
            dt = base_time + timedelta(days=int(np.random.exponential(scale=5)))
            amount = np.random.gamma(shape=2, scale=50)

            # inject some suspicious patterns for high-risk segment
            merchant = np.random.choice(merchant_cats)
            channel = np.random.choice(channels)
            country = np.random.choice(["US", "UK", "IN", "DE", "SG", "NG", "RU"])

            if row["risk_segment"] == "high":
                if np.random.rand() < 0.2:
                    amount *= np.random.uniform(3, 10)
                if np.random.rand() < 0.2:
                    merchant = "crypto"
                if np.random.rand() < 0.2:
                    country = "NG"

            is_chargeback = bool(np.random.rand() < 0.02)

            rows.append({
                "customer_id": row["customer_id"],
                "txn_ts": dt,
                "amount": round(amount, 2),
                "currency": "USD",
                "merchant_cat": merchant,
                "channel": channel,
                "country": country,
                "is_chargeback": is_chargeback
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    customers = simulate_customers()
    txns = simulate_transactions(customers)

    customers.to_csv(r"D:\Projects\Finance\Financial_Transaction_Risk_Analyzer\data\customers.csv", index=False)
    txns.to_csv(r"D:\Projects\Finance\Financial_Transaction_Risk_Analyzer\data\transactions.csv", index=False)
    print(customers.head())
    print(txns.head())

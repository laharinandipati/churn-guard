"""Generate a realistic synthetic churn dataset and save CSVs."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# contract + payment categories
CONTRACTS = ["month_to_month", "one_year", "two_year"]
PAYMENTS = ["credit_card", "debit_card", "bank_transfer", "paypal"]

def synth_dataset(n: int, pos_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, size=n)
    monthly = rng.normal(70, 20, size=n).clip(10, 200)
    tickets = rng.poisson(lam=2, size=n).clip(0, 15)
    contract = rng.choice(CONTRACTS, size=n, p=[0.55, 0.3, 0.15])
    payment = rng.choice(PAYMENTS, size=n)
    streaming = rng.choice([0, 1], size=n, p=[0.6, 0.4])

    total = monthly * tenure * (rng.normal(1.0, 0.05, size=n))

    # churn probability logit
    logit = (
            -0.04 * tenure
            + 0.015 * monthly
            + 0.25 * tickets
            - 0.5 * (contract == "two_year").astype(float)
            - 0.25 * (contract == "one_year").astype(float)
            + 0.1 * (payment == "paypal").astype(float)
            + 0.2 * (streaming == 0)
    )
    bias = np.percentile(logit, 100 * (1 - pos_rate))
    prob = 1 / (1 + np.exp(-(logit - bias)))
    churn = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "num_support_tickets": tickets,
            "contract_type": contract,
            "payment_method": payment,
            "has_addon_streaming": streaming.astype(bool),
            "churned": churn,
        }
    )
    return df

def main():
    n = 5000
    pos = 0.28
    seed = 42
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    df = synth_dataset(n, pos, seed)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["churned"]
    )
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)
    print(f"Wrote {len(train_df)} train rows and {len(test_df)} test rows to {data_dir}")

if __name__ == "__main__":
    main()

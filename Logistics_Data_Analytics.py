import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def process_week2_data():

    print("=== WEEK 2: DATA CLEANING & PREPROCESSING ===")

    # 1. Load Dataset
    df = pd.read_csv("logistics_shipments_demo.csv")

    print("\n1. Initial Dataset Preview (First 5 Rows):")
    print(df.head())

    # 2. Check Missing Values
    print("\n2. Missing Values Count per Column:")
    print(df.isnull().sum())

    # Remove rows with missing values
    df = df.dropna()

    # 3. IQR Outlier Treatment on Transport Cost

    Q1 = df["transport_cost_inr"].quantile(0.25)
    Q3 = df["transport_cost_inr"].quantile(0.75)

    IQR = Q3 - Q1

    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR

    df["Cost_Capped"] = np.where(
        df["transport_cost_inr"] > upper_bound,
        upper_bound,
        np.where(
            df["transport_cost_inr"] < lower_bound,
            lower_bound,
            df["transport_cost_inr"]
        )
    )

    print(
        f"\n3. Outlier Treatment Done: "
        f"Lower Bound = {lower_bound:.2f}, "
        f"Upper Bound = {upper_bound:.2f}"
    )

    # 4. Feature Scaling

    scaler = MinMaxScaler()

    df[[
        "Weight_Scaled",
        "Cost_Scaled",
        "Delivery_Time_Scaled"
    ]] = scaler.fit_transform(
        df[[
            "shipment_weight_kg",
            "Cost_Capped",
            "delivery_time_days"
        ]]
    )

    print("\n4. Feature Scaling (Min-Max Normalization) Completed.")

    # 5. Save Cleaned Dataset

    output_filename = "logistics_cleaned_dataset.csv"

    df.to_csv(output_filename, index=False)

    print(
        f"\nCleaned dataset saved successfully as "
        f"'{output_filename}'."
    )


if __name__ == "__main__":
    process_week2_data()
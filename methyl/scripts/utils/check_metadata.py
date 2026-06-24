import pandas as pd

meta = pd.read_csv("data/pretrain_metadata.csv.gz")
print("Columns:", list(meta.columns))
print("Shape:", meta.shape)
print(meta[["GSM_ID", "PATIENT_ID", "tissue", "age"]].head(10).to_string())
print("\nPATIENT_ID sample values:", meta["PATIENT_ID"].dropna().head(5).tolist())
print("GSM_ID sample values:", meta["GSM_ID"].dropna().head(5).tolist())

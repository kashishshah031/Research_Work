import pandas as pd

# paths
non_qac_path = "updated_dataset/non_QAC.csv"
qac_path = "updated_dataset/QAC.xlsx"
out_path = "updated_dataset/qac_data.csv"

# load files
df_non = pd.read_csv(non_qac_path)[["QAC_label","smiles",]]
df_qac = pd.read_excel(qac_path, engine="openpyxl")[["QAC_label","smiles"]]

# rename label column consistently
df_non.rename(columns={"QAC_labels": "label"}, inplace=True)
df_qac.rename(columns={"QAC_labels": "label"}, inplace=True)

# combine
df = pd.concat([df_non, df_qac], ignore_index=True)

# shuffle randomly
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# save
df.to_csv(out_path, index=False)
print(f"Saved combined dataset: {out_path}, total rows = {len(df)}")
print(df.head())

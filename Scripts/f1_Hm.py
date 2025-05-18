import pandas as pd
from sklearn.metrics import f1_score

# Load your CSV file
df = pd.read_csv("LORS_PRED_8.csv")  # Replace with your CSV path
s_df = pd.read_csv("LORS_PRED_16.csv")
df1 = pd.read_csv('vqa_baseline_results_normalized.csv')

# Compute F1 score (macro-averaged for multi-class)
f1 = f1_score(df["answer"], df["predicted_answer"], average="macro")
f2 = f1_score(s_df["answer"], s_df["predicted_answer"], average="macro")
#f3 = f1_score(df1["gt_answer"], df1["pred_answer"], average="macro")
print(f"F1 Score o r=8 (Macro): {f1:.4f}")
print(f"F1 Score of r=16 (Macro): {f2:.4f}")
#print(f"F1 Score of r=16 (Macro): {f3:.4f}")

# import pandas as pd
# from sklearn.metrics import f1_score

# # Load your CSV
# df1 = pd.read_csv('vqa_baseline_results_normalized.csv')

# # Check actual column names (debug print)
# print("Columns:", df1.columns)

# # Drop missing values and ensure same types
# df1 = df1.dropna(subset=["gt_answer", "pred_answer"])
# df1["gt_answer"] = df1["gt_answer"].astype(str)
# df1["pred_answer"] = df1["pred_answer"].astype(str)

# # Compute F1
# f3 = f1_score(df1["gt_answer"], df1["pred_answer"], average="macro")
# print(f"F1 Score (Macro): {f3:.4f}")

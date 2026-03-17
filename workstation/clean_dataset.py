import pandas as pd
import os
import shutil

print("=== Dataset Cleaner ===\n")

# Load data
df = pd.read_csv("dataset/labels.csv")
print(f"Original dataset: {len(df)} samples")
print(df["throttle"].describe())

# Filter: Remove frames with throttle < 0.2
df_clean = df[df["throttle"] >= 0.2].copy()

# Also remove extreme steering with low throttle (probably errors)
df_clean = df_clean[~((abs(df_clean["steering"]) > 0.8) & (df_clean["throttle"] < 0.3))]

print(f"\nCleaned dataset: {len(df_clean)} samples ({len(df) - len(df_clean)} removed)")
print(df_clean["throttle"].describe())

# Backup original
if not os.path.exists("dataset/labels_original.csv"):
    shutil.copy("dataset/labels.csv", "dataset/labels_original.csv")
    print("\n✓ Backed up original to labels_original.csv")

# Save cleaned version
df_clean.to_csv("dataset/labels.csv", index=False)
print("✓ Saved cleaned dataset\n")
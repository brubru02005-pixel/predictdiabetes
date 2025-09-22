import pandas as pd

# Load the combined dataset
input_path = 'data/combined_diabetes.csv'
output_path = 'data/combined_diabetes_cleaned.csv'

df = pd.read_csv(input_path)

# Unify target column: prefer 'Outcome', else use 'CLASS'
if 'Outcome' in df.columns and 'CLASS' in df.columns:
    df['Outcome'] = df['Outcome'].combine_first(df['CLASS'])
    df = df.drop(columns=['CLASS'])
elif 'CLASS' in df.columns:
    df = df.rename(columns={'CLASS': 'Outcome'})

# Drop columns that are completely empty or have only one unique value (except Outcome)
feature_cols = [col for col in df.columns if col != 'Outcome']
for col in feature_cols:
    if df[col].isnull().all() or (df[col].nunique(dropna=True) <= 1):
        df = df.drop(columns=[col])

# Drop rows with missing or empty Outcome
df = df[df['Outcome'].notnull() & (df['Outcome'] != '')]

# Move Outcome to last column
cols = [c for c in df.columns if c != 'Outcome'] + ['Outcome']
df = df[cols]

# Save cleaned dataset
df.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to {output_path}. Shape: {df.shape}")

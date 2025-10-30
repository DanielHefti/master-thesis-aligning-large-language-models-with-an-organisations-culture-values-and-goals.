import pandas as pd
import random

# Load the Excel file
file_path = "selected_models.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")

# Drop rows with missing strata
df = df.dropna(subset=['Strata'])

# Set the random seed for reproducibility
random.seed("Assign Large Language Models to Treatment and Control Groups - Public Dataset")

# Number of models to select per stratum
n_per_stratum = 1  # Change as needed

# Prepare a list to collect sampled models
selected_models = []

# Group by 'Strata' and sample
for stratum, group in df.groupby('Strata'):
    # If fewer models than needed, take all
    if len(group) <= n_per_stratum:
        sampled = group
    else:
        sampled = group.sample(n=n_per_stratum, random_state=42)
    selected_models.append(sampled)

# Concatenate all sampled models
selected_df = pd.concat(selected_models)

# Reset index and show result
selected_df = selected_df.reset_index(drop=True)
print(selected_df[['Model Family', 'Size in B', 'Context Length', 'Modality', 'Strata']])

# Write the selected models to a new Excel file
selected_df.to_excel("selected_models_treatment.xlsx", index=False)
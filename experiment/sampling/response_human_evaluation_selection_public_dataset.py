import os
import pandas as pd
import random

# Load the response and grading files
response_file = os.path.join(os.getcwd(), "experiment", "responses", "all_responses_joined.csv")
grading_file = os.path.join(os.getcwd(), "experiment", "analysis", "without_reference_answer", "all_responses_grading_separated_cleaned.csv")

df = pd.read_csv(response_file, header=0)
grading_df = pd.read_csv(grading_file, header=0)

# Remove all rows that contain the value 'Error' in any column
df = df[~df.apply(lambda row: row.astype(str).str.contains('Error').any(), axis=1)]

# Ensure 'phase' is of integer type, skip headers
df['phase'] = df['phase'].astype(int)



# Filter df to only rows present in grading_df (by model-name, phase, attempt)
merge_cols = ['model-name', 'phase', 'attempt']
filtered_df = df.merge(grading_df[merge_cols], on=merge_cols, how='inner')

# Set the random seed for reproducibility
random.seed("Aligning Large Language Models with an Organisation's Culture, Values and Goals")

# Number of responses to select per phase
n_per_phase = 5  # Change as needed

# Prepare a list to collect sampled models
selected_responses = []

# Group by phase
for stratum, group in filtered_df.groupby('phase'):
    # If fewer models than needed, take all
    if len(group) <= n_per_phase:
        sampled = group
    else:
        sampled = group.sample(n=n_per_phase, random_state=42)
    selected_responses.append(sampled)

# Concatenate all sampled responses
selected_df = pd.concat(selected_responses)

# Reset index and show result
selected_df = selected_df.reset_index(drop=True)

# Write the selected models to a new Excel file
selected_df.to_excel("selected_responses_public_vllm.xlsx", index=False)
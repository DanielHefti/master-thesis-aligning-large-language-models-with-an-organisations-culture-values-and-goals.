import os
import glob
import pandas as pd

# Path to the responses directory
base_dir = os.path.join(os.getcwd(), "responses")

# Find all CSV files in subfolders, EXCLUDING the output file
csv_files = [
    f for f in glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
    if not f.endswith("all_responses_joined.csv")
]

# Read and concatenate all CSV files
dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e:
        print(f"Could not read {file}: {e}")

if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(os.path.join(base_dir, "all_responses_joined.csv"), index=False)
    print(f"Combined {len(csv_files)} files into all_responses_joined.csv")
else:
    print("No CSV files found or all files failed to read.")
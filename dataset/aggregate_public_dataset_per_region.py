import pandas as pd

# Input and output file paths
input_file = "online_retail_II.xlsx"
output_file = "aggregated_by_country.csv"

# Sheets to process
sheets = ["Year 2009-2010", "Year 2010-2011"]


# Read sheets and add Fiscal Year column
df_list = []
for sheet in sheets:
	temp_df = pd.read_excel(input_file, sheet_name=sheet)
	# Assign fiscal year based on sheet name
	if sheet == "Year 2009-2010":
		temp_df["Fiscal Year"] = "1"
	elif sheet == "Year 2010-2011":
		temp_df["Fiscal Year"] = "2"
	else:
		temp_df["Fiscal Year"] = "unknown"
	df_list.append(temp_df)
df = pd.concat(df_list, ignore_index=True)

# Replace missing Country with 'unknown' and ensure string type
df["Country"] = df["Country"].fillna("unknown").astype(str)

# Create salesamount column as Quantity * Price
if 'Quantity' in df.columns and 'Price' in df.columns:
	df['salesamount'] = df['Quantity'] * df['Price']

# Replace missing Customer ID with -1, then convert to integer
df["Customer ID"] = df["Customer ID"].fillna(-1)
try:
	df["Customer ID"] = df["Customer ID"].astype(int)
except Exception:
	# If conversion fails, coerce errors to -1
	df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce").fillna(-1).astype(int)

# Aggregate by Country, Customer ID, and Fiscal Year (sum all numeric columns)
agg_df = df.groupby(["Country", "Fiscal Year"], as_index=False).sum(numeric_only=True)

# Remove Customer ID column if present
if 'Customer ID' in agg_df.columns:
	agg_df = agg_df.drop(columns=['Customer ID'])

# Round salesamount to integer if present
if 'salesamount' in agg_df.columns:
	agg_df['salesamount'] = agg_df['salesamount'].round(0).astype(int)

# Remove Quantity and Price columns if present
for col in ['Quantity', 'Price']:
	if col in agg_df.columns:
		agg_df = agg_df.drop(columns=[col])

# Output to text file in format: Country:FiscalYear1=Sales1,FiscalYear2=Sales2,...
lines = []
for country, group in agg_df.groupby("Country"):
	fy_sales = [f"{row['Fiscal Year']}={int(row['salesamount'])}" for _, row in group.iterrows()]
	line = f"{country}:" + ",".join(fy_sales)
	lines.append(line)

with open("aggregated_by_country.txt", "w", encoding="utf-8") as f:
	for line in lines:
		f.write(line + "\n")
#           Cognate Clustering          #



from pathlib import Path
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm # progress bar
import numpy as np # numerical operations

cognet_filelist = ['vashantor_cognets_barishal_ipa_with_bangla_ipa.csv', 'vashantor_cognets_sylhet_ipa_with_bangla_ipa.csv', 'vashantor_cognets_mymensingh_ipa_with_bangla_ipa.csv', 'vashantor_cognets_noakhali_ipa_with_bangla_ipa.csv', 'vashantor_cognets_chittagong_ipa_with_bangla_ipa.csv']

region_names = ['barishal', 'sylhet', 'mymensingh', 'noakhali', 'chittagong']

# Load and clean data
dfs = []
for file, region in tqdm(zip(cognet_filelist, region_names), total=len(region_names), desc="Reading"):
	df = pd.read_csv("data/" + file)
	df.replace('', np.nan, inplace=True)
	df.dropna(inplace=True, how='any')
     
	# Remove rows with missing data
	df = df.dropna(subset=['bangla_speech_word', 'bangla_speech_ipa', 'regional_word', 'regional_ipa'])
	df = df[~df.astype(str).apply(lambda row: row.str.contains('"')).any(axis=1)]
	# 🔄 Drop duplicates to avoid merge issues
	df = df.drop_duplicates(subset=['bangla_speech_word', 'bangla_speech_ipa'])

	# Rename columns
	df = df.rename(columns={
		'regional_word': f'{region}_word',
		'regional_ipa': f'{region}_ipa'
	})

	dfs.append(df)


# Start with the first dataframe
merged_df = dfs[0]

# Incrementally merge and print shape at each step
for df, region in tqdm(zip(dfs[1:], region_names[1:]), total=len(region_names) - 1, desc="Merging"):
    print(f"Merging with {region}... (before: {merged_df.shape})")
    merged_df = pd.merge(
        merged_df, df,
        on=['bangla_speech_word', 'bangla_speech_ipa'],
        how='outer'  # You can try 'inner' for faster test
    )
    print(f"After merging {region}: {merged_df.shape}")

# Drop incomplete rows (only keep rows with all dialects)
required_columns = [f'{r}_word' for r in region_names] + [f'{r}_ipa' for r in region_names]
filtered_df = merged_df.dropna(subset=required_columns)

filtered_df.replace('', np.nan, inplace=True)
filtered_df.dropna(inplace=True, how='any')

# Sort and save
filtered_df = filtered_df.sort_values(by='bangla_speech_word')
filtered_df.to_csv('data/merged_regional_alignment.csv', index=False)

print("Merged file saved as 'merged_regional_cognates.csv'")


input_file = Path.cwd() / "data/merged_regional_alignment.csv"
output_file = "data/clean_merged_regional_cognates.csv"
target_char = '"'  # Character to search for

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Keep only lines that DO NOT contain the target character
cleaned_lines = [line for line in lines if target_char not in line]

# Save to a new file (or overwrite the same file)
with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(cleaned_lines)

print(f"Removed lines containing '{target_char}' and saved to '{output_file}'")
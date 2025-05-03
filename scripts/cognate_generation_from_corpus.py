#           Cognate Generation From Corpus          #


from pathlib import Path
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm # progress bar



filelist = ['vashantor_barishal_ipa_with_bangla_ipa.csv', 'vashantor_sylhet_ipa_with_bangla_ipa.csv', 'vashantor_mymensingh_ipa_with_bangla_ipa.csv', 'vashantor_noakhali_ipa_with_bangla_ipa.csv', 'vashantor_chittagong_ipa_with_bangla_ipa.csv']

for file in filelist:
	filename = file
	vashantor_input_file = Path.cwd() / "data/" + filename

	# # Read the CSV file
	df = pd.read_csv(vashantor_input_file, encoding='utf-8')

	vashantor_output_file = filename.replace('vashantor_', 'vashantor_cognets_')

	# Prepare output storage
	output_rows = []

	# Setup tqdm for iteration
	for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
		# Extract and clean sentences
		bangla_words = str(row['bangla_speech']).strip().split()
		bangla_ipa = str(row['bangla_speech_ipa']).strip().split()
		
		regional_words = str(row['regional']).strip().split()
		regional_ipa = str(row['regional_ipa']).strip().split()
		
		# Remove the first word from regional and regional_ipa (e.g., "বরিশাল" and "bɐɪʃɐl")
		if len(regional_words) > 0 and len(regional_ipa) > 0:
			regional_words = regional_words[1:]
			regional_ipa = regional_ipa[1:]

		# Find minimum length to avoid mismatches
		min_len = min(len(bangla_words), len(bangla_ipa), len(regional_words), len(regional_ipa))
		
		# Track seen IPA pairs to remove duplicates
		seen_pairs = set()

		for i in range(min_len):
			b_word = bangla_words[i]
			b_ipa = bangla_ipa[i]
			r_word = regional_words[i]
			r_ipa = regional_ipa[i]

			# Skip if IPA already seen
			if (b_ipa, r_ipa) in seen_pairs:
				continue

			seen_pairs.add((b_ipa, r_ipa))

			output_rows.append({
				'bangla_speech_word': b_word,
				'bangla_speech_ipa': b_ipa,
				'regional_word': r_word,
				'regional_ipa': r_ipa
			})

	# Convert to DataFrame and export
	out_df = pd.DataFrame(output_rows)
	out_df.to_csv("data/" + vashantor_output_file, index=False)

	print(f"Done. Output saved to: {"data/" + vashantor_output_file}")
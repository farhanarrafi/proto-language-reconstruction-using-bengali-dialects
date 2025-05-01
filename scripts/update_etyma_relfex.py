from pathlib import Path
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from rapidfuzz import process, fuzz


# Load your Bengali dataset
bangla_df = pd.read_csv(Path.cwd() /'data/final_translated_cognates.csv', encoding='utf-8')

# Load Pokorny's PIE dictionary dataset Reflex,Etyma
# Ensure you have downloaded and placed the dataset in the same directory
english_reflex_index = pd.read_csv('PIE_English_Reflex_Index.csv',encoding='utf-8')

# Preprocess Pokorny dataset: lowercasing glosses for better matching
#pokorny_df['English_gloss'] = pokorny_df['English_gloss'].str.lower()

# Function to find the best matching PIE root for a given English translation
def find_pie_root(english_word, pokorny_glosses, threshold=80):
    if pd.isnull(english_word):
        return None, None
    english_word = english_word.lower()
    match = process.extractOne(english_word, pokorny_glosses, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= threshold:
        matched_gloss = match[0]
        pie_entry = english_reflex_index[english_reflex_index['Reflex'] == matched_gloss].iloc[0]
        return pie_entry['Etyma'], pie_entry['Reflex']
    else:
        return None, None

# Apply the matching function to each row
pokorny_glosses = english_reflex_index['Reflex'].dropna().unique()
bangla_df[['PIE_root', 'PIE_gloss']] = bangla_df['english_translation'].apply(
    lambda x: pd.Series(find_pie_root(x, pokorny_glosses))
)

dropped = bangla_df[bangla_df['PIE_root'].isnull()]
dropped.to_csv("data/bangla_with_no_pie_root_matches_reflex_index.csv", index=False)

# Drop rows where PIE root could not be found
bangla_df = bangla_df.dropna(subset=['PIE_root'])

# Save the enriched dataset
bangla_df.to_csv('data/bangla_with_pie_roots_reflex_index.csv', index=False, encoding='utf-8')
print("Enriched dataset saved as 'bangla_with_pie_roots_reflex_index.csv'")


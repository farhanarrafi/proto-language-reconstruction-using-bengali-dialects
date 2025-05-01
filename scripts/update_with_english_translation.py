from pathlib import Path
import epitran
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm # progress bar
from functools import reduce
import numpy as np # numerical operations
import asyncio
from deep_translator import GoogleTranslator



filename = "data/clean_merged_regional_cognates.csv"
file_path = Path.cwd() / filename

# Read the CSV file
df = pd.read_csv(file_path, encoding='utf-8')

# Optional: remove duplicates before translation
# df = df.drop_duplicates(subset=['bangla_speech_word'])

output_folder = Path.cwd() / "data/translated_batches"
# Create output folder if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

batch_size = 100
num_batches = (len(df) + batch_size - 1) // batch_size  # ceiling division

async def translate_batch(words, batch_num):
    translations = []
    for word in words:
        try:
            # Using GoogleTranslator instead of async Translator
            result = GoogleTranslator(source='bn', target='en').translate(word)
            translations.append(result)
        except Exception as e:
            print(f"[Batch {batch_num}] Translation failed for '{word}': {e}")
            translations.append("")
        # Add a small delay to avoid hitting rate limits
        await asyncio.sleep(2)
    return translations

async def process_batches():
    for i in range(num_batches):
        if i < 26:
            continue
        start = i * batch_size
        end = min((i + 1) * batch_size, len(df))
        batch_df = df.iloc[start:end].copy()
        words = batch_df['bangla_speech_word'].tolist()

        print(f"Translating batch {i + 1}/{num_batches} ({start}–{end})...")
        batch_df['english_translation'] = await translate_batch(words, i + 1)

        # Save each batch
        batch_file = output_folder / f"translated_batch_{i + 1}.csv"
        batch_df.to_csv(batch_file, index=False)
        print(f"Saved batch {i + 1} to {batch_file.name}")

    # Merge all saved batches into one file
    print("Merging all translated batches...")
    all_batches = [pd.read_csv(f) for f in sorted(output_folder.glob("translated_batch_*.csv"))]
    merged_df = pd.concat(all_batches).reset_index(drop=True)
    merged_df.to_csv("data/final_translated_cognates.csv", index=False)
    print("All done! Final file saved as 'final_translated_cognates.csv'")

# Run
if __name__ == "__main__":
    asyncio.run(process_batches())

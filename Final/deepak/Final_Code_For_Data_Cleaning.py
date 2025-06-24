import pandas as pd
import re

# ===== STEP 1: Load input file =====
# just update the file name here before running
input_file = "bullionvault_articles.csv"
output_file = "tokenizer_ready_output.csv"

# read the CSV file using pandas
df = pd.read_csv(input_file)

# fix column names in case they have extra spaces
df.columns = [c.strip() for c in df.columns]

# ===== STEP 2: Detect and prepare content column =====
# some files use 'News', some use 'Content'
if 'News' in df.columns:
    df['Text'] = df['News']
elif 'Content' in df.columns:
    df['Text'] = df['Content']
else:
    raise Exception("no valid text column found (check if column is named News or Content)")

# fix date column also
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
elif 'Dates' in df.columns:
    df['Date'] = pd.to_datetime(df['Dates'], errors='coerce')
else:
    df['Date'] = pd.NaT  # no date available, it's fine

# ===== STEP 3: Basic cleaning =====
# remove Title: from start and clean up spaces
def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = t.replace("Title:", "")
    t = t.strip()
    return ' '.join(t.split())  # normalize spaces

df['Text'] = df['Text'].apply(clean_text)
df = df[df['Text'] != ""]  # drop empty rows

# ===== STEP 4: Break long content into chunks (max ~512 tokens) =====
def split_into_sentences(txt):
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', txt)

def chunk_text(text, max_len=1500):  # roughly 512 tokens
    sents = split_into_sentences(text)
    result = []
    chunk = ""
    for s in sents:
        if len(chunk) + len(s) + 1 <= max_len:
            chunk += " " + s if chunk else s
        else:
            result.append(chunk)
            chunk = s
    if chunk:
        result.append(chunk)
    return result

# ===== STEP 5: Sentiment labeling (heuristic based) =====
# 0 = positive, 1 = neutral, 2 = negative
def get_label(text):
    text = text.lower()
    if any(word in text for word in ['gold prices rise', 'surge', 'bullish', 'rate cut', 'safe-haven', 'investing trends higher']):
        return 0
    elif any(word in text for word in ['fall', 'drop', 'bearish', 'loss', 'decline', 'sell-off']):
        return 2
    else:
        return 1

# if label already exists (like Price Sentiment), use it
if 'Price Sentiment' in df.columns:
    convert = {"positive": 0, "neutral": 1, "none": 1, "negative": 2}
    df['Label'] = df['Price Sentiment'].str.lower().map(convert)
else:
    df['Label'] = df['Text'].apply(get_label)

# drop rows where label is still missing
df = df.dropna(subset=['Label'])

# ===== STEP 6: Build final rows =====
final_rows = []

# loop through each article and break into chunks
for idx, row in df.iterrows():
    text = row['Text']
    date = row['Date']
    label = int(row['Label'])
    parts = chunk_text(text)
    for p in parts:
        line = clean_text(p.lower())  # lowercase also
        if line:
            final_rows.append({
                'Date': date,
                'Content': line,
                'Label': label
            })

# ===== STEP 7: Save the final cleaned and labeled file =====
final_df = pd.DataFrame(final_rows)
final_df.to_csv(output_file, index=False)
print("Done! Saved to:", output_file)

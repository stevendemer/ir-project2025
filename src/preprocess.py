import pandas as pd
import spacy 
import time
import os
import pyarrow as pa
import pyarrow.parquet as pq


BATCH_SIZE = 3000
INPUT_FILENAME = "full_speech.csv"
OUTPUT_FILENAME = "cleaned_data.parquet"


# Limit the number of rows to process (for testing). Set to None to process all.
LIMIT = 50_000

print("Loading spacy model ...")

try:
    nlp = spacy.load("el_core_news_md")
except OSError:
    try:
        nlp = spacy.load("el_core_news_sm", disable=['ner', 'parser'])
    except OSError:
        print("Error: Model el_core_news_md not found. Run python -m spacy download el_core_news_md to install it.")
        exit()


def clean_text_batch(texts):
    """Clean a batch of texts using Spacy pipes."""
    cleaned_texts = []
    
    docs = nlp.pipe(texts, batch_size=200, n_process=1)

    for doc in docs:
        tokens = [
            token.lemma_.lower() for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.like_num
            and len(token) > 2
        ]
        cleaned_texts.append(' '.join(tokens))
    
    return cleaned_texts




def load_data(filepath):
    """Load data from a CSV file."""

    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
    except FileNotFoundError:
        print('File not found')
        exit()

    df['speech'] = df['speech'].fillna('')
    df['member_name'] = df['member_name'].fillna('Unknown')
    df['political_party'] = df['political_party'].fillna('Unknown')
        
    if 'sitting_date' in df.columns:
        df['sitting_date'] = pd.to_datetime(df['sitting_date'], format='%d/%m/%Y', errors='coerce')
    
    print(f'Loaded {len(df)} rows from {filepath}')
    return df

def clean_text_batch(texts):
    """Clean a batch of texts using spaCy."""
    cleaned_texts = []

    # disable heavy pipeline components for speed
    docs = nlp.pipe(texts, batch_size=100, disable=["ner", "parser"])

    for doc in docs:
        tokens = [
            token.lemma_.lower() for token in doc 
            if not token.is_stop 
            and not token.is_punct 
            and not token.like_num 
            and len(token) > 2
        ]
        cleaned_texts.append(' '.join(tokens))
    
    return cleaned_texts


if __name__ == "__main__":
    start_time = time.time()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_path = os.path.join(project_root, "data", INPUT_FILENAME)
    output_path = os.path.join(project_root, "data", OUTPUT_FILENAME)

    
    if not os.path.exists(input_path):
        print(f'Error: {input_path} not found.')
        exit()
    
    print(f'Processing {input_path} in chunks of {BATCH_SIZE}')
    
    if os.path.exists(output_path):
        os.remove(output_path)
    
    
    writer = None
    rows_processed = 0

    try:
        with pd.read_csv(input_path, chunksize=BATCH_SIZE, on_bad_lines='skip', encoding='utf-8') as reader:
            for chunk in reader:
                if LIMIT and rows_processed >= LIMIT:
                    break
                
                chunk['speech'] = chunk['speech'].fillna('')
                chunk['member_name'] = chunk['member_name'].fillna("Unknown")
                chunk['political_party'] = chunk['political_party'].fillna('Unknown')
                
                if 'sitting_date' in chunk.columns:
                    chunk['sitting_date'] = pd.to_datetime(chunk['sitting_date'], dayfirst=True, errors='coerce')
                
                # clean text
                print(f'Processing rows: {rows_processed} to {rows_processed + len(chunk)} ...')
                chunk['processed_speech'] = clean_text_batch(chunk['speech'].tolist())

                # filter empty results
                chunk = chunk[chunk['processed_speech'].str.len() > 2]
                
                # write to parquet incrementally
                table = pa.Table.from_pandas(chunk)
                
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                
                writer.write_table(table)
                rows_processed += len(chunk)
    except Exception as e:
        print(f'Error occurred: {e}')
    finally:
        if writer:
            writer.close()
    
    end_time = time.time()

    print(f'Completed ! Processed {rows_processed} speeches in {(end_time - start_time) / 60:.2f} minutes')

    print(f'Saved to {output_path}')
                
                



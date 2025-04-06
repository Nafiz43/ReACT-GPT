import os
import pymupdf4llm
import pandas as pd
import multiprocessing as mp
from _constant_func import *
import signal
import csv

def timeout_handler(signum, frame):
    print("Warning: The function took too long but will continue.")
    # raise TimeoutError("Timeout: The function took too long.")


signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(20)  

local_directory = "/mnt/data/nafiz43/projects/ReACT-GPT/data/Paper-Set/"
pdf_files = get_pdfs(local_directory)

# Use a shared counter
counter = mp.Value('i', 0)

# Leave at least 2 cores free
num_workers = max(mp.cpu_count() - 10, 1)  
print("Number of workers:", num_workers)

output_csv = 'data/article_data.csv'
os.makedirs('data/', exist_ok=True)

def parse_article(pdf_path):
    """Parse a single PDF file and return a DataFrame."""
    article_path = pdf_path.replace("/mnt/data/nafiz43/projects/ReACT-GPT/data/Paper-Set/", "")

    try:
        # markdown_text = pymupdf4llm.to_markdown(pdf_path)
        try:
            print("starts processing:", article_path)
            markdown_text = pymupdf4llm.to_markdown(pdf_path)
        except:
            signal.alarm(0)  # Always cancel the alarm
            print("Failed for :", article_path)

        cleaned_article_text = clean_article_text(markdown_text)
        article_title = extract_article_title(cleaned_article_text)
        article_link = extract_article_link(cleaned_article_text)
        print("ends processing:", article_path)

        return pd.DataFrame({
            "Title": [article_title],
            "Link": [article_link],
            "Text": [cleaned_article_text],
            "File-path": [article_path]
        })
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return None

def process_and_save(pdf_path):
    """Process a single PDF and return the result."""
    df = parse_article(pdf_path)
    if df is not None:
        with counter.get_lock():
            counter.value += 1
            print(counter.value, "out of:", len(pdf_files), end="\r")
        return df
    return None

if __name__ == "__main__":
    # Create CSV with headers
    pd.DataFrame(columns=["Title", "Link", "Text", "File-path"]).to_csv(output_csv, index=False)

    # Use multiprocessing pool
    with mp.Pool(processes=num_workers) as pool:
        results = []
        # Process files with error handling
        cnt=1
        for result in pool.imap_unordered(process_and_save, pdf_files):
            if result is not None:
                results.append(result)

                # with open(output_csv, mode="a", newline="", encoding="utf-8") as file:
                #     writer = csv.writer(file)
                #     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                #     writer.writerow([result['Title'],result['Link'], result['Text'], result['File-path']])

            if(cnt>=800):
                final_df = pd.concat(results, ignore_index=True)
                final_df.to_csv(output_csv, index=False)

            cnt = cnt+1

    # Combine all results and write once
    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(output_csv, index=False)

    print(f"Total Number of Articles Processed: {len(final_df)}")
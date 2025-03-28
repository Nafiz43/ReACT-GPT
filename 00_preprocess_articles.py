import os
import pymupdf4llm
import pandas as pd
import re
from _constant_func import *

local_directory = "/mnt/data/nafiz43/projects/ReACT-GPT/data/Paper-Set/"
pdf_files = get_pdfs(local_directory)

# print(pdf_files)

master_df = pd.DataFrame()

def parse_article(pdf_path):
    # Extract the entire document as a single Markdown string
    markdown_text = pymupdf4llm.to_markdown(pdf_path)
    cleaned_article_text = clean_article_text(markdown_text)
    article_title = extract_article_title(cleaned_article_text)[0]
    article_link = extract_article_link(cleaned_article_text)

    df = pd.DataFrame({
        "Title": [article_title],
        "Link": [article_link],
        "Text": [cleaned_article_text]
    })

    return df

cnt = 1
for pdf in pdf_files:
    master_df = pd.concat([master_df, parse_article(pdf)], ignore_index=True)
    print("Processed", cnt, "out of ", len(pdf_files), "articles", end="\r")
    cnt = cnt+1

os.makedirs('data/', exist_ok=True)

master_df.to_csv('data/article_data.csv', index=False)

print(master_df.head())
print("Total Number of Articles Processed:", len(master_df))
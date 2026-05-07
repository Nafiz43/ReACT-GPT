"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
import os
import logging
import click
import csv
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from langchain_ollama import OllamaLLM as Ollama
from tqdm import tqdm
from _constant_func import *


MD_DIR = "/data/Deep_Angiography/ReACT-GPT/data/paper-set/processed-infer-set"

TIMEOUT_SECONDS = 1800  # 30 minutes per article

# --------------------------------------------------------------------------- #
#  Model-family → CSV filename mapping                                         #
#  Matching is done on the lowercase model name string.                        #
#  The order matters: first substring match wins.                              #
# --------------------------------------------------------------------------- #
FAMILY_CSV_MAP = [
    ("deepseek", "deepseek.csv"),
    ("llama",    "llama.csv"),
    ("gpt",      "gpt.csv"),
    ("qwen",     "qwen.csv"),
    ("gemma",    "gemma.csv"),
    ("mixtral",  "mixtral.csv"),
]


# --------------------------------------------------------------------------- #
#  Model-family helpers                                                        #
# --------------------------------------------------------------------------- #

def resolve_log_path(log_dir: str, model_name: str) -> str:
    """
    Return the CSV path for this model's family.
    Falls back to 'other.csv' if no family keyword matches.
    """
    model_lower = model_name.lower()
    for keyword, filename in FAMILY_CSV_MAP:
        if keyword in model_lower:
            return os.path.join(log_dir, filename)
    return os.path.join(log_dir, "other.csv")


def load_already_processed(log_path: str) -> set[str]:
    """
    Read the existing family CSV (if it exists) and return the set of
    article titles that have already been processed.
    Matching is done on the 'article_title' column.
    Returns an empty set if the file does not exist or cannot be read.
    """
    if not os.path.exists(log_path):
        return set()

    done = set()
    try:
        with open(log_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = row.get("article_title", "").strip()
                if title:
                    done.add(title)
    except Exception as e:
        print(f"[warn] Could not read existing log {log_path}: {e}")

    return done


def ensure_csv_header(log_path: str) -> None:
    """
    Create the CSV with a header row if it does not already exist.
    If it already exists, leave it completely untouched.
    """
    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "venue", "article_title",
                "article_link", "answer", "model_name",
            ])


# --------------------------------------------------------------------------- #
#  MD parsing helpers                                                          #
# --------------------------------------------------------------------------- #

def load_md_files(md_dir: str) -> list[dict]:
    """
    Read all .md files in md_dir and parse them into article dicts.
    Expected md structure written by 00_preprocess_articles.py:
        # <Title>

        **Source:** <link>
        **Original file:** <file_path>

        ---

        <body text>
    """
    articles = []
    md_paths = sorted([
        os.path.join(md_dir, f)
        for f in os.listdir(md_dir)
        if f.endswith(".md")
    ])

    for md_path in md_paths:
        with open(md_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # -- Title: first line starting with "# "
        title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else ""

        # -- Source link
        link_match = re.search(r"\*\*Source:\*\*\s*(.+)$", raw, re.MULTILINE)
        link = link_match.group(1).strip() if link_match else ""

        # -- Original file path
        file_match = re.search(r"\*\*Original file:\*\*\s*(.+)$", raw, re.MULTILINE)
        file_path = file_match.group(1).strip() if file_match else md_path

        # -- Body: everything after the "---" divider
        parts = raw.split("---\n", maxsplit=1)
        body = parts[1].strip() if len(parts) > 1 else raw.strip()

        articles.append({
            "Title":     title,
            "Link":      link,
            "File-path": file_path,
            "Text":      body,
            "md_path":   md_path,
        })

    return articles


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

@click.command()
@click.option(
    "--model_name",
    default="llama3.1:latest",
    type=str,
    help="Ollama model to use for inference.",
)
@click.option(
    "--temp",
    default=0,
    type=int,
    help="Randomness of the model (temperature).",
)
@click.option(
    "--prompting_method",
    default="IP",
    type=click.Choice(allowable_prompting_methods),
    help="Prompting strategy: IP (Instruction), CoT (Chain-of-Thought), RA (Retrieval-Augmented).",
)
@click.option(
    "--reports_to_process",
    default=-1,
    type=int,
    help="Number of articles to process (-1 = all, applied after skipping already-done ones).",
)
@click.option(
    "--md_dir",
    default=MD_DIR,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing preprocessed .md files.",
)
@click.option(
    "--keep_alive",
    default="120m",
    type=str,
    help=(
        "How long Ollama keeps the model loaded in VRAM between calls. "
        "Accepts Ollama duration strings: '5m', '1h', '-1' (forever). "
        "Default: '120m' (2 hours). Set to '0' to unload after each call."
    ),
)
@click.option(
    "--log_dir",
    default="local_history",
    type=str,
    help="Directory where family CSV logs are stored (default: local_history).",
)
def main(model_name, prompting_method, reports_to_process, temp,
         md_dir, keep_alive, log_dir):

    # -- Resolve the family CSV for this model
    os.makedirs(log_dir, exist_ok=True)
    log_path = resolve_log_path(log_dir, model_name)

    print(f"Model            : {model_name}")
    print(f"Temperature      : {temp}")
    print(f"Prompting method : {prompting_method}")
    print(f"MD directory     : {md_dir}")
    print(f"Keep-alive       : {keep_alive}")
    print(f"Log file         : {log_path}")
    print(f"Timeout          : {TIMEOUT_SECONDS}s ({TIMEOUT_SECONDS // 60} min) per article")

    # -- Select question template
    question = {
        "IP":  IP_template,
        "CoT": CoT_template,
        "RA":  RA_template,
    }[prompting_method]

    # -- Load all articles from .md files
    all_articles = load_md_files(md_dir)
    if not all_articles:
        print(f"No .md files found in {md_dir}. Exiting.")
        return

    # -- Load already-processed titles from the family CSV
    already_done = load_already_processed(log_path)
    print(f"\nTotal articles on disk     : {len(all_articles)}")
    print(f"Already processed (skipped): {len(already_done)}")

    # -- Filter to only fresh (not-yet-processed) articles
    fresh_articles = [
        a for a in all_articles
        if a["Title"].strip() not in already_done
    ]
    print(f"Fresh articles to process  : {len(fresh_articles)}")

    if not fresh_articles:
        print("\nAll articles already processed. Nothing to do. Exiting.")
        return

    # -- Apply --reports_to_process cap on the fresh set
    if reports_to_process > 0:
        fresh_articles = fresh_articles[:reports_to_process]
        print(f"Capped to                  : {len(fresh_articles)} (--reports_to_process)")

    print()

    # -- Ensure the CSV exists with a header (does NOT overwrite existing data)
    ensure_csv_header(log_path)

    # -- Build Ollama client
    # keep_alive  : model stays in VRAM between articles — no cold-start reload
    # num_ctx     : article bodies can be up to 24k tokens; 32768 is safe
    # num_predict : cap output; ReACT extraction lists don't need more than 2048
    # temperature : from CLI (default 0 → deterministic)
    # timeout     : HTTP-level client timeout
    ollama = Ollama(
        model=model_name,
        temperature=temp,
        keep_alive=keep_alive,
        num_ctx=32768,
        num_predict=2048,
        timeout=TIMEOUT_SECONDS,
    )
    logging.getLogger().setLevel(logging.ERROR)

    succeeded = 0
    failed    = 0
    timed_out = 0
    skipped   = len(already_done)   # for final summary

    with tqdm(fresh_articles, desc="Running inference",
              unit="article", dynamic_ncols=True) as pbar:
        for article in pbar:
            try:
                query = prompt_template + article["Text"] + question

                # ThreadPoolExecutor is a hard safety net in case the
                # HTTP-level timeout is swallowed by the client internally.
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(ollama.invoke, query)
                    try:
                        response = future.result(timeout=TIMEOUT_SECONDS)
                    except FuturesTimeout:
                        tqdm.write(
                            f"\nTimed out ({TIMEOUT_SECONDS // 60} min) on: "
                            f"{article.get('md_path', '?')}"
                        )
                        timed_out += 1
                        failed += 1
                        continue

                response = clean_response(response)

                # Append to the family CSV — never overwrite
                with open(log_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        article["File-path"],
                        article["Title"],
                        article["Link"],
                        response,
                        model_name,
                    ])

                succeeded += 1

            except Exception as e:
                tqdm.write(f"\nFailed on: {article.get('md_path', '?')} | {e}")
                failed += 1

            pbar.set_postfix(
                {"done": succeeded, "failed": failed, "timeout": timed_out},
                refresh=False,
            )

    print(f"\nSkipped (already done) : {skipped}")
    print(f"Processed this run     : {succeeded}")
    print(f"Timed out              : {timed_out}")
    print(f"Failed                 : {failed}")
    print(f"Log file               : {log_path}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    main()
    

# """
# Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
# Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
# """
# import os
# import logging
# import click
# import csv
# import re
# from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
# from datetime import datetime
# from langchain_ollama import OllamaLLM as Ollama
# from tqdm import tqdm
# from _constant_func import *


# MD_DIR = "/data/Deep_Angiography/ReACT-GPT/data/paper-set/processed-infer-set"

# TIMEOUT_SECONDS = 1800  # 10 minutes


# # --------------------------------------------------------------------------- #
# #  MD parsing helpers                                                          #
# # --------------------------------------------------------------------------- #

# def load_md_files(md_dir: str) -> list[dict]:
#     """
#     Read all .md files in md_dir and parse them into article dicts.
#     Expected md structure written by 00_preprocess_articles.py:
#         # <Title>
        
#         **Source:** <link>
#         **Original file:** <file_path>
        
#         ---
        
#         <body text>
#     """
#     articles = []
#     md_paths = sorted([
#         os.path.join(md_dir, f)
#         for f in os.listdir(md_dir)
#         if f.endswith(".md")
#     ])

#     for md_path in md_paths:
#         with open(md_path, "r", encoding="utf-8") as f:
#             raw = f.read()

#         # -- Title: first line starting with "# "
#         title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
#         title = title_match.group(1).strip() if title_match else ""

#         # -- Source link
#         link_match = re.search(r"\*\*Source:\*\*\s*(.+)$", raw, re.MULTILINE)
#         link = link_match.group(1).strip() if link_match else ""

#         # -- Original file path
#         file_match = re.search(r"\*\*Original file:\*\*\s*(.+)$", raw, re.MULTILINE)
#         file_path = file_match.group(1).strip() if file_match else md_path

#         # -- Body: everything after the "---" divider
#         parts = raw.split("---\n", maxsplit=1)
#         body = parts[1].strip() if len(parts) > 1 else raw.strip()

#         articles.append({
#             "Title":     title,
#             "Link":      link,
#             "File-path": file_path,
#             "Text":      body,
#             "md_path":   md_path,
#         })

#     return articles


# # --------------------------------------------------------------------------- #
# #  CLI                                                                         #
# # --------------------------------------------------------------------------- #

# @click.command()
# @click.option(
#     "--model_name",
#     default="llama3.1:latest",
#     type=str,
#     help="Ollama model to use for inference.",
# )
# @click.option(
#     "--temp",
#     default=0,
#     type=int,
#     help="Randomness of the model (temperature).",
# )
# @click.option(
#     "--prompting_method",
#     default="IP",
#     type=click.Choice(allowable_prompting_methods),
#     help="Prompting strategy: IP (Instruction), CoT (Chain-of-Thought), RA (Retrieval-Augmented).",
# )
# @click.option(
#     "--reports_to_process",
#     default=-1,
#     type=int,
#     help="Number of articles to process. -1 means all.",
# )
# @click.option(
#     "--md_dir",
#     default=MD_DIR,
#     type=click.Path(exists=True, file_okay=False),
#     help="Directory containing preprocessed .md files.",
# )
# def main(model_name, prompting_method, reports_to_process, temp, md_dir):
#     print(f"Model            : {model_name}")
#     print(f"Temperature      : {temp}")
#     print(f"Prompting method : {prompting_method}")
#     print(f"MD directory     : {md_dir}")
#     print(f"Timeout          : {TIMEOUT_SECONDS}s ({TIMEOUT_SECONDS // 60} min) per article")

#     # -- Select question template
#     question = {
#         "IP":  IP_template,
#         "CoT": CoT_template,
#         "RA":  RA_template,
#     }[prompting_method]

#     # -- Load articles from .md files
#     articles = load_md_files(md_dir)

#     if not articles:
#         print(f"No .md files found in {md_dir}. Exiting.")
#         return

#     if reports_to_process > 0:
#         articles = articles[:reports_to_process]

#     print(f"Articles to process: {len(articles)}")

#     # -- Set up log CSV
#     log_dir = "local_history"
#     safe_model_name = re.sub(r"[/:\\]", "_", model_name)
#     log_file = (
#         f"{prompting_method}{temp}{safe_model_name}"
#         f"{reports_to_process}"
#         f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
#     )
#     os.makedirs(log_dir, exist_ok=True)
#     log_path = os.path.join(log_dir, log_file)

#     with open(log_path, mode="w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "timestamp", "venue", "article_title",
#             "article_link", "answer", "model_name"
#         ])

#     # -- Inference loop
#     # timeout=TIMEOUT_SECONDS guards at the HTTP/client level
#     ollama = Ollama(model=model_name, temperature=temp, timeout=TIMEOUT_SECONDS)
#     logging.getLogger().setLevel(logging.ERROR)

#     succeeded = 0
#     failed    = 0
#     timed_out = 0

#     with tqdm(articles, desc="Running inference", unit="article", dynamic_ncols=True) as pbar:
#         for article in pbar:
#             try:
#                 query = prompt_template + article["Text"] + question

#                 # ThreadPoolExecutor acts as a hard safety net in case the
#                 # HTTP-level timeout is swallowed internally by the client.
#                 with ThreadPoolExecutor(max_workers=1) as executor:
#                     future = executor.submit(ollama.invoke, query)
#                     try:
#                         response = future.result(timeout=TIMEOUT_SECONDS)
#                     except FuturesTimeout:
#                         tqdm.write(
#                             f"\nTimed out ({TIMEOUT_SECONDS // 60} min) on: "
#                             f"{article.get('md_path', '?')}"
#                         )
#                         timed_out += 1
#                         failed += 1
#                         continue

#                 response = clean_response(response)

#                 with open(log_path, mode="a", newline="", encoding="utf-8") as f:
#                     writer = csv.writer(f)
#                     writer.writerow([
#                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                         article["File-path"],
#                         article["Title"],
#                         article["Link"],
#                         response,
#                         model_name,
#                     ])

#                 succeeded += 1

#             except Exception as e:
#                 tqdm.write(f"\nFailed on: {article.get('md_path', '?')} | {e}")
#                 failed += 1

#             pbar.set_postfix(
#                 {"done": succeeded, "failed": failed, "timeout": timed_out},
#                 refresh=False
#             )

#     print(f"\nTotal processed  : {succeeded}")
#     print(f"Total timed out  : {timed_out}")
#     print(f"Total failed     : {failed}")
#     print(f"Log written to   : {log_path}")


# if __name__ == "__main__":
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
#         level=logging.INFO,
#     )
#     main()
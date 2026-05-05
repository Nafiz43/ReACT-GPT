"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
import os
import re
import pymupdf4llm
import multiprocessing as mp
from _constant_func import *
import signal
from tqdm import tqdm


OUTPUT_DIR  = "/data/Deep_Angiography/ReACT-GPT/data/paper-set/processed-infer-set"
LOCAL_DIR   = "/data/Deep_Angiography/ReACT-GPT/data/paper-set/infer-set/"

def timeout_handler(signum, frame):
    print("Warning: The function took too long but will continue.")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(20)

pdf_files   = get_pdfs(LOCAL_DIR)
counter     = mp.Value('i', 0)       # shared success counter — used for sequential file naming
num_workers = max(mp.cpu_count() - 10, 1)

print(f"Number of workers : {num_workers}")
print(f"Total PDFs found  : {len(pdf_files)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def safe_extract_title(markdown_text: str) -> str:
    """Always returns a plain string, never a list."""
    if not markdown_text:
        return ""
    matches = re.findall(r'^# (.+)', markdown_text, re.MULTILINE)
    return matches[0].strip() if matches else ""


def safe_extract_link(markdown_text: str) -> str:
    """Always returns a plain string, never a list."""
    if not markdown_text:
        return ""
    matches = re.findall(r'https?://[^\s\)]+', markdown_text)
    return matches[0].strip() if matches else ""


# --------------------------------------------------------------------------- #
#  Core processing                                                             #
# --------------------------------------------------------------------------- #

def parse_article(args):
    """
    Parse a single PDF and return the md content + metadata.
    Returns (pdf_path, article_path, article_title, article_link, md_content)
    on success, or (pdf_path, None, ...) on failure.
    Writing to disk is done in the main process so we can assign
    sequential filenames without races.
    """
    pdf_path, = args   # unpack single-element tuple
    article_path = pdf_path.replace(LOCAL_DIR, "").strip("/")

    try:
        # -- Step 1: PDF → markdown
        try:
            markdown_text = pymupdf4llm.to_markdown(pdf_path)
        except Exception as inner_e:
            signal.alarm(0)
            tqdm.write(f"[MARKDOWN FAIL] {article_path} | {inner_e}")
            return (pdf_path, None, None, None, None)

        if not markdown_text or not markdown_text.strip():
            tqdm.write(f"[EMPTY MARKDOWN] {article_path}")
            return (pdf_path, None, None, None, None)

        # -- Step 2: Clean and extract fields
        cleaned_text  = clean_article_text(markdown_text)
        article_title = safe_extract_title(cleaned_text)
        article_link  = safe_extract_link(cleaned_text)

        # -- Step 3: Build markdown content (no filename logic here)
        pdf_stem   = os.path.splitext(os.path.basename(pdf_path))[0]
        md_content = "\n".join([
            f"# {article_title}" if article_title else f"# {pdf_stem}",
            "",
            f"**Source:** {article_link}" if article_link else "**Source:** NOT FOUND",
            f"**Original file:** {article_path}",
            "",
            "---",
            "",
            cleaned_text,
        ])

        return (pdf_path, article_path, article_title, article_link, md_content)

    except Exception as e:
        tqdm.write(f"[ERROR] {article_path} | {type(e).__name__}: {e}")
        return (pdf_path, None, None, None, None)


def process_file(args):
    """Worker entry point."""
    return parse_article(args)


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    succeeded    = 0
    failed       = 0
    failed_paths = []

    # Wrap each path in a tuple for pool.imap_unordered
    args = [(p,) for p in pdf_files]

    with mp.Pool(processes=num_workers) as pool:
        with tqdm(
            total=len(pdf_files),
            desc="Processing PDFs",
            unit="file",
            dynamic_ncols=True,
        ) as pbar:
            for pdf_path, article_path, title, link, md_content in pool.imap_unordered(
                process_file, args
            ):
                if md_content is not None:
                    # Sequential filename assigned in main process — no race condition
                    succeeded += 1
                    out_filename = f"{succeeded}.md"
                    out_path     = os.path.join(OUTPUT_DIR, out_filename)

                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(md_content)

                else:
                    failed += 1
                    failed_paths.append(pdf_path)

                pbar.set_postfix({"success": succeeded, "failed": failed}, refresh=False)
                pbar.update(1)

    # -- Summary
    print(f"\n{'='*55}")
    print(f"Total attempted  : {len(pdf_files)}")
    print(f"Total succeeded  : {succeeded}")
    print(f"Total failed     : {failed}")
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"{'='*55}")

    # -- Write failed paths log
    if failed_paths:
        failed_log = os.path.join(OUTPUT_DIR, "_failed_files.txt")
        with open(failed_log, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_paths))
        print(f"Failed file list : {failed_log}")
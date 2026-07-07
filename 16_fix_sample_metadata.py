"""Fix `article_title` / `article_link` (DOI) for the 200 human-validation
sample rows. An earlier pass (00_preprocess_articles.py) regex-scraped these
off the PDF-to-markdown conversion (first Markdown heading as title, first
URL-shaped string as link), which frequently grabbed running headers, page
numbers, or truncated footer text instead of the real title/DOI (e.g. item
H001's title was just "14", and its "DOI" was the dangling string
"http://www.seng").

This script re-derives both fields from the source PDFs themselves and
cross-checks them against Crossref (the DOI registration agency's public
API), for all 200 sampled rows:

  1. Extract page 1-2 text from the PDF (`pdftotext`) and look for an
     explicit DOI string printed on the page (ACM/IEEE papers print this in
     the header or footer).
  2. If found, resolve that DOI against the Crossref API
     (`api.crossref.org/works/<doi>`) to get the publisher's own canonical
     title, and to catch OCR/extraction typos (a malformed DOI 404s and
     falls through to step 3).
  3. If no DOI was printed on the page (or it didn't resolve), build a
     title candidate from the existing `article_title` field, PDF metadata
     (`pdfinfo` Title), or the first substantial line of page-1 text, and
     run a Crossref bibliographic search, accepting the top hit only if its
     title is a close text match (similarity ratio >= 0.55) to the
     candidate.
  4. Anything that still doesn't resolve is left with its original
     title/link but flagged `unresolved` in the report for manual lookup.

Usage:
    python3 16_fix_sample_metadata.py

Reads:
    results/human_validation/sample_200.csv

Writes:
    results/human_validation/sample_200.csv (overwritten in place;
        the pre-fix version is preserved as sample_200.csv.prefix_backup)
    results/human_validation/metadata_fix_report.csv (per-item method/
        confidence, for auditing)
    results/human_validation/metadata_fix_cache.json (cache of PDF path ->
        resolved result, so re-running this script doesn't re-hit the
        network for PDFs already resolved)

After running this, re-run 14_export_human_annotation_package.py and
15_build_annotator_workbooks.py to propagate the corrected title/DOI into
the annotator hand-off files.
"""

import csv
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher

DIR = "results/human_validation"
SAMPLE_CSV = os.path.join(DIR, "sample_200.csv")
BACKUP_CSV = os.path.join(DIR, "sample_200.csv.prefix_backup")
CACHE_PATH = os.path.join(DIR, "metadata_fix_cache.json")
REPORT_PATH = os.path.join(DIR, "metadata_fix_report.csv")

USER_AGENT = "ReACT-GPT-metadata-fix/1.0 (mailto:nikhan@ucdavis.edu)"
DOI_RE = re.compile(r'10\.\d{4,9}/[^\s"<>\]\)]+', re.IGNORECASE)
BAD_TITLE_RE = re.compile(r'^(microsoft word|untitled|document\d*|\d+)\b', re.IGNORECASE)
NAME_LINE_RE = re.compile(r'^[A-ZÀ-Ý][\w.À-ü\'-]+(\s+(and|,)?\s*[A-ZÀ-Ý][\w.À-ü\'-]+){1,4}$')
SIMILARITY_THRESHOLD = 0.55


def clean_title(t):
    return (t or "").strip().strip("*").strip()


def is_bad_title(t):
    t = clean_title(t)
    if len(t) < 6:
        return True
    if BAD_TITLE_RE.search(t):
        return True
    if t.replace(".", "").isdigit():
        return True
    return False


def pdf_text(path, pages=2):
    try:
        out = subprocess.run(
            ["pdftotext", "-f", "1", "-l", str(pages), path, "-"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout or ""
    except Exception:
        return ""


def pdf_info_title(path):
    try:
        out = subprocess.run(["pdfinfo", path], capture_output=True, text=True, timeout=15).stdout
        for line in out.splitlines():
            if line.startswith("Title:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


def http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, Exception):
        return None


def crossref_by_doi(doi):
    doi = doi.rstrip(".,;)")
    data = http_json(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    if not data or "message" not in data:
        return None
    item = data["message"]
    titles = item.get("title") or []
    if not titles:
        return None
    return {"title": titles[0], "doi": item.get("DOI", doi)}


def crossref_search(query):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": query, "rows": 3}
    )
    data = http_json(url)
    if not data:
        return []
    return data.get("message", {}).get("items", [])


def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def author_corroborates(hit, page_text_lower):
    """True if any Crossref-listed author's family name appears on the PDF
    page itself -- disambiguates generic/short titles (e.g. "Release
    Planning") that collide across many unrelated papers by title text
    alone."""
    for a in hit.get("author") or []:
        family = (a.get("family") or "").strip().lower()
        if len(family) > 2 and family in page_text_lower:
            return True
    return False


def resolve(row, cache):
    path = row["pdf_path"]
    if path in cache:
        return cache[path]

    text = pdf_text(path)
    # For author-name corroboration, only look at the byline area (first
    # ~12 lines of page 1) -- searching full page text (or page 2) risks
    # false-positive matches against unrelated names in the references list.
    byline_text = "\n".join(text.splitlines()[:12]).lower()
    result = None

    m = DOI_RE.search(text)
    if m:
        candidate_doi = m.group(0).rstrip(".,;)")
        cr = crossref_by_doi(candidate_doi)
        if cr:
            result = {"title": cr["title"], "doi": cr["doi"], "method": "page_doi+crossref_verify",
                      "confidence": "high"}
        time.sleep(0.4)

    if result is None:
        # Each candidate is (query_string, title_string_for_scoring). The
        # query is what's sent to Crossref; the title is what a hit's own
        # title is compared against for similarity (for a title+author
        # combined query, that's just the title part, not the author name).
        candidates = []
        old_title = clean_title(row["article_title"])
        if not is_bad_title(old_title):
            candidates.append((old_title, old_title))
        info_title = pdf_info_title(path)
        if not is_bad_title(info_title):
            candidates.append((info_title, info_title))
        # The true title isn't always the first line on the page -- book/
        # proceedings series front matter (e.g. "Handbook Software
        # Engineering ... Vol. 3") often precedes it. Try the first several
        # substantial lines as separate candidates rather than only the
        # first one, and let the best Crossref match across all of them win.
        lines = [l.strip() for l in text.splitlines()]
        title_lines = []
        line_candidates = 0
        for idx, line in enumerate(lines):
            if len(line) > 15 and not is_bad_title(line):
                candidates.append((line, line))
                title_lines.append((idx, line))
                line_candidates += 1
                if line_candidates >= 5:
                    break
        # A title line immediately followed by an author-name-shaped line
        # (e.g. "Software Release Planning" / "Günther Ruhe") is a strong
        # signal -- query title+author together, which disambiguates
        # generic/short titles far better than either alone.
        for idx, line in title_lines:
            for nxt in lines[idx + 1: idx + 3]:
                if nxt and NAME_LINE_RE.match(nxt):
                    candidates.append((f"{line} {nxt}", line))
                    break

        best = None  # (combined_score, sim, author_match, title, doi)
        for query, scoring_title in candidates:
            hits = crossref_search(query)
            time.sleep(0.4)
            for h in hits:
                h_titles = h.get("title") or []
                if not h_titles or not h.get("DOI"):
                    continue
                sim = similarity(scoring_title, h_titles[0])
                am = author_corroborates(h, byline_text)
                # Generic/short candidate strings, and long front-matter
                # lines (e.g. a book/series title preceding the actual
                # chapter title), both produce deceptively high textual
                # similarity to unrelated papers. Only trust a match that
                # isn't corroborated by an author name on the page if the
                # title similarity is near-exact.
                if not am and sim < 0.85:
                    continue
                score = sim + (0.2 if am else 0.0)
                if best is None or score > best[0]:
                    best = (score, sim, am, h_titles[0], h.get("DOI"))

        if best and best[1] >= SIMILARITY_THRESHOLD:
            confidence = "high" if (best[2] or best[1] >= 0.9) else "medium"
            result = {"title": best[3], "doi": best[4], "method": "crossref_search",
                      "confidence": confidence, "match_score": round(best[1], 3),
                      "author_corroborated": best[2]}
        else:
            result = {"title": old_title or info_title or "MANUAL_REVIEW_NEEDED",
                      "doi": "", "method": "unresolved", "confidence": "low"}

    cache[path] = result
    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1)
    return result


def main():
    if not os.path.exists(BACKUP_CSV):
        shutil.copy(SAMPLE_CSV, BACKUP_CSV)
        print(f"Backed up original to {BACKUP_CSV}")

    with open(SAMPLE_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as fh:
            cache = json.load(fh)

    report_rows = []
    resolved = unresolved = 0
    for i, row in enumerate(rows, start=1):
        result = resolve(row, cache)
        old_title, old_doi = row["article_title"], row["article_link"]
        row["article_title"] = result["title"]
        row["article_link"] = f"https://doi.org/{result['doi']}" if result["doi"] else old_doi
        if result["method"] == "unresolved":
            unresolved += 1
        else:
            resolved += 1
        report_rows.append({
            "item_id": row["item_id"], "pdf_path": row["pdf_path"],
            "old_title": old_title, "old_doi_or_url": old_doi,
            "new_title": row["article_title"], "new_doi": row["article_link"],
            "method": result["method"], "confidence": result["confidence"],
        })
        if i % 20 == 0:
            print(f"...{i}/{len(rows)} processed ({resolved} resolved, {unresolved} unresolved so far)")

    with open(SAMPLE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {SAMPLE_CSV} ({resolved} resolved, {unresolved} unresolved)")

    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "pdf_path", "old_title", "old_doi_or_url",
                                            "new_title", "new_doi", "method", "confidence"])
        w.writeheader()
        w.writerows(report_rows)
    print(f"Wrote {REPORT_PATH}")

    if unresolved:
        print(f"\n{unresolved} item(s) still need manual title/DOI lookup -- see rows with "
              f"method=unresolved in {REPORT_PATH}")


if __name__ == "__main__":
    main()

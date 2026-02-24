"""
Author: Su Min Park
Description: Collects raw paper metadata from arXiv API for cs.CY and cs.AI
             categories, 2018-2026. Scrapes each category separately, then
             merges and deduplicates by paper_id. No feature engineering —
             pure collection only.

Setup:
    pip install arxiv pandas

Run:
    python collect_arxiv_raw.py

Output:
    data/raw/arxiv_raw.csv  — one row per unique paper
"""

import arxiv
import pandas as pd
import os
import time

os.makedirs("data/raw", exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────

CATEGORIES  = ["cs.CY", "cs.AI"]
START_YEAR  = 2018
END_YEAR    = 2026
MAX_RESULTS = 9000  # per category
# the max is 10,000 for its limitation. 

# ── Collect one category ──────────────────────────────────────────────────────

def collect_category(client, cat):
    """Collect all papers from a single arXiv category within the year window."""
    print(f"\nCollecting {cat}...")

    search = arxiv.Search(
        query=f"cat:{cat}",
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    for result in client.results(search):
        pid  = result.entry_id.split("/")[-1].split("v")[0]  # strip version
        year = result.published.year

        if year < START_YEAR or year > END_YEAR:
            continue

        papers.append({
            "paper_id":       pid,
            "title":          result.title.replace("\n", " ").strip(),
            "abstract":       result.summary.replace("\n", " ").strip(),
            "authors":        "; ".join(a.name for a in result.authors),
            "n_authors":      len(result.authors),
            "published_date": result.published.strftime("%Y-%m-%d"),
            "year":           result.published.year,
            "month":          result.published.month,
            "primary_cat":    result.primary_category,
            "all_cats":       "; ".join(result.categories),
            "journal_ref":    result.journal_ref or "",
            "url":            result.entry_id,
        })

    print(f"  → {len(papers)} papers collected from {cat}")
    return pd.DataFrame(papers)


# ── Merge & deduplicate ───────────────────────────────────────────────────────

def merge_categories(dfs):
    """
    Concatenate per-category dataframes and deduplicate by paper_id.

    For cross-listed papers that appear in both scrapes, we keep the first
    occurrence for all fields EXCEPT all_cats — for that we take the union
    of all category tags from both scrapes to ensure nothing is lost.
    """
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows before dedup: {len(combined):,}")
    print(f"Unique paper IDs:        {combined['paper_id'].nunique():,}")
    print(f"Cross-listed papers:     {(combined.duplicated(subset='paper_id')).sum():,}")

    # For all_cats: take union of category strings across duplicate rows
    all_cats_merged = (
        combined.groupby("paper_id")["all_cats"]
        .apply(lambda x: "; ".join(sorted(set("; ".join(x).split("; ")))))
        .reset_index()
        .rename(columns={"all_cats": "all_cats"})
    )

    # For all other columns: keep first occurrence
    deduped = combined.drop_duplicates(subset="paper_id", keep="first").copy()
    deduped = deduped.drop(columns=["all_cats"])
    deduped = deduped.merge(all_cats_merged, on="paper_id", how="left")

    # Restore original column order
    cols = ["paper_id", "title", "abstract", "authors", "n_authors",
            "published_date", "year", "month", "primary_cat", "all_cats",
            "journal_ref", "url"]
    deduped = deduped[cols]

    return deduped


# ── Main ──────────────────────────────────────────────────────────────────────

def collect():
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3,
        num_retries=3,
    )

    # Scrape each category separately
    dfs = []
    for cat in CATEGORIES:
        df_cat = collect_category(client, cat)
        dfs.append(df_cat)
        time.sleep(2)  # pause between categories

    # Merge and deduplicate
    df = merge_categories(dfs)

    # Save
    out = "data/raw/arxiv_raw.csv"
    df.to_csv(out, index=False)

    # Summary
    cross = df[df["all_cats"].str.contains("cs.CY") & df["all_cats"].str.contains("cs.AI")]
    return df


if __name__ == "__main__":
    print(f"Collecting arXiv papers ({START_YEAR}–{END_YEAR})")
    print(f"Categories: {CATEGORIES}")
    # This takes ~20–30 minutes. Do not interrupt
    collect()
    print("\nDone. Raw data saved to data/raw/arxiv_raw.csv")
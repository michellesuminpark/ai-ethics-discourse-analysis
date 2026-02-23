"""
arXiv AI Ethics Paper Collector
=================================
Collects papers from arXiv using the official API (no scraping, no auth needed).
Targets cs.CY (Computers and Society) and cs.AI categories, filtered by
AI ethics keywords, from 2018–2024.

Setup:
  pip install arxiv pandas

Run:
  python collect_arxiv.py

Output:
  data/arxiv_raw.csv         — one row per paper (~3,000–8,000 papers)
  data/arxiv_sentences.csv   — one row per sentence (~30,000–80,000 rows)
"""

import arxiv
import pandas as pd
import re
import os
import time
from datetime import datetime

os.makedirs("data", exist_ok=True)

# ── Search queries ────────────────────────────────────────────────────────────
# arXiv API supports boolean queries. We run multiple focused queries
# and deduplicate by paper ID.

QUERIES = [
    # Core AI ethics framing
    'cat:cs.CY AND (ti:"AI ethics" OR ti:"algorithmic fairness" OR ti:"responsible AI")',
    'cat:cs.CY AND (ti:"machine learning" AND ti:"fairness")',
    'cat:cs.CY AND (ti:"bias" AND ti:"algorithm")',
    'cat:cs.CY AND (ti:"AI safety" OR ti:"value alignment")',
    'cat:cs.CY AND (ti:"explainability" OR ti:"transparency" OR ti:"accountability")',
    'cat:cs.CY AND (ti:"privacy" AND ti:"machine learning")',
    'cat:cs.CY AND (ti:"ethics" AND ti:"artificial intelligence")',
    # Broader cs.AI with ethics keywords
    'cat:cs.AI AND (ti:"fairness" OR ti:"bias" OR ti:"ethics")',
    'cat:cs.AI AND (ti:"responsible" OR ti:"trustworthy")',
    # Governance / policy framing
    'cat:cs.CY AND (ti:"AI governance" OR ti:"AI regulation" OR ti:"AI policy")',
    # Social impact framing
    'cat:cs.CY AND (ti:"algorithmic" AND (ti:"harm" OR ti:"discrimination" OR ti:"justice"))',
]

MAX_PER_QUERY = 500   # arXiv API limit per request is 2000, but 500 keeps it fast
                      # 11 queries × 500 = up to 5,500 before dedup

# ── Affiliation detection ─────────────────────────────────────────────────────
# arXiv abstracts don't include affiliations, but author names often appear
# alongside company names in the full metadata or we can heuristically detect
# industry-linked papers from known lab names in the abstract.

INDUSTRY_ORGS = [
    "google", "deepmind", "openai", "microsoft", "meta", "facebook",
    "amazon", "apple", "ibm", "nvidia", "hugging face", "anthropic",
    "twitter", "linkedin", "salesforce", "baidu", "alibaba", "tencent",
    "intel", "qualcomm", "samsung", "snap", "uber", "airbnb",
]

ACADEMIC_ORGS = [
    "university", "college", "institute", "school of", "department of",
    "laboratory", "mit", "stanford", "cmu", "carnegie mellon",
    "berkeley", "oxford", "cambridge", "harvard", "princeton",
    "cornell", "yale", "nyu", "uchicago", "columbia", "caltech",
]

def detect_affiliation(authors_str, abstract):
    """Heuristic: classify paper as industry/academic/mixed based on author affiliations."""
    # arXiv doesn't expose affiliations in the API, but we can check
    # if known industry lab names appear in the abstract acknowledgements
    combined = (authors_str + " " + abstract).lower()
    has_industry = any(org in combined for org in INDUSTRY_ORGS)
    has_academic = any(org in combined for org in ACADEMIC_ORGS)
    if has_industry and has_academic:
        return "mixed"
    elif has_industry:
        return "industry"
    elif has_academic:
        return "academic"
    else:
        return "unknown"

# ── Ethics framing categories ─────────────────────────────────────────────────
# Same schema as Stanford EthiCS — enables future comparison

ETHICS_CATEGORIES = {
    "has_fairness_bias":      r"fair|bias|discriminat|equalit|representation|disparit",
    "has_privacy":            r"privacy|surveillance|data protect|personal data|confidential",
    "has_safety_risk":        r"safety|risk|harm|dual.use|misuse|accident|robustness",
    "has_governance_policy":  r"governance|policy|regulat|law|compliance|accountability|audit",
    "has_values_autonomy":    r"values?|autonomy|trust|alignment|human.in.the.loop",
    "has_transparency":       r"transparency|explainab|interpretab|black.box|opaque",
    "has_social_power":       r"race|gender|disabilit|power|marginali|minorit|equity",
    "has_environment":        r"environment|climate|carbon|sustainab|energy",
    "has_labor":              r"labor|worker|job|employment|automation|displacement",
    "has_sociotechnical":     r"socio.technical|STS|social impact|society|community",
}

def label_row(text):
    """Return dict of binary ethics category labels."""
    labels = {}
    for col, pattern in ETHICS_CATEGORIES.items():
        labels[col] = int(bool(re.search(pattern, text, re.IGNORECASE)))
    labels["ethics_topic_count"] = sum(labels.values())
    return labels


# ── Year/period features ──────────────────────────────────────────────────────

def get_period(year):
    """Bin year into pre/post landmark AI ethics moments."""
    if year < 2018:   return "pre-2018"
    elif year <= 2019: return "2018-2019"  # early wave (Fairness/ML)
    elif year <= 2020: return "2020"        # pandemic + BLM + algorithmic justice
    elif year <= 2021: return "2021"        # EU AI Act draft
    elif year <= 2022: return "2022"        # ChatGPT pre-release
    elif year <= 2023: return "2023"        # ChatGPT explosion
    else:              return "2024"


# ── Collect ───────────────────────────────────────────────────────────────────

def collect():
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3,   # respectful delay
        num_retries=3,
    )

    seen_ids = set()
    all_papers = []

    for i, query in enumerate(QUERIES):
        print(f"\nQuery {i+1}/{len(QUERIES)}: {query[:80]}...")
        search = arxiv.Search(
            query=query,
            max_results=MAX_PER_QUERY,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        count = 0
        for result in client.results(search):
            pid = result.entry_id.split("/")[-1].split("v")[0]  # strip version
            if pid in seen_ids:
                continue
            seen_ids.add(pid)

            year = result.published.year
            if year < 2018 or year > 2024:
                continue   # keep 2018–2024 only

            authors_str = "; ".join(a.name for a in result.authors[:10])
            abstract    = result.summary.replace("\n", " ").strip()
            title       = result.title.replace("\n", " ").strip()

            # Primary category
            primary_cat = result.primary_category

            # All categories
            all_cats = "; ".join(result.categories)

            # Affiliation heuristic
            affiliation = detect_affiliation(authors_str, abstract)

            # Ethics labels
            combined = title + " " + abstract
            labels = label_row(combined)

            row = {
                "paper_id":       pid,
                "title":          title,
                "abstract":       abstract,
                "authors":        authors_str,
                "year":           year,
                "month":          result.published.month,
                "published_date": result.published.strftime("%Y-%m-%d"),
                "period":         get_period(year),
                "primary_cat":    primary_cat,
                "all_cats":       all_cats,
                "affiliation":    affiliation,
                "url":            result.entry_id,
                "journal_ref":    result.journal_ref or "",
                "n_authors":      len(result.authors),
                **labels,
            }
            all_papers.append(row)
            count += 1

        print(f"  → {count} new papers (total so far: {len(all_papers)})")
        time.sleep(1)

    df = pd.DataFrame(all_papers)
    df = df.drop_duplicates(subset="paper_id")

    out = "data/arxiv_raw.csv"
    df.to_csv(out, index=False)

    print(f"\n{'='*60}")
    print(f"✓ Saved {len(df)} papers → {out}")
    print(f"\n  Year distribution:")
    print(df["year"].value_counts().sort_index().to_string())
    print(f"\n  Affiliation distribution:")
    print(df["affiliation"].value_counts().to_string())
    print(f"\n  Ethics topic distribution:")
    for col in [c for c in df.columns if c.startswith("has_")]:
        n = df[col].sum()
        print(f"    {col:<35} {n:>4}/{len(df)}  ({n/len(df)*100:.0f}%)")

    return df


# ── Sentence segmentation ─────────────────────────────────────────────────────

def make_sentence_dataset(df):
    """
    Explode each abstract into individual sentences.
    Each sentence becomes one row, inheriting all paper metadata.
    This is your primary unit of analysis for NLP classification.
    Target: ~30,000–80,000 sentence rows.
    """
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    rows = []
    meta_cols = [c for c in df.columns if c != "abstract"]

    for _, paper in df.iterrows():
        sentences = nltk.sent_tokenize(paper["abstract"])
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) < 30:   # skip very short fragments
                continue
            sent_labels = label_row(sent)
            row = {col: paper[col] for col in meta_cols}
            row["sentence_index"] = i
            row["sentence"]       = sent
            row["sentence_len"]   = len(sent.split())
            # Override paper-level labels with sentence-level labels
            for col in ETHICS_CATEGORIES:
                row[col] = sent_labels[col]
            row["ethics_topic_count"] = sent_labels["ethics_topic_count"]
            rows.append(row)

    sent_df = pd.DataFrame(rows)
    out = "data/arxiv_sentences.csv"
    sent_df.to_csv(out, index=False)

    print(f"\n✓ Sentence dataset: {len(sent_df)} rows → {out}")
    print(f"  Avg sentences per abstract: {len(sent_df)/len(df):.1f}")
    return sent_df


if __name__ == "__main__":
    print("Collecting arXiv papers (this takes ~10–15 min)...")
    df = collect()
    print("\nBuilding sentence-level dataset...")
    make_sentence_dataset(df)
    print("\nDone! Your two datasets are in data/")
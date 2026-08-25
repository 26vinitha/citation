import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIG -- FullText only, all years 1993-2003, N=5 topics
# ============================================================
DATA_ROOT = Path(r"C:\Users\vinipumba\Desktop\dataset")
CITATION_DIR = DATA_ROOT / "citation count"
FULLTEXT_DIR = DATA_ROOT / "full text"

PAPERS_JSON = r"C:\Users\vinipumba\Desktop\citation-count-simple\try 2\all_papers_topics_fulltext_1993_2003.json"
OUTPUT_JSON = r"C:\Users\vinipumba\Desktop\citation-count-simple\try 2\tfidf_similarity_fulltext_1993_2003_n10.json"
OUTPUT_CSV = r"C:\Users\vinipumba\Desktop\citation-count-simple\try 2\tfidf_similarity_fulltext_1993_2003_n10.csv"

TARGET_YEAR = 2022
MIN_TRENDING_CITATIONS = 20
YEARS = list(range(1993, 2004))


def find_id_column(df):
    candidates = ["Article Id", "article_id", "Article ID", "ArticleId", "article id"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "article" in c.lower() or c.lower() == "id":
            return c
    raise KeyError(f"Could not find an article-ID column. Available columns: {list(df.columns)}")


def load_all_years_labels():
    """Load and combine all 11 citation-count files, zero-padding IDs, keeping the target-year column."""
    frames = []
    for y in YEARS:
        df = pd.read_excel(CITATION_DIR / f"{y}.xlsx")
        id_col = find_id_column(df)
        df[TARGET_YEAR] = pd.to_numeric(df[TARGET_YEAR], errors="coerce").fillna(0)
        df["pub_year"] = y
        df["aid_str"] = df[id_col].astype(str).str.zfill(7)
        frames.append(df[["aid_str", "pub_year", TARGET_YEAR]])
    return pd.concat(frames, ignore_index=True)


def calculate_tfidf_similarity(paper_texts, trending_doc):
    print("\nCalculating TF-IDF similarity...")
    all_documents = [trending_doc] + paper_texts
    print(f"  Total documents: {len(all_documents)}")
    print(f"  Trending topics document length: {len(trending_doc)} chars")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1
    )

    print("  Fitting TF-IDF vectorizer (this can take a while at full-corpus scale)...")
    tfidf_matrix = vectorizer.fit_transform(all_documents)
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

    print("  Calculating cosine similarities...")
    trending_vector = tfidf_matrix[0:1]
    paper_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(trending_vector, paper_vectors)[0]

    return similarities * 100


def main():
    print("=" * 70)
    print("CALCULATING TF-IDF SIMILARITY -- FullText, ALL YEARS 1993-2003")
    print("=" * 70)

    print("\nLoading citation labels for all 11 years...")
    labels = load_all_years_labels()
    print(f"Total papers across all years: {len(labels)}")

    trending_ids = set(labels.loc[labels[TARGET_YEAR] > MIN_TRENDING_CITATIONS, "aid_str"])
    print(f"Trending papers (>{MIN_TRENDING_CITATIONS} citations in {TARGET_YEAR}): {len(trending_ids)}")

    print("\nLoading extracted topics...")
    with open(PAPERS_JSON, 'r', encoding="utf-8") as f:
        topics_raw = json.load(f)
    print(f"Papers with extracted topics: {len(topics_raw)}")

    # Safety check: this script expects N=10 topics per paper. Verify a
    # sample before proceeding, so an accidentally-repointed file (e.g. an
    # old N=5 extraction) fails loudly here instead of silently producing
    # a mislabeled "N=10" result set downstream.
    sample_ids = list(topics_raw.keys())[:50]
    topic_counts = [len(topics_raw[aid]["topics"]) for aid in sample_ids]
    most_common_count = max(set(topic_counts), key=topic_counts.count)
    if most_common_count != 10:
        raise ValueError(
            f"PAPERS_JSON at {PAPERS_JSON} appears to contain {most_common_count} "
            f"topics per paper (checked a sample of {len(sample_ids)}), not 10. "
            f"This looks like N=5 data, not the N=10 extraction this script expects. "
            f"Stopping before building a mislabeled result set -- verify the file, "
            f"or set EXPECTED_N_TOPICS below if this is intentional."
        )
    print(f"Verified: sampled papers have {most_common_count} topics each (matches expected N=10).")

    trending_topic_words = []
    for aid, rec in topics_raw.items():
        if aid in trending_ids:
            trending_topic_words.extend([t for t in rec["topics"] if t and t != "Unknown Topic"])
    print(f"Trending topic terms collected: {len(trending_topic_words)}")
    if not trending_topic_words:
        print("WARNING: no trending topic terms found -- check MIN_TRENDING_CITATIONS "
              "or ID overlap between the two files.")
    trending_doc = " ".join(trending_topic_words)

    print("\nReading FullText files for all extracted papers (this will take a while)...")
    papers = []
    missing = 0
    # need pub_year per article to find the right year subfolder
    year_lookup = dict(zip(labels["aid_str"], labels["pub_year"]))
    for i, aid in enumerate(topics_raw.keys(), 1):
        year = year_lookup.get(aid)
        if year is None:
            missing += 1
            continue
        path = FULLTEXT_DIR / str(year) / f"{aid}.txt"
        if not path.exists():
            missing += 1
            continue
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        papers.append({"article_id": aid, "text": text})
        if i % 2000 == 0:
            print(f"  ...{i}/{len(topics_raw)} read")
    print(f"Papers with text loaded: {len(papers)} (missing: {missing})")

    paper_texts = [p["text"] for p in papers]
    similarities = calculate_tfidf_similarity(paper_texts, trending_doc)

    results = []
    for i, paper in enumerate(papers):
        results.append({
            'article_id': paper['article_id'],
            'tfidf_similarity_percentage': round(float(similarities[i]), 4)
        })

    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Mean similarity: {np.mean(similarities):.2f}%")
    print(f"Median similarity: {np.median(similarities):.2f}%")
    print(f"Min similarity: {np.min(similarities):.2f}%")
    print(f"Max similarity: {np.max(similarities):.2f}%")
    print(f"Std deviation: {np.std(similarities):.2f}%")

    sorted_results = sorted(results, key=lambda x: x['tfidf_similarity_percentage'], reverse=True)
    print("\nTop 10 papers by TF-IDF similarity:")
    for i, result in enumerate(sorted_results[:10], 1):
        print(f"  {i}. Article {result['article_id']}: {result['tfidf_similarity_percentage']:.2f}%")

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    output_data = {
        'total_papers': len(results),
        'trending_topics_count': len(set(t.lower() for t in trending_topic_words)),
        'statistics': {
            'mean': round(float(np.mean(similarities)), 4),
            'median': round(float(np.median(similarities)), 4),
            'min': round(float(np.min(similarities)), 4),
            'max': round(float(np.max(similarities)), 4),
            'std': round(float(np.std(similarities)), 4)
        },
        'results': results
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"JSON saved to: {OUTPUT_JSON}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV saved to: {OUTPUT_CSV}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

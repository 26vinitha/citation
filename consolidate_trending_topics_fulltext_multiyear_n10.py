import json
from collections import Counter
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG -- FullText only, all years 1993-2003, N=5 topics
# ============================================================
DATA_ROOT = Path(r"C:\Users\vinipumba\Desktop\dataset")
CITATION_DIR = DATA_ROOT / "citation count"

INPUT_JSON = r"C:\Users\vinipumba\Desktop\citation-count-simple\try 2\all_papers_topics_fulltext_1993_2003.json"
OUTPUT_JSON = r"C:\Users\vinipumba\Desktop\citation-count-simple\trending_topics_fulltext_1993_2003_n10.json"

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
    """Load and combine all 11 citation-count files, zero-padding IDs."""
    frames = []
    for y in YEARS:
        df = pd.read_excel(CITATION_DIR / f"{y}.xlsx")
        id_col = find_id_column(df)
        df[TARGET_YEAR] = pd.to_numeric(df[TARGET_YEAR], errors="coerce").fillna(0)
        df["pub_year"] = y
        df["aid_str"] = df[id_col].astype(str).str.zfill(7)
        frames.append(df[["aid_str", "pub_year", TARGET_YEAR]])
    return pd.concat(frames, ignore_index=True)


def main():
    """
    Consolidate technical topics from TRENDING papers only (>20 citations
    in 2022) into a frequency-ranked trending-topics list, across all
    eleven publication-year cohorts (1993-2003), FullText only.
    """
    print("Loading citation labels for all 11 years...")
    labels = load_all_years_labels()
    print(f"Total papers across all years: {len(labels)}")

    trending_ids = set(labels.loc[labels[TARGET_YEAR] > MIN_TRENDING_CITATIONS, "aid_str"])
    print(f"Trending papers (>{MIN_TRENDING_CITATIONS} citations in {TARGET_YEAR}): {len(trending_ids)}")

    print(f"\nLoading technical topics from {INPUT_JSON}")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        topics_raw = json.load(f)
    print(f"Total papers with extracted topics: {len(topics_raw)}")

    sample_ids = list(topics_raw.keys())[:50]
    topic_counts = [len(topics_raw[aid]["topics"]) for aid in sample_ids]
    most_common_count = max(set(topic_counts), key=topic_counts.count)
    if most_common_count != 10:
        raise ValueError(
            f"INPUT_JSON at {INPUT_JSON} appears to contain {most_common_count} "
            f"topics per paper (checked a sample of {len(sample_ids)}), not 10. "
            f"Verify this is the N=10 extraction file before proceeding."
        )
    print(f"Verified: sampled papers have {most_common_count} topics each (matches expected N=10).")

    trending_results = []
    for aid, rec in topics_raw.items():
        if aid in trending_ids:
            trending_results.append({
                'article_id': aid,
                'pub_year': rec.get('pub_year'),
                'technical_topics': [t for t in rec['topics'] if t and t != "Unknown Topic"],
            })
    print(f"Trending papers with topics available: {len(trending_results)}")
    if not trending_results:
        print("WARNING: no overlap between trending_ids and papers with extracted topics -- "
              "double check MIN_TRENDING_CITATIONS and that IDs match between the two files.")

    all_topics = []
    topic_to_articles = {}

    for result in trending_results:
        article_id = result['article_id']
        for topic in result['technical_topics']:
            all_topics.append(topic)
            if topic not in topic_to_articles:
                topic_to_articles[topic] = []
            topic_to_articles[topic].append(article_id)

    topic_counts = Counter(all_topics)

    trending_topics = []
    n_trending_papers = len(trending_results) if trending_results else 1
    for topic, count in topic_counts.most_common():
        trending_topics.append({
            'topic': topic,
            'frequency': count,
            'percentage': round((count / n_trending_papers) * 100, 2),
            'mentioned_in_articles': topic_to_articles[topic]
        })

    # breakdown of trending papers by publication year -- useful for the paper
    by_year = Counter(r['pub_year'] for r in trending_results)

    output_data = {
        'metadata': {
            'total_articles': len(trending_results),
            'total_unique_topics': len(topic_counts),
            'total_topic_mentions': len(all_topics),
            'topics_per_paper': 10,
            'years_covered': YEARS,
            'trending_papers_by_year': dict(sorted(by_year.items())),
            'year_filter': TARGET_YEAR,
            'citation_threshold': MIN_TRENDING_CITATIONS,
            'timestamp': pd.Timestamp.now().isoformat()
        },
        'trending_topics': trending_topics
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("TRENDING TOPICS SUMMARY -- ALL YEARS 1993-2003 (FullText)")
    print(f"{'='*70}\n")

    print(f"Trending articles analyzed: {len(trending_results)}")
    print(f"Total unique topics: {len(topic_counts)}")
    print(f"Total topic mentions: {len(all_topics)}")
    print(f"\nTrending papers by publication year:")
    for y in YEARS:
        print(f"  {y}: {by_year.get(y, 0)}")

    print(f"\nTop 15 Most Frequent Topics:\n")
    for i, item in enumerate(trending_topics[:15], 1):
        print(f"{i:2d}. {item['topic']:<40} - {item['frequency']} articles ({item['percentage']}%)")

    print(f"\n{'='*70}")
    print(f"Trending topics saved to: {OUTPUT_JSON}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

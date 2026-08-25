import pandas as pd
import os
import sys
import json
import requests
import time
from pathlib import Path


def safe_print(*args, **kwargs):
    """print() that never crashes the whole script on a Windows console
    encoding error -- replaces characters the console can't display
    instead of raising UnicodeEncodeError."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = [str(a).encode(sys.stdout.encoding or "utf-8", errors="replace")
                     .decode(sys.stdout.encoding or "utf-8", errors="replace") for a in args]
        print(*safe_args, **kwargs)

# ============================================================
# CONFIG -- FullText only, all years 1993-2003
# ============================================================
DATA_ROOT = Path(r"C:\Users\vinipumba\Desktop\dataset")
CITATION_DIR = DATA_ROOT / "citation count"
FULLTEXT_DIR = DATA_ROOT / "full text"

OUTPUT_JSON = r"C:\Users\vinipumba\Desktop\citation-count-simple\try 2\all_papers_topics_fulltext_1993_2003_n10.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_RETRIES = 3
N_TOPICS = 10
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


def read_article_text(aid_str, year, text_dir):
    file_path = text_dir / str(year) / f"{aid_str}.txt"
    if not file_path.exists():
        return None
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"  Error reading {aid_str}: {e}")
        return None


def clean_topic(topic):
    topic = topic.strip()
    topic = topic.lstrip('0123456789.*-) ')
    topic = topic.replace('*', '')
    return topic.strip()


def extract_topics_with_ollama(text):
    prompt = f"""You are a physics research expert. Analyze this physics paper and extract EXACTLY {N_TOPICS} specific technical topics, theories, or phenomena discussed.

Focus on:
- Specific physical theories (e.g., "Virasoro Algebra", "Bethe Ansatz")
- Mathematical frameworks (e.g., "Chern-Simons Theory", "Conformal Field Theory")
- Specific phenomena (e.g., "Quantum Hall Effect", "Topological Invariants")

DO NOT include:
- Generic words like "String", "Theory", "Model", "Physics"
- Author names or paper metadata
- General concepts without specificity

Paper text:
{text[:4000]}

Return ONLY {N_TOPICS} topics, one per line, no numbering or formatting."""

    payload = {"model": "llama3.2", "prompt": prompt, "stream": False, "temperature": 0.3}

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            response_text = result.get('response', '').strip()

            topics = []
            for line in response_text.split('\n'):
                cleaned = clean_topic(line)
                if cleaned and len(cleaned) > 3:
                    topics.append(cleaned)

            if len(topics) == N_TOPICS:
                return topics
            if len(topics) > N_TOPICS:
                return topics[:N_TOPICS]
            print(f"    Got {len(topics)} topics (need {N_TOPICS}), retrying...")

        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    return ["Unknown Topic"] * N_TOPICS


def load_all_years_labels():
    """Load and combine all 11 citation-count files, zero-padding IDs."""
    frames = []
    for y in YEARS:
        df = pd.read_excel(CITATION_DIR / f"{y}.xlsx")
        id_col = find_id_column(df)
        df["pub_year"] = y
        df["aid_str"] = df[id_col].astype(str).str.zfill(7)
        frames.append(df[["aid_str", "pub_year"]])
    return pd.concat(frames, ignore_index=True)


def main():
    print("=" * 70)
    print(f"EXTRACTING TOP-{N_TOPICS} TOPICS -- FullText, ALL YEARS 1993-2003")
    print("=" * 70)

    print("\nLoading citation labels for all 11 years...")
    labels = load_all_years_labels()
    print(f"Total papers across all years: {len(labels)}")

    # resume support -- read what's already done
    out_path = Path(OUTPUT_JSON)
    results = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} papers already processed.")

    print("\nChecking for FullText files...")
    to_process = []
    missing = 0
    for _, row in labels.iterrows():
        aid, year = row["aid_str"], row["pub_year"]
        if aid in results:
            continue
        if (FULLTEXT_DIR / str(year) / f"{aid}.txt").exists():
            to_process.append((aid, year))
        else:
            missing += 1
    print(f"Papers to process: {len(to_process)}  (already done: {len(results)}, missing files: {missing})")

    n_errors = 0
    for idx, (aid, year) in enumerate(to_process, 1):
        try:
            safe_print(f"\n[{idx}/{len(to_process)}] Processing {aid} ({year})...")
            text = read_article_text(aid, year, FULLTEXT_DIR)
            if not text:
                safe_print("  No text found, skipping...")
                continue

            safe_print(f"  Text length: {len(text)} characters -- extracting topics...")
            topics = extract_topics_with_ollama(text)

            results[aid] = {'article_id': aid, 'pub_year': int(year), 'topics': topics, 'text_length': len(text)}
            safe_print(f"  Topics: {topics}")

        except Exception as e:
            # Never let a single bad paper kill a multi-hour run. Log it,
            # save a placeholder so it isn't retried forever, and move on.
            n_errors += 1
            safe_print(f"  ERROR on {aid}: {type(e).__name__}: {e}  -- skipping this paper, continuing run.")
            results[aid] = {'article_id': aid, 'pub_year': int(year), 'topics': ["Unknown Topic"] * N_TOPICS,
                             'text_length': 0, 'error': str(e)}

        # checkpoint every 50 papers, and also right after any error --
        # so a crash right after this point loses at most 1 paper, not 50
        if idx % 50 == 0 or n_errors:
            safe_print(f"\n  Saving checkpoint at {idx} new papers ({len(results)} total, {n_errors} errors so far)...")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            n_errors = 0  # reset so we go back to the normal every-50 cadence until the next error

    safe_print("\n" + "=" * 70)
    safe_print("SAVING FINAL RESULTS")
    safe_print("=" * 70)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    safe_print(f"\nTotal papers processed: {len(results)}")
    safe_print(f"Output saved to: {OUTPUT_JSON}")
    safe_print("=" * 70)


if __name__ == "__main__":
    main()

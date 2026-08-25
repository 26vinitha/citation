import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG -- FullText only, all years 1993-2003
# ============================================================
DATA_ROOT = Path(r"C:\Users\vinipumba\Desktop\dataset")
CITATION_DIR = DATA_ROOT / "citation count"

TFIDF_CSV = r"C:\Users\vinipumba\Desktop\citation-count-simple\try 2\tfidf_similarity_fulltext_1993_2003_n10.csv"
OUTPUT_CSV = r"C:\Users\vinipumba\Desktop\citation-count-simple\try 2\mlp_training_data_fulltext_1993_2003_n10.csv"

TARGET_YEAR = 2022
FEATURE_YEARS = list(range(1993, 2022))   # 1993-2021 -- 2022 is the target, never a feature
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


def main():
    print("Loading citation labels for all 11 years...")
    frames = []
    for y in YEARS:
        df = pd.read_excel(CITATION_DIR / f"{y}.xlsx")
        id_col = find_id_column(df)
        df["pub_year"] = y
        df["aid_str"] = df[id_col].astype(str).str.zfill(7)
        frames.append(df)

    # FIX: concatenate FIRST, then fillna on the full unified column set.
    # Doing fillna per-file (before concat) misses NaN values that pd.concat
    # itself introduces when different years' files have different column
    # sets (e.g. a paper published in 2003 has no 1993-2002 columns in its
    # own source file at all -- those become NaN only after concat, and a
    # per-file fillna can never catch them).
    cite_df = pd.concat(frames, ignore_index=True)
    print(f"Total papers across all years: {len(cite_df)}")

    dup = cite_df["aid_str"].duplicated().sum()
    if dup:
        print(f"Warning: {dup} duplicate article IDs across years -- keeping first occurrence.")
        cite_df = cite_df.drop_duplicates(subset="aid_str", keep="first")

    # Ensure every feature year + the target year exist as columns (even if
    # completely absent from every source file, which shouldn't happen but
    # would otherwise crash the next step) and are numeric.
    all_needed_year_cols = FEATURE_YEARS + [TARGET_YEAR]
    n_nan_before = 0
    for c in all_needed_year_cols:
        if c not in cite_df.columns:
            print(f"  NOTE: year column {c} was not present in ANY source file -- creating as all-zero.")
            cite_df[c] = 0.0
        else:
            cite_df[c] = pd.to_numeric(cite_df[c], errors="coerce")
            n_nan_before += cite_df[c].isna().sum()

    print(f"Total NaN cells across all year columns before fill: {n_nan_before}")
    print("Filling with 0 -- this is correct here: a NaN in a pre-publication "
          "year column means the paper did not exist yet, so it had zero citations.")
    cite_df[all_needed_year_cols] = cite_df[all_needed_year_cols].fillna(0)

    print("Loading TF-IDF similarity scores...")
    tfidf_df = pd.read_csv(TFIDF_CSV, dtype={"article_id": str})
    tfidf_df["_id_key"] = tfidf_df["article_id"].astype(str).str.zfill(7)

    print("Merging...")
    merged = cite_df.merge(
        tfidf_df[["_id_key", "tfidf_similarity_percentage"]],
        left_on="aid_str", right_on="_id_key", how="inner"
    )
    print(f"Papers with both citation history and similarity score: {len(merged)}")

    out = pd.DataFrame()
    out["article_id"] = merged["aid_str"]
    out["pub_year"] = merged["pub_year"]
    for y in FEATURE_YEARS:
        out[f"citations_{y}"] = merged[y].astype(float)
    out["tfidf_similarity"] = merged["tfidf_similarity_percentage"].astype(float)
    out["citations_2022_target"] = merged[TARGET_YEAR].astype(float)

    # final safety check -- confirm zero NaN remain anywhere in the output
    remaining_nan = out.isna().sum().sum()
    print(f"\nRemaining NaN in final output: {remaining_nan}")
    if remaining_nan > 0:
        print("WARNING: NaN still present after fill -- do not proceed to training "
              "until this is resolved. Columns with NaN:")
        print(out.isna().sum()[out.isna().sum() > 0])

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out)} rows, {len(out.columns)} columns to:")
    print(f"  {OUTPUT_CSV}")
    print(f"\nPapers per publication year:")
    print(out["pub_year"].value_counts().sort_index())
    print("\nReady to run train_all_splits_5models_torch_multiyear.py against this file.")


if __name__ == "__main__":
    main()

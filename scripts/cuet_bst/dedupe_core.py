from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .intelligence import normalize_question_identity, question_hash, source_tier, source_weight, year_weight
from .settings import PROCESSED_DIR


@dataclass
class UnionFind:
    parent: list[int]

    @classmethod
    def create(cls, size: int) -> "UnionFind":
        return cls(list(range(size)))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def run_deduplication(logger: logging.Logger | None = None) -> dict[str, int]:
    logger = logger or logging.getLogger(__name__)
    questions_path = PROCESSED_DIR / "questions.csv"
    if not questions_path.exists():
        raise FileNotFoundError("Missing data/processed/questions.csv. Run python scripts/process_questions.py first.")
    df = pd.read_csv(questions_path).fillna("")
    if df.empty:
        _write_empty_outputs(df)
        return {"raw_rows": 0, "unique_questions": 0, "duplicate_rows": 0}

    df = df.copy()
    df["normalized_question_key"] = df.apply(normalize_question_identity, axis=1)
    df["exact_question_hash"] = df.apply(question_hash, axis=1)
    uf = UnionFind.create(len(df))

    for _, group in df.groupby("exact_question_hash"):
        indices = list(group.index)
        for index in indices[1:]:
            uf.union(indices[0], index)

    _fuzzy_union(df, uf, logger)

    df["_root"] = [uf.find(i) for i in range(len(df))]
    root_to_id = {
        root: f"cq_{rank:06d}_{df.loc[group.index[0], 'exact_question_hash'][:10]}"
        for rank, (root, group) in enumerate(df.groupby("_root"), start=1)
    }
    df["canonical_question_id"] = df["_root"].map(root_to_id)
    df["source_tier"] = df.apply(lambda row: source_tier(str(row.get("source", "")), str(row.get("source_url", ""))), axis=1)
    df["source_weight"] = df.apply(lambda row: source_weight(str(row.get("source", "")), str(row.get("source_url", ""))), axis=1)
    df["year_weight"] = df["year"].map(year_weight)
    df["weighted_frequency_score"] = df["source_weight"]
    df["recency_weighted_score"] = df["source_weight"] * df["year_weight"]

    duplicate_meta = (
        df.groupby("canonical_question_id")
        .agg(
            duplicate_group_size=("question_id", "size"),
            duplicate_source_urls=("source_url", lambda values: " | ".join(sorted({str(value) for value in values if str(value).strip()}))),
        )
        .reset_index()
    )
    df = df.merge(duplicate_meta, on="canonical_question_id", how="left")
    duplicate_rows = int((df["duplicate_group_size"] > 1).sum())

    canonical = _choose_canonical_rows(df)
    duplicate_groups = _duplicate_groups(df)
    data_quality = _data_quality(df, canonical)

    df.drop(columns=["_root"], errors="ignore").to_csv(PROCESSED_DIR / "questions_with_canonical_ids.csv", index=False)
    canonical.to_csv(PROCESSED_DIR / "questions_canonical.csv", index=False)
    duplicate_groups.to_csv(PROCESSED_DIR / "duplicate_groups.csv", index=False)
    data_quality.to_csv(PROCESSED_DIR / "data_quality_summary.csv", index=False)
    logger.info("Deduped %s raw rows to %s canonical questions", len(df), len(canonical))
    return {"raw_rows": len(df), "unique_questions": len(canonical), "duplicate_rows": duplicate_rows}


def _fuzzy_union(df: pd.DataFrame, uf: UnionFind, logger: logging.Logger) -> None:
    texts = df["normalized_question_key"].astype(str).tolist()
    if len(texts) < 2:
        return
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    # Chunked full similarity keeps memory reasonable for this dataset and avoids extra deps.
    threshold = 0.94
    for start in range(0, len(texts), 500):
        sims = cosine_similarity(matrix[start : start + 500], matrix)
        rows, cols = np.where(sims >= threshold)
        for local_row, col in zip(rows, cols):
            left = start + int(local_row)
            right = int(col)
            if left >= right:
                continue
            if _compatible_duplicate(df.iloc[left], df.iloc[right], float(sims[local_row, col])):
                uf.union(left, right)


def _compatible_duplicate(left: pd.Series, right: pd.Series, similarity: float) -> bool:
    if similarity >= 0.985:
        return True
    left_year = str(left.get("year", "")).strip()
    right_year = str(right.get("year", "")).strip()
    if left_year and right_year and left_year != right_year:
        return False
    left_options = {str(left.get(column, "")).lower().strip() for column in ["option_a", "option_b", "option_c", "option_d"] if str(left.get(column, "")).strip()}
    right_options = {str(right.get(column, "")).lower().strip() for column in ["option_a", "option_b", "option_c", "option_d"] if str(right.get(column, "")).strip()}
    if left_options and right_options:
        overlap = len(left_options & right_options) / max(len(left_options | right_options), 1)
        return overlap >= 0.50
    return similarity >= 0.97


def _choose_canonical_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby("canonical_question_id", sort=False):
        group = group.copy()
        group["_question_len"] = group["question_text"].astype(str).str.len()
        chosen = group.sort_values(["source_weight", "_question_len"], ascending=False).iloc[0].to_dict()
        chosen["duplicate_group_size"] = int(group["duplicate_group_size"].iloc[0])
        chosen["duplicate_source_urls"] = group["duplicate_source_urls"].iloc[0]
        chosen["weighted_frequency_score"] = round(float(group["source_weight"].max()), 4)
        chosen["recency_weighted_score"] = round(float((group["source_weight"] * group["year_weight"]).max()), 4)
        rows.append(chosen)
    return pd.DataFrame(rows).drop(columns=["_question_len", "_root"], errors="ignore")


def _duplicate_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for canonical_id, group in df.groupby("canonical_question_id"):
        rows.append(
            {
                "canonical_question_id": canonical_id,
                "duplicate_group_size": len(group),
                "years": ", ".join(sorted({str(value) for value in group["year"] if str(value).strip()})),
                "sources": ", ".join(sorted({str(value) for value in group["source"] if str(value).strip()})),
                "source_urls": " | ".join(sorted({str(value) for value in group["source_url"] if str(value).strip()})),
                "question_text": str(group.iloc[0].get("question_text", ""))[:700],
            }
        )
    return pd.DataFrame(rows).sort_values("duplicate_group_size", ascending=False)


def _data_quality(raw: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    raw_rows = len(raw)
    unique_questions = len(canonical)
    duplicate_rate = 0 if raw_rows == 0 else round((raw_rows - unique_questions) / raw_rows * 100, 2)
    low_confidence = int((pd.to_numeric(raw.get("confidence_score", 0), errors="coerce").fillna(0) < 0.75).sum())
    official = int(raw["source_tier"].isin(["official_nta", "official_manual", "official_syllabus_ncert"]).sum())
    third_party = raw_rows - official
    rows = [
        {"metric": "total_parsed_rows", "value": raw_rows},
        {"metric": "unique_questions_after_dedupe", "value": unique_questions},
        {"metric": "duplicate_rate_percent", "value": duplicate_rate},
        {"metric": "official_source_rows", "value": official},
        {"metric": "third_party_source_rows", "value": third_party},
        {"metric": "low_confidence_classifications_lt_075", "value": low_confidence},
    ]
    for tier, count in raw["source_tier"].value_counts().items():
        rows.append({"metric": f"source_tier_{tier}", "value": int(count)})
    return pd.DataFrame(rows)


def _write_empty_outputs(df: pd.DataFrame) -> None:
    df.to_csv(PROCESSED_DIR / "questions_with_canonical_ids.csv", index=False)
    df.to_csv(PROCESSED_DIR / "questions_canonical.csv", index=False)
    pd.DataFrame(columns=["canonical_question_id", "duplicate_group_size"]).to_csv(PROCESSED_DIR / "duplicate_groups.csv", index=False)
    pd.DataFrame(
        [
            {"metric": "total_parsed_rows", "value": 0},
            {"metric": "unique_questions_after_dedupe", "value": 0},
            {"metric": "duplicate_rate_percent", "value": 0},
        ]
    ).to_csv(PROCESSED_DIR / "data_quality_summary.csv", index=False)

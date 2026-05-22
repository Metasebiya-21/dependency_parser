"""UAS, disagreements, and error taxonomy for parser comparison."""
from __future__ import annotations

from typing import Any

import pandas as pd

from dashboard.config import ERROR_TAXONOMY
from dashboard.types import SentenceBundle, SentenceParse, TokenParse


def _content_tokens(parse: SentenceParse) -> list[TokenParse]:
    return [t for t in parse.tokens if t.deprel != "punct" and t.upos != "PUNCT"]


def uas(gold: SentenceParse, pred: SentenceParse) -> tuple[float, int, int]:
    """Unlabeled Attachment Score for one sentence pair."""
    correct = total = 0
    for g, p in zip(_content_tokens(gold), _content_tokens(pred)):
        total += 1
        correct += int(g.head == p.head)
    return (correct / total if total else 0.0, correct, total)


def corpus_uas(
    bundles: list[SentenceBundle],
    parser: str = "spacy",
) -> tuple[float, int, int]:
    """Corpus-level UAS against gold."""
    correct = total = 0
    for b in bundles:
        if not b.has_gold():
            continue
        pred = b.spacy if parser == "spacy" else b.stanza
        if pred is None:
            continue
        _, c, t = uas(b.gold, pred)
        correct += c
        total += t
    score = correct / total if total else 0.0
    return score, correct, total


def token_disagreements(
    spacy: SentenceParse,
    stanza: SentenceParse,
) -> list[dict[str, Any]]:
    """Tokens where spaCy and Stanza differ on head or label."""
    rows = []
    for s, t in zip(_content_tokens(spacy), _content_tokens(stanza)):
        if s.head != t.head or s.deprel != t.deprel:
            rows.append(
                {
                    "index": s.index,
                    "token": s.form,
                    "spacy_head": s.head,
                    "spacy_deprel": s.deprel,
                    "stanza_head": t.head,
                    "stanza_deprel": t.deprel,
                    "head_disagree": s.head != t.head,
                    "label_disagree": s.deprel != t.deprel,
                }
            )
    return rows


def gold_errors(
    gold: SentenceParse,
    pred: SentenceParse,
) -> list[dict[str, Any]]:
    """Tokens where prediction differs from gold."""
    errors = []
    for g, p in zip(_content_tokens(gold), _content_tokens(pred)):
        if g.head != p.head or g.deprel != p.deprel:
            errors.append(
                {
                    "token": g.form,
                    "gold_head": g.head,
                    "gold_deprel": g.deprel,
                    "pred_head": p.head,
                    "pred_deprel": p.deprel,
                    "head_wrong": g.head != p.head,
                }
            )
    return errors


def categorize_error(disagreement_rows: list[dict[str, Any]]) -> str:
    rels: set[str] = set()
    for row in disagreement_rows:
        for key in ("gold_deprel", "spacy_deprel", "stanza_deprel", "pred_deprel"):
            if key in row and row[key]:
                rels.add(row[key])
    for label, deprels in ERROR_TAXONOMY:
        if rels & deprels:
            return label
    return "Other Attachment/Labeling Difference"


def build_uas_dataframe(bundles: list[SentenceBundle]) -> pd.DataFrame:
    rows = []
    for parser in ("spacy", "Stanza"):
        key = "spacy" if parser == "spacy" else "stanza"
        score, correct, total = corpus_uas(bundles, key)
        rows.append(
            {
                "Parser": parser,
                "UAS": round(score, 4),
                "Correct Heads": correct,
                "Total Tokens": total,
                "UAS %": round(score * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def build_per_sentence_uas(bundles: list[SentenceBundle]) -> pd.DataFrame:
    rows = []
    for b in bundles:
        if not b.has_gold():
            continue
        s_score, _, _ = uas(b.gold, b.spacy) if b.spacy else (0, 0, 0)
        t_score, _, _ = uas(b.gold, b.stanza) if b.stanza else (0, 0, 0)
        disagreements = (
            len(token_disagreements(b.spacy, b.stanza))
            if b.spacy and b.stanza
            else 0
        )
        rows.append(
            {
                "sent_id": b.sent_id,
                "text": b.text[:80] + ("…" if len(b.text) > 80 else ""),
                "spaCy UAS": round(s_score, 3),
                "Stanza UAS": round(t_score, 3),
                "Disagreements": disagreements,
            }
        )
    return pd.DataFrame(rows)


def build_error_table(bundles: list[SentenceBundle]) -> pd.DataFrame:
    """Error analysis table: parser disagreements with optional gold."""
    rows = []
    for b in bundles:
        if not b.spacy or not b.stanza:
            continue
        disagreements = token_disagreements(b.spacy, b.stanza)
        if not disagreements:
            continue

        enriched = []
        for d in disagreements:
            row = dict(d)
            if b.gold:
                gt = _content_tokens(b.gold)
                match = next((t for t in gt if t.form == d["token"]), None)
                if match:
                    row["gold_head"] = match.head
                    row["gold_deprel"] = match.deprel
            enriched.append(row)

        error_type = categorize_error(enriched)
        rows.append(
            {
                "Sentence": b.text,
                "spaCy Parse": b.spacy.compact(),
                "Stanza Parse": b.stanza.compact(),
                "Gold Standard": b.gold.compact() if b.gold else "—",
                "Error Type": error_type,
                "Disagreement Count": len(disagreements),
                "Disagreement Tokens": ", ".join(d["token"] for d in disagreements),
            }
        )
    return pd.DataFrame(rows)


def error_type_counts(error_df: pd.DataFrame) -> pd.DataFrame:
    if error_df.empty:
        return pd.DataFrame(columns=["Error Type", "Count"])
    counts = error_df["Error Type"].value_counts().reset_index()
    counts.columns = ["Error Type", "Count"]
    return counts


def pp_attachment_target(parse: SentenceParse, pp_token: str = "with") -> str | None:
    """
    Return the attachment site for a prepositional phrase headed by pp_token.

    Handles spaCy-style (prep -> governor) and UD-style (case -> noun -> governor).
    """
    for t in parse.tokens:
        if t.form.lower() != pp_token.lower():
            continue
        if t.deprel in {"case", "mark"} and t.head:
            noun = next((x for x in parse.tokens if x.index == t.head), None)
            if noun:
                return noun.head_form(parse.tokens)
        return t.head_form(parse.tokens)
    return None


def summary_metrics(bundles: list[SentenceBundle]) -> dict[str, Any]:
    gold_bundles = [b for b in bundles if b.has_gold()]
    total_disagree_sentences = sum(
        1
        for b in bundles
        if b.spacy and b.stanza and token_disagreements(b.spacy, b.stanza)
    )
    total_token_disagreements = sum(
        len(token_disagreements(b.spacy, b.stanza))
        for b in bundles
        if b.spacy and b.stanza
    )
    spacy_uas, _, _ = corpus_uas(gold_bundles, "spacy") if gold_bundles else (0, 0, 0)
    stanza_uas, _, _ = corpus_uas(gold_bundles, "stanza") if gold_bundles else (0, 0, 0)

    return {
        "sentences": len(bundles),
        "gold_sentences": len(gold_bundles),
        "disagreement_sentences": total_disagree_sentences,
        "token_disagreements": total_token_disagreements,
        "spacy_uas": spacy_uas,
        "stanza_uas": stanza_uas,
    }

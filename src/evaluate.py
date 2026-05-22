"""Evaluate UAS, find disagreements, and categorize error types."""
from __future__ import annotations

from pathlib import Path
from conllu import parse_incr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
GOLD = DATA_DIR / "manual_gold_10.conllu"  # use verified subset for final scoring
SPACY = OUT_DIR / "spacy_parses.conllu"
STANZA = OUT_DIR / "stanza_parses.conllu"

ERROR_RULES = [
    ("Prepositional Phrase Attachment", {"case", "obl", "nmod", "prep", "pobj"}),
    ("Coordination Scope", {"cc", "conj", "preconj", "cc:preconj"}),
    ("Relative Clause Attachment", {"acl:relcl", "relcl", "acl", "ref"}),
    ("Noun Compound Parsing", {"compound", "nn", "amod", "nmod"}),
]


def load(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return list(parse_incr(fh))


def words(sent):
    return [t for t in sent if isinstance(t.get("id"), int)]


def uas(gold, pred):
    correct = total = 0
    for gsent, psent in zip(gold, pred):
        for g, p in zip(words(gsent), words(psent)):
            if g.get("upos") == "PUNCT" or g.get("deprel") == "punct":
                continue
            total += 1
            correct += int(g.get("head") == p.get("head"))
    return correct / total if total else 0.0, correct, total


def categorize(token_rows):
    rels = {r["gold_deprel"] for r in token_rows} | {r["spacy_deprel"] for r in token_rows} | {r["stanza_deprel"] for r in token_rows}
    for label, deprels in ERROR_RULES:
        if rels & deprels:
            return label
    return "Other Attachment/Labeling Difference"


def compact_parse(sent):
    toks = words(sent)
    out = []
    for t in toks:
        head = t.get("head")
        head_word = "ROOT" if head == 0 else toks[head - 1]["form"] if isinstance(head, int) and 0 < head <= len(toks) else "?"
        out.append(f"{t['form']}→{head_word}/{t.get('deprel')}")
    return "; ".join(out)


def main() -> None:
    gold = load(GOLD)
    spacy = load(SPACY)[: len(gold)]
    stanza = load(STANZA)[: len(gold)]

    rows = []
    score_rows = []
    for name, pred in [("spaCy", spacy), ("Stanza", stanza)]:
        score, correct, total = uas(gold, pred)
        score_rows.append({"Parser": name, "UAS": round(score, 4), "CorrectHeads": correct, "TotalTokens": total})

    for gsent, ssent, tsent in zip(gold, spacy, stanza):
        disagreements = []
        for g, s, t in zip(words(gsent), words(ssent), words(tsent)):
            if g.get("deprel") == "punct":
                continue
            if s.get("head") != t.get("head") or s.get("deprel") != t.get("deprel"):
                disagreements.append({
                    "token": g["form"],
                    "gold_head": g.get("head"), "gold_deprel": g.get("deprel"),
                    "spacy_head": s.get("head"), "spacy_deprel": s.get("deprel"),
                    "stanza_head": t.get("head"), "stanza_deprel": t.get("deprel"),
                })
        if disagreements:
            rows.append({
                "Sentence": gsent.metadata.get("text", ""),
                "spaCy Parse": compact_parse(ssent),
                "Stanza Parse": compact_parse(tsent),
                "Gold Standard": compact_parse(gsent),
                "Error Type": categorize(disagreements),
                "Disagreement Tokens": ", ".join(d["token"] for d in disagreements),
            })

    pd.DataFrame(score_rows).to_csv(OUT_DIR / "uas_scores.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT_DIR / "error_analysis_table.csv", index=False)
    print(pd.DataFrame(score_rows).to_string(index=False))
    print(f"Wrote {OUT_DIR / 'error_analysis_table.csv'}")


if __name__ == "__main__":
    main()

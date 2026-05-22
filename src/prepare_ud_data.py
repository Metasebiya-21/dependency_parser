"""Download/select 50 sentences from UD English EWT.

The selection favors syntactically interesting sentences containing PP attachment,
coordination, relative clauses, and noun compounds, while still falling back to
regular UD examples if too few complex sentences are found.
"""
from __future__ import annotations

from pathlib import Path
import urllib.request
from conllu import parse_incr, TokenList

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)
RAW = DATA_DIR / "en_ewt-ud-dev.conllu"
SELECTED = DATA_DIR / "selected_50.conllu"

UD_DEV_RAW = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu"

TARGET_TYPES = {
    "pp_attachment": {"case", "obl", "nmod"},
    "coordination": {"cc", "conj"},
    "relative_clause": {"acl:relcl", "acl", "ref"},
    "noun_compound": {"compound", "amod", "nmod"},
}


def download_if_missing() -> None:
    if not RAW.exists():
        print(f"Downloading {UD_DEV_RAW}")
        urllib.request.urlretrieve(UD_DEV_RAW, RAW)


def sentence_text(sent: TokenList) -> str:
    return sent.metadata.get("text", " ".join(tok["form"] for tok in sent if isinstance(tok["id"], int)))


def complexity_score(sent: TokenList) -> int:
    rels = {tok.get("deprel") for tok in sent if isinstance(tok.get("id"), int)}
    length = sum(1 for tok in sent if isinstance(tok.get("id"), int))
    score = 0
    for labels in TARGET_TYPES.values():
        if rels & labels:
            score += 2
    if 12 <= length <= 35:
        score += 1
    if length > 35:
        score -= 1
    return score


def main() -> None:
    download_if_missing()
    with RAW.open("r", encoding="utf-8") as fh:
        sentences = list(parse_incr(fh))

    ranked = sorted(sentences, key=complexity_score, reverse=True)
    selected = ranked[:50]

    with SELECTED.open("w", encoding="utf-8") as out:
        for sent in selected:
            out.write(sent.serialize())
            out.write("\n")

    print(f"Wrote {len(selected)} sentences to {SELECTED}")
    print("First 10 selected sentences:")
    for i, sent in enumerate(selected[:10], 1):
        print(f"{i:02d}. {sentence_text(sent)}")


if __name__ == "__main__":
    main()

"""Run spaCy and Stanza dependency parsers and save CoNLL-U-like outputs."""
from __future__ import annotations

from pathlib import Path
from conllu import parse_incr, TokenList
import spacy
import stanza

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True)
INPUT = DATA_DIR / "selected_50.conllu"


def get_text(sent: TokenList) -> str:
    return sent.metadata.get("text", " ".join(tok["form"] for tok in sent if isinstance(tok["id"], int)))


def write_simple_conllu(records, path: Path) -> None:
    with path.open("w", encoding="utf-8") as out:
        for sent_id, text, rows in records:
            out.write(f"# sent_id = {sent_id}\n# text = {text}\n")
            for i, row in enumerate(rows, 1):
                # ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
                out.write("\t".join([
                    str(i), row["form"], "_", row.get("upos", "_"), "_", "_",
                    str(row["head"]), row["deprel"], "_", "_"
                ]) + "\n")
            out.write("\n")


def spacy_parse(nlp, text: str):
    doc = nlp(text)
    rows = []
    for tok in doc:
        head = 0 if tok.head.i == tok.i else tok.head.i + 1
        rows.append({"form": tok.text, "upos": tok.pos_, "head": head, "deprel": tok.dep_})
    return rows


def stanza_parse(nlp, text: str):
    doc = nlp(text)
    rows = []
    for sent in doc.sentences:
        for word in sent.words:
            rows.append({"form": word.text, "upos": word.upos, "head": word.head, "deprel": word.deprel})
    return rows


def main() -> None:
    with INPUT.open("r", encoding="utf-8") as fh:
        gold_sents = list(parse_incr(fh))

    spacy_nlp = spacy.load("en_core_web_sm")
    stanza_nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", tokenize_no_ssplit=True)

    spacy_records, stanza_records = [], []
    for idx, sent in enumerate(gold_sents, 1):
        text = get_text(sent)
        sent_id = sent.metadata.get("sent_id", f"s{idx}")
        spacy_records.append((sent_id, text, spacy_parse(spacy_nlp, text)))
        stanza_records.append((sent_id, text, stanza_parse(stanza_nlp, text)))

    write_simple_conllu(spacy_records, OUT_DIR / "spacy_parses.conllu")
    write_simple_conllu(stanza_records, OUT_DIR / "stanza_parses.conllu")
    print("Wrote parser outputs to outputs/.")


if __name__ == "__main__":
    main()

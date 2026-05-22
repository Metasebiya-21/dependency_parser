"""Create a manual-verification file containing 10 complex sentences.

Workflow:
1. Run prepare_ud_data.py first.
2. Run this script to copy the top 10 complex UD parses into manual_gold_10.conllu.
3. Open manual_gold_10.conllu and manually verify/correct HEAD and DEPREL columns.
"""
from pathlib import Path
from conllu import parse_incr

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SELECTED = DATA_DIR / "selected_50.conllu"
MANUAL = DATA_DIR / "manual_gold_10.conllu"


def main() -> None:
    with SELECTED.open("r", encoding="utf-8") as fh:
        sentences = list(parse_incr(fh))[:10]

    with MANUAL.open("w", encoding="utf-8") as out:
        out.write("# MANUAL VERIFICATION FILE\n")
        out.write("# Verify/correct columns 7 HEAD and 8 DEPREL for each token.\n")
        out.write("# Keep valid CoNLL-U format.\n\n")
        for sent in sentences:
            out.write(sent.serialize())
            out.write("\n")

    print(f"Wrote {MANUAL}. Manually inspect/correct HEAD and DEPREL columns.")


if __name__ == "__main__":
    main()

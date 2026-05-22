"""Paths, constants, and taxonomy for the dependency parser dashboard."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "report"
# Stanza models (downloaded at runtime; not committed — see .gitignore)
STANZA_DIR = PROJECT_ROOT / "stanza_resources"

UD_DEV_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu"
)
RAW_CONLLU = DATA_DIR / "en_ewt-ud-dev.conllu"
SELECTED_50 = DATA_DIR / "selected_50.conllu"
MANUAL_GOLD_10 = DATA_DIR / "manual_gold_10.conllu"
AMBIGUOUS_GOLD = DATA_DIR / "ambiguous_sentence_gold.conllu"

NUM_SENTENCES = 50
NUM_GOLD_COMPLEX = 10
RANDOM_SEED = 42

TELESCOPE_SENTENCE = "The man saw the girl with the telescope."

ERROR_TAXONOMY: list[tuple[str, set[str]]] = [
    ("Prepositional Phrase Attachment", {"case", "obl", "nmod", "prep", "pobj"}),
    ("Coordination Scope", {"cc", "conj", "preconj", "cc:preconj"}),
    ("Relative Clause Attachment", {"acl:relcl", "relcl", "acl", "ref"}),
    ("Noun Compound Parsing", {"compound", "nn", "amod", "nmod"}),
]

COMPLEXITY_DEPS = {
    "pp_attachment": {"case", "obl", "nmod", "prep", "pobj"},
    "coordination": {"cc", "conj"},
    "relative_clause": {"acl:relcl", "acl", "ref"},
    "noun_compound": {"compound", "amod", "nmod"},
}

PARSER_COLORS = {
    "spacy": "#2563eb",
    "stanza": "#7c3aed",
    "gold": "#059669",
    "disagree": "#dc2626",
    "agree": "#64748b",
}

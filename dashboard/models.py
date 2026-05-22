"""Ensure spaCy and Stanza English models are available."""
from __future__ import annotations

import subprocess
import sys


def ensure_spacy_model(model_name: str = "en_core_web_sm"):
    """Load spaCy model; download if missing."""
    import spacy

    try:
        return spacy.load(model_name)
    except OSError:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True,
            capture_output=True,
        )
        return spacy.load(model_name)


def ensure_stanza_pipeline(lang: str = "en", verbose: bool = False):
    """Load Stanza pipeline; download resources if missing."""
    import stanza

    processors = "tokenize,pos,lemma,depparse"
    try:
        return stanza.Pipeline(
            lang,
            processors=processors,
            tokenize_no_ssplit=True,
            verbose=verbose,
        )
    except Exception:
        stanza.download(lang, verbose=verbose)
        return stanza.Pipeline(
            lang,
            processors=processors,
            tokenize_no_ssplit=True,
            verbose=verbose,
        )

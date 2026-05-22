"""Ensure spaCy and Stanza English models are available (local + Streamlit Cloud)."""
from __future__ import annotations

import os
import sys

from dashboard.config import PROJECT_ROOT, STANZA_DIR

SPACY_MODEL = "en_core_web_sm"
SPACY_WHEEL = (
    "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)


def _is_streamlit_cloud() -> bool:
    return bool(os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("STREAMLIT_CLOUD"))


def ensure_spacy_model(model_name: str = SPACY_MODEL):
    """
    Load spaCy model.

    On Streamlit Community Cloud, models must be installed via requirements.txt
    (pip wheel), not `python -m spacy download`.
    """
    import spacy

    try:
        return spacy.load(model_name)
    except OSError as exc:
        hint = (
            f"Install the English model in requirements.txt:\n\n{SPACY_WHEEL}\n\n"
            "Streamlit Cloud cannot run `python -m spacy download` at runtime."
        )
        if _is_streamlit_cloud():
            raise RuntimeError(hint) from exc
        # Local fallback only (not used on Cloud)
        import subprocess

        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=False,
            capture_output=True,
        )
        try:
            return spacy.load(model_name)
        except OSError:
            raise RuntimeError(hint) from exc


def _stanza_ready(lang: str) -> bool:
    resources = STANZA_DIR / "resources.json"
    if not resources.exists():
        return False
    lang_dir = STANZA_DIR / lang
    return lang_dir.is_dir() and any(lang_dir.iterdir())


def ensure_stanza_pipeline(lang: str = "en", verbose: bool = False):
    """Load Stanza pipeline; download English resources into project stanza_resources/."""
    import stanza

    STANZA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["STANZA_RESOURCES_DIR"] = str(STANZA_DIR)

    processors = "tokenize,pos,lemma,depparse"
    model_dir = str(STANZA_DIR)

    if not _stanza_ready(lang):
        stanza.download(lang, dir=model_dir, logging_level="WARNING")

    return stanza.Pipeline(
        lang,
        processors=processors,
        dir=model_dir,
        tokenize_no_ssplit=True,
        verbose=verbose,
    )

"""Load and prepare UD English EWT sentences for the dashboard."""
from __future__ import annotations

import random
import urllib.request
from pathlib import Path

from conllu import parse_incr, TokenList

from dashboard.config import (
    AMBIGUOUS_GOLD,
    COMPLEXITY_DEPS,
    DATA_DIR,
    MANUAL_GOLD_10,
    NUM_GOLD_COMPLEX,
    NUM_SENTENCES,
    RANDOM_SEED,
    RAW_CONLLU,
    SELECTED_50,
    UD_DEV_URL,
)
from dashboard.types import SentenceParse, TokenParse


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_ud_dev(force: bool = False) -> Path:
    """Download UD English EWT dev CoNLL-U if missing."""
    ensure_data_dir()
    if force or not RAW_CONLLU.exists():
        urllib.request.urlretrieve(UD_DEV_URL, RAW_CONLLU)
    return RAW_CONLLU


def sentence_text(sent: TokenList) -> str:
    return sent.metadata.get(
        "text",
        " ".join(tok["form"] for tok in sent if isinstance(tok.get("id"), int)),
    )


def complexity_score(sent: TokenList) -> int:
    rels = {tok.get("deprel") for tok in sent if isinstance(tok.get("id"), int)}
    length = sum(1 for tok in sent if isinstance(tok.get("id"), int))
    score = 0
    for labels in COMPLEXITY_DEPS.values():
        if rels & labels:
            score += 2
    if 12 <= length <= 35:
        score += 1
    return score


def conllu_to_parse(sent: TokenList, sent_id: str | None = None) -> SentenceParse:
    """Convert a conllu TokenList to SentenceParse."""
    tokens: list[TokenParse] = []
    id_map: dict[int, int] = {}
    raw_tokens = [t for t in sent if isinstance(t.get("id"), int)]

    for i, t in enumerate(raw_tokens, 1):
        id_map[t["id"]] = i
        tokens.append(
            TokenParse(
                index=i,
                form=t["form"],
                lemma=t.get("lemma") or "_",
                upos=t.get("upos") or "_",
                head=0,
                deprel=t.get("deprel") or "_",
            )
        )

    for t_raw, t_parse in zip(raw_tokens, tokens):
        head = t_raw.get("head")
        if head == 0:
            t_parse.head = 0
        elif isinstance(head, int) and head in id_map:
            t_parse.head = id_map[head]
        elif isinstance(head, int):
            t_parse.head = min(head, len(tokens))

    sid = sent_id or str(sent.metadata.get("sent_id", "unknown"))
    return SentenceParse(
        sent_id=sid,
        text=sentence_text(sent),
        tokens=tokens,
        metadata=dict(sent.metadata),
    )


def load_conllu_file(path: Path) -> list[SentenceParse]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [
            conllu_to_parse(sent)
            for sent in parse_incr(fh)
        ]


def select_sentences_random(
    sentences: list[TokenList],
    n: int = NUM_SENTENCES,
    seed: int = RANDOM_SEED,
) -> list[TokenList]:
    """Randomly sample n sentences with fixed seed."""
    rng = random.Random(seed)
    if len(sentences) <= n:
        return sentences
    return rng.sample(sentences, n)


def select_sentences_mixed(
    sentences: list[TokenList],
    n: int = NUM_SENTENCES,
    seed: int = RANDOM_SEED,
) -> list[TokenList]:
    """Prefer syntactically complex sentences, then fill with random samples."""
    ranked = sorted(sentences, key=complexity_score, reverse=True)
    complex_pool = ranked[: max(n * 2, n)]
    rng = random.Random(seed)
    chosen = rng.sample(complex_pool, min(n, len(complex_pool)))
    if len(chosen) < n:
        remaining = [s for s in sentences if s not in chosen]
        chosen.extend(rng.sample(remaining, n - len(chosen)))
    rng.shuffle(chosen)
    return chosen[:n]


def write_selected_conllu(sentences: list[TokenList], path: Path = SELECTED_50) -> Path:
    ensure_data_dir()
    with path.open("w", encoding="utf-8") as out:
        for sent in sentences:
            out.write(sent.serialize())
            out.write("\n")
    return path


def prepare_dataset(
    strategy: str = "mixed",
    force_download: bool = False,
    force_regenerate: bool = False,
) -> list[SentenceParse]:
    """
    Ensure UD data exists and return 50 selected sentences.

    strategy: 'random' | 'mixed' | 'file' (load existing selected_50.conllu only)
    """
    if strategy == "file" and SELECTED_50.exists() and not force_regenerate:
        return load_conllu_file(SELECTED_50)

    download_ud_dev(force=force_download)
    with RAW_CONLLU.open("r", encoding="utf-8") as fh:
        all_sents = list(parse_incr(fh))

    if strategy == "random":
        selected = select_sentences_random(all_sents)
    else:
        selected = select_sentences_mixed(all_sents)

    write_selected_conllu(selected)
    return [conllu_to_parse(s) for s in selected]


def load_gold_complex(limit: int = NUM_GOLD_COMPLEX) -> list[SentenceParse]:
    """Load manually verified gold parses (top N complex sentences)."""
    if MANUAL_GOLD_10.exists():
        gold = load_conllu_file(MANUAL_GOLD_10)
        if gold:
            return gold[:limit]

    if SELECTED_50.exists():
        return load_conllu_file(SELECTED_50)[:limit]

    return []


def load_ambiguous_gold() -> SentenceParse | None:
    gold = load_conllu_file(AMBIGUOUS_GOLD)
    return gold[0] if gold else None


def ensure_manual_gold_template() -> Path:
    """Copy top 10 complex sentences into manual verification file."""
    ensure_data_dir()
    if not SELECTED_50.exists():
        prepare_dataset(strategy="mixed")

    with SELECTED_50.open("r", encoding="utf-8") as fh:
        sentences = list(parse_incr(fh))[:NUM_GOLD_COMPLEX]

    with MANUAL_GOLD_10.open("w", encoding="utf-8") as out:
        out.write("# MANUAL VERIFICATION FILE\n")
        out.write("# Verify/correct columns 7 HEAD and 8 DEPREL for each token.\n\n")
        for sent in sentences:
            out.write(sent.serialize())
            out.write("\n")
    return MANUAL_GOLD_10


def dataset_stats(parses: list[SentenceParse]) -> dict[str, float | int]:
    lengths = [len(p.tokens) for p in parses]
    return {
        "count": len(parses),
        "avg_tokens": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
    }

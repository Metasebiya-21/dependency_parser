"""Shared data structures for parses and evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenParse:
    """Single token with dependency annotation (1-based indexing)."""

    index: int
    form: str
    lemma: str = "_"
    upos: str = "_"
    head: int = 0
    deprel: str = "_"

    def head_form(self, tokens: list["TokenParse"]) -> str:
        if self.head == 0:
            return "ROOT"
        for t in tokens:
            if t.index == self.head:
                return t.form
        return "?"


@dataclass
class SentenceParse:
    """Full dependency parse for one sentence."""

    sent_id: str
    text: str
    tokens: list[TokenParse] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def forms(self) -> list[str]:
        return [t.form for t in self.tokens]

    @property
    def heads(self) -> list[int]:
        return [t.head for t in self.tokens]

    @property
    def deprels(self) -> list[str]:
        return [t.deprel for t in self.tokens]

    @property
    def upos_tags(self) -> list[str]:
        return [t.upos for t in self.tokens]

    def compact(self) -> str:
        parts = []
        for t in self.tokens:
            if t.deprel == "punct":
                continue
            parts.append(f"{t.form}→{t.head_form(self.tokens)}/{t.deprel}")
        return "; ".join(parts)

    def to_dict_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "id": t.index,
                "form": t.form,
                "lemma": t.lemma,
                "upos": t.upos,
                "head": t.head,
                "deprel": t.deprel,
            }
            for t in self.tokens
        ]


@dataclass
class SentenceBundle:
    """Gold + both parser outputs for one sentence."""

    sent_id: str
    text: str
    gold: SentenceParse | None = None
    spacy: SentenceParse | None = None
    stanza: SentenceParse | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_gold(self) -> bool:
        return self.gold is not None and len(self.gold.tokens) > 0

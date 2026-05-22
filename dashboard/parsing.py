"""spaCy (transition-based) and Stanza (graph-based) parsing pipelines."""
from __future__ import annotations

from dashboard.types import SentenceParse, SentenceBundle, TokenParse


def spacy_parse_text(nlp, text: str, sent_id: str = "custom") -> SentenceParse:
    """Run transition-based spaCy dependency parser."""
    doc = nlp(text)
    tokens: list[TokenParse] = []
    for i, tok in enumerate(doc, 1):
        head = 0 if tok.head.i == tok.i else tok.head.i + 1
        tokens.append(
            TokenParse(
                index=i,
                form=tok.text,
                lemma=tok.lemma_,
                upos=tok.pos_,
                head=head,
                deprel=tok.dep_,
            )
        )
    return SentenceParse(sent_id=sent_id, text=text, tokens=tokens)


def stanza_parse_text(nlp, text: str, sent_id: str = "custom") -> SentenceParse:
    """Run graph-based Stanza dependency parser."""
    doc = nlp(text)
    tokens: list[TokenParse] = []
    idx = 1
    for sent in doc.sentences:
        for word in sent.words:
            tokens.append(
                TokenParse(
                    index=idx,
                    form=word.text,
                    lemma=word.lemma or "_",
                    upos=word.upos or "_",
                    head=word.head,
                    deprel=word.deprel or "_",
                )
            )
            idx += 1
    return SentenceParse(sent_id=sent_id, text=text, tokens=tokens)


def parse_corpus(
    sentences: list[SentenceParse],
    spacy_nlp,
    stanza_nlp,
    progress_callback=None,
) -> list[SentenceBundle]:
    """Parse all sentences with both parsers."""
    bundles: list[SentenceBundle] = []
    total = len(sentences)
    for i, sent in enumerate(sentences):
        text = sent.text
        sid = sent.sent_id
        bundle = SentenceBundle(
            sent_id=sid,
            text=text,
            gold=sent,
            spacy=spacy_parse_text(spacy_nlp, text, sid),
            stanza=stanza_parse_text(stanza_nlp, text, sid),
            metadata=sent.metadata,
        )
        bundles.append(bundle)
        if progress_callback:
            progress_callback((i + 1) / total)
    return bundles


def parse_single(
    text: str,
    spacy_nlp,
    stanza_nlp,
    gold: SentenceParse | None = None,
    sent_id: str = "adhoc",
) -> SentenceBundle:
    return SentenceBundle(
        sent_id=sent_id,
        text=text,
        gold=gold,
        spacy=spacy_parse_text(spacy_nlp, text, sent_id),
        stanza=stanza_parse_text(stanza_nlp, text, sent_id),
    )

"""Dependency tree rendering and Plotly charts."""
from __future__ import annotations

import html
import math
from typing import Iterable

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from dashboard.config import PARSER_COLORS
from dashboard.types import SentenceParse


def _disagreement_indices(
    parse_a: SentenceParse,
    parse_b: SentenceParse,
) -> set[int]:
    indices: set[int] = set()
    from dashboard.evaluation import token_disagreements

    for row in token_disagreements(parse_a, parse_b):
        indices.add(row["index"])
    return indices


def _resolve_head_index(head: int, tokens: list, visible: set[int]) -> int | None:
    """Map head index to a visible token (handles punctuation-only governors)."""
    if head == 0:
        return None
    if head in visible:
        return head
    by_idx = {t.index: t for t in tokens}
    cur = head
    seen: set[int] = set()
    while cur and cur not in visible and cur not in seen:
        seen.add(cur)
        parent = by_idx.get(cur)
        if not parent:
            return None
        cur = parent.head
    return cur if cur in visible else None


def render_dependency_svg(
    parse: SentenceParse,
    title: str = "",
    highlight_indices: Iterable[int] | None = None,
    color: str = PARSER_COLORS["spacy"],
    width: int = 680,
    height: int = 340,
) -> str:
    """Render a dependency tree as SVG with optional token highlighting."""
    all_tokens = parse.tokens
    tokens = [t for t in all_tokens if t.deprel != "punct" and t.upos != "PUNCT"]
    if not tokens:
        tokens = list(all_tokens)
    if not tokens:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="80"><text x="20" y="40">No tokens</text></svg>'

    highlight = set(highlight_indices or [])
    visible = {t.index for t in tokens}
    n = len(tokens)
    margin_x, margin_top, margin_bottom = 40, 50, 70
    usable_w = width - 2 * margin_x
    spacing = usable_w / max(n - 1, 1)
    positions = {
        t.index: (margin_x + i * spacing, height - margin_bottom)
        for i, t in enumerate(tokens)
    }

    arcs: list[str] = []
    labels: list[str] = []
    for t in tokens:
        gov = _resolve_head_index(t.head, all_tokens, visible)
        if gov is None:
            continue
        if t.index not in positions or gov not in positions:
            continue
        x1, y1 = positions[t.index]
        x2, y2 = positions[gov]
        mid_x = (x1 + x2) / 2
        order = {tok.index: i for i, tok in enumerate(tokens)}
        dist = abs(order[t.index] - order[gov])
        lift = 50 + dist * 22
        path = f"M {x1},{y1 - 18} Q {mid_x},{y1 - lift} {x2},{y2 - 18}"
        stroke = PARSER_COLORS["disagree"] if t.index in highlight else color
        arcs.append(
            f'<path d="{path}" fill="none" stroke="{stroke}" stroke-width="2.5" '
            f'stroke-opacity="0.9"/>'
        )
        labels.append(
            f'<text x="{mid_x}" y="{y1 - lift - 6}" text-anchor="middle" font-size="11" '
            f'fill="#475569" font-family="system-ui,sans-serif">'
            f"{html.escape(t.deprel)}</text>"
        )

    token_nodes = []
    for t in tokens:
        x, y = positions[t.index]
        fill = "#fee2e2" if t.index in highlight else "#f8fafc"
        stroke = PARSER_COLORS["disagree"] if t.index in highlight else "#334155"
        token_nodes.append(
            f'<rect x="{x - 32}" y="{y - 20}" width="64" height="36" rx="8" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )
        token_nodes.append(
            f'<text x="{x}" y="{y + 2}" text-anchor="middle" font-size="13" '
            f'font-weight="600" fill="#0f172a" font-family="system-ui,sans-serif">'
            f"{html.escape(t.form)}</text>"
        )
        token_nodes.append(
            f'<text x="{x}" y="{y + 16}" text-anchor="middle" font-size="10" '
            f'fill="#64748b" font-family="system-ui,sans-serif">'
            f"{html.escape(t.upos)}</text>"
        )

    title_el = ""
    if title:
        title_el = (
            f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="15" '
            f'font-weight="700" fill="#0f172a" font-family="system-ui,sans-serif">'
            f"{html.escape(title)}</text>"
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="{height}" '
        f'viewBox="0 0 {width} {height}" style="max-width:{width}px;background:#fff;'
        f'border:1px solid #e2e8f0;border-radius:10px;display:block;margin:0 auto;">'
        f"{title_el}{''.join(arcs)}{''.join(labels)}{''.join(token_nodes)}</svg>"
    )


def wrap_html_fragment(body: str, height: int = 400) -> str:
    """Wrap HTML for st.components.v1.html (Streamlit strips SVG from st.markdown)."""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
  body {{ margin: 0; padding: 8px; background: #fff; font-family: system-ui, sans-serif; }}
  .tree-wrap {{ width: 100%; overflow-x: auto; }}
</style></head>
<body><div class="tree-wrap">{body}</div></body></html>"""


def render_comparison_html(
    spacy: SentenceParse,
    stanza: SentenceParse,
    gold: SentenceParse | None = None,
) -> str:
    """Full HTML document with side-by-side SVG trees (for download / components)."""
    disagree = _disagreement_indices(spacy, stanza)
    note = ""
    if disagree:
        note = (
            '<p style="color:#dc2626;font-size:0.9rem;margin:0 0 12px 0;">'
            "Red highlight = tokens where spaCy and Stanza disagree on attachment</p>"
        )
    panels = [
        ("spaCy (transition-based)", render_dependency_svg(
            spacy, "spaCy (transition-based)", disagree, color=PARSER_COLORS["spacy"])),
        ("Stanza (graph-based)", render_dependency_svg(
            stanza, "Stanza (graph-based)", disagree, color=PARSER_COLORS["stanza"])),
    ]
    if gold:
        panels.append(
            ("Gold (UD)", render_dependency_svg(
                gold, "Gold (UD)", color=PARSER_COLORS["gold"])),
        )
    grid = "".join(
        f'<div style="flex:1;min-width:280px;text-align:center;">{svg}</div>'
        for _, svg in panels
    )
    body = (
        f"{note}<div style=\"display:flex;flex-wrap:wrap;gap:16px;"
        f'justify-content:center;align-items:flex-start;">{grid}</div>'
    )
    return wrap_html_fragment(body, height=380)


def tree_panel_height(parse: SentenceParse) -> int:
    n = sum(1 for t in parse.tokens if t.deprel != "punct")
    return max(360, min(520, 300 + n * 12))


def spacy_displacy_html(nlp, text: str, options: dict | None = None) -> str:
    """Render spaCy displaCy dependency visualization as HTML."""
    from spacy import displacy

    doc = nlp(text)
    opts = {"compact": True, "bg": "#0f172a", "color": "#38bdf8", "font": "Source Sans Pro"}
    if options:
        opts.update(options)
    return displacy.render(doc, style="dep", page=True, options=opts)


def uas_comparison_chart(uas_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        uas_df,
        x="Parser",
        y="UAS %",
        color="Parser",
        color_discrete_map={
            "spacy": PARSER_COLORS["spacy"],
            "Stanza": PARSER_COLORS["stanza"],
        },
        text="UAS %",
        title="Unlabeled Attachment Score (UAS)",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        template="plotly_white",
        yaxis_range=[0, 100],
        showlegend=False,
        height=380,
        margin=dict(t=60, b=40),
    )
    return fig


def per_sentence_uas_chart(per_sent_df: pd.DataFrame) -> go.Figure:
    if per_sent_df.empty:
        return go.Figure()
    melted = per_sent_df.melt(
        id_vars=["sent_id"],
        value_vars=["spaCy UAS", "Stanza UAS"],
        var_name="Parser",
        value_name="UAS",
    )
    fig = px.scatter(
        melted,
        x="sent_id",
        y="UAS",
        color="Parser",
        title="Per-sentence UAS (gold-evaluated subset)",
        color_discrete_sequence=[PARSER_COLORS["spacy"], PARSER_COLORS["stanza"]],
    )
    fig.update_layout(template="plotly_white", height=400, xaxis_tickangle=-45)
    return fig


def error_distribution_chart(error_counts: pd.DataFrame) -> go.Figure:
    if error_counts.empty:
        return go.Figure()
    fig = px.pie(
        error_counts,
        names="Error Type",
        values="Count",
        title="Parser disagreement by error taxonomy",
        hole=0.35,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig


def disagreement_frequency_chart(error_df: pd.DataFrame) -> go.Figure:
    if error_df.empty or "Disagreement Count" not in error_df.columns:
        return go.Figure()
    top = error_df.nlargest(15, "Disagreement Count")
    fig = px.bar(
        top,
        x="Disagreement Count",
        y="Sentence",
        orientation="h",
        title="Top sentences by parser disagreement count",
        color_discrete_sequence=[PARSER_COLORS["disagree"]],
    )
    fig.update_layout(template="plotly_white", height=450, yaxis={"categoryorder": "total ascending"})
    return fig


def token_table_html(parse: SentenceParse, highlight: set[int] | None = None) -> str:
    highlight = highlight or set()
    rows = []
    for t in parse.tokens:
        bg = "#fee2e2" if t.index in highlight else "#ffffff"
        rows.append(
            f"<tr style='background:{bg}'>"
            f"<td>{t.index}</td><td>{html.escape(t.form)}</td>"
            f"<td>{html.escape(t.upos)}</td>"
            f"<td>{t.head}</td><td>{html.escape(t.head_form(parse.tokens))}</td>"
            f"<td>{html.escape(t.deprel)}</td></tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:0.85rem;'>"
        "<thead><tr style='background:#f1f5f9'>"
        "<th>ID</th><th>Token</th><th>POS</th><th>Head</th><th>Head Word</th><th>Dep</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def attachment_explanation(pp_head: str | None) -> dict[str, str]:
    """Linguistic interpretation for PP attachment on telescope sentence."""
    if pp_head is None:
        return {
            "reading": "Unknown",
            "description": "Could not detect PP attachment.",
        }
    head = pp_head.lower()
    if head == "saw":
        return {
            "reading": "Instrumental (VP attachment)",
            "description": (
                '"with the telescope" modifies the verb **saw**: the man used a telescope '
                "as an instrument to see the girl."
            ),
        }
    if head == "girl":
        return {
            "reading": "Nominal modification (NP attachment)",
            "description": (
                '"with the telescope" modifies **girl**: the girl is the one associated '
                "with the telescope (possession or accompaniment)."
            ),
        }
    return {
        "reading": f"Attached to «{pp_head}»",
        "description": f'The preposition "with" attaches to **{pp_head}**.',
    }

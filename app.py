"""
Dependency Parser Comparative Analysis — Streamlit Dashboard

Compare spaCy (transition-based) vs Stanza (graph-based) on UD English EWT.

Run: streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.config import (
    MANUAL_GOLD_10,
    PROJECT_ROOT,
    SELECTED_50,
    TELESCOPE_SENTENCE,
)
from dashboard.data_loading import (
    dataset_stats,
    ensure_manual_gold_template,
    load_ambiguous_gold,
    load_gold_complex,
    prepare_dataset,
)
from dashboard.evaluation import (
    build_error_table,
    build_per_sentence_uas,
    build_uas_dataframe,
    error_type_counts,
    pp_attachment_target,
    summary_metrics,
    token_disagreements,
    uas,
)
from dashboard.models import ensure_spacy_model, ensure_stanza_pipeline
from dashboard.parsing import parse_corpus, parse_single
from dashboard.config import PARSER_COLORS
from dashboard.visualization import (
    attachment_explanation,
    disagreement_frequency_chart,
    error_distribution_chart,
    per_sentence_uas_chart,
    render_comparison_html,
    render_dependency_svg,
    spacy_displacy_html,
    token_table_html,
    tree_panel_height,
    uas_comparison_chart,
    wrap_html_fragment,
    _disagreement_indices,
)

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dependency Parser Lab",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Source+Serif+4:wght@600&display=swap');
    html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
    .main-header {
        font-family: 'Source Serif 4', serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.15rem;
    }
    .sub-header { color: #475569; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #cbd5e1;
    }
    .theory-box {
        background: #f0f9ff;
        border-left: 4px solid #2563eb;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .ambig-box {
        background: #fef3c7;
        border-left: 4px solid #d97706;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
    }
    div[data-testid="stSidebar"] { background: #0f172a; }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    div[data-testid="stSidebar"] .stSelectbox label { color: #94a3b8 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading spaCy (transition-based parser)…")
def load_spacy():
    return ensure_spacy_model()


@st.cache_resource(show_spinner="Loading Stanza (graph-based parser)…")
def load_stanza():
    return ensure_stanza_pipeline(verbose=False)


@st.cache_data(show_spinner="Preparing UD English EWT dataset…")
def load_dataset(strategy: str):
    return prepare_dataset(strategy=strategy)


@st.cache_data(show_spinner="Running dual-parser pipeline on corpus…")
def run_full_parse(_version: int, strategy: str):
    sentences = load_dataset(strategy)
    gold_complex = load_gold_complex()
    spacy_nlp = load_spacy()
    stanza_nlp = load_stanza()
    bundles = parse_corpus(sentences, spacy_nlp, stanza_nlp)
    for i, b in enumerate(bundles):
        match = next((g for g in gold_complex if g.sent_id == b.sent_id), None)
        if match:
            b.gold = match
        elif i < len(gold_complex):
            b.gold = gold_complex[i]
    return bundles


def init_session_state():
    defaults = {
        "parse_version": 0,
        "dataset_strategy": "mixed",
        "selected_sent_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_bundles():
    return run_full_parse(st.session_state.parse_version, st.session_state.dataset_strategy)


def filter_bundles(bundles, query: str):
    if not query:
        return bundles
    q = query.lower()
    return [b for b in bundles if q in b.text.lower() or q in b.sent_id.lower()]


def sidebar():
    st.sidebar.markdown("## 🌳 Parser Lab")
    st.sidebar.markdown("**Transition vs graph-based parsing**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Dataset",
            "Parsing & Trees",
            "Evaluation",
            "Error Analysis",
            "Ambiguity Case Study",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.session_state.dataset_strategy = st.sidebar.selectbox(
        "Sentence selection",
        ["mixed", "random", "file"],
        format_func=lambda x: {
            "mixed": "Mixed (complexity-biased + shuffle)",
            "random": "Random 50 (seed=42)",
            "file": "Load existing selected_50.conllu",
        }[x],
    )

    if st.sidebar.button("Reload dataset & re-parse", type="primary"):
        load_dataset.clear()
        run_full_parse.clear()
        st.session_state.parse_version += 1
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("UD English EWT · spaCy · Stanza")
    st.sidebar.caption(f"Project: `{PROJECT_ROOT.name}`")
    return page


def header():
    st.markdown('<p class="main-header">Dependency Parser Comparative Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">spaCy (transition-based) vs Stanza (graph-based) · '
        "syntactic ambiguity · attachment decisions · UAS evaluation</p>",
        unsafe_allow_html=True,
    )


def overview_page(bundles):
    metrics = summary_metrics(bundles)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sentences", metrics["sentences"])
    c2.metric("Gold-eval subset", metrics["gold_sentences"])
    c3.metric("Disagreement sents", metrics["disagreement_sentences"])
    c4.metric("spaCy UAS", f"{metrics['spacy_uas']:.1%}")
    c5.metric("Stanza UAS", f"{metrics['stanza_uas']:.1%}")

    st.markdown(
        """
        <div class="theory-box">
        <strong>Transition-based (spaCy)</strong> builds trees incrementally via shift/reduce/arc
        actions — fast but locally committed.<br><br>
        <strong>Graph-based (Stanza)</strong> scores all head-dependent arcs and decodes a global
        tree — better global optimization, still statistical.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uas_df = build_uas_dataframe(bundles)
    err_df = build_error_table(bundles)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(uas_comparison_chart(uas_df), width="stretch")
    with col_b:
        st.plotly_chart(error_distribution_chart(error_type_counts(err_df)), width="stretch")

    st.subheader("Quick start")
    st.markdown(
        "1. **Dataset** — inspect 50 UD EWT sentences  \n"
        "2. **Parsing & Trees** — side-by-side dependency visualizations  \n"
        "3. **Evaluation** — UAS and per-sentence accuracy  \n"
        "4. **Error Analysis** — taxonomy-filtered disagreement table  \n"
        "5. **Ambiguity Case Study** — classic PP attachment example"
    )


def dataset_page(bundles):
    st.subheader("Universal Dependencies English EWT")
    stats = dataset_stats([b.gold or b.spacy for b in bundles if b.spacy])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected sentences", stats["count"])
    c2.metric("Avg tokens", stats["avg_tokens"])
    c3.metric("Min tokens", stats["min_tokens"])
    c4.metric("Max tokens", stats["max_tokens"])

    if st.button("Generate manual gold template (10 complex sents)"):
        path = ensure_manual_gold_template()
        st.success(f"Wrote `{path}`. Edit HEAD and DEPREL columns, then reload.")

    st.info(
        f"Gold parses: `{MANUAL_GOLD_10}` — verify/correct 10 complex sentences for evaluation. "
        f"Raw selection saved to `{SELECTED_50}`."
    )

    rows = []
    for b in bundles:
        p = b.gold or b.spacy
        rows.append({
            "ID": b.sent_id,
            "Text": b.text,
            "Tokens": len(p.tokens) if p else 0,
            "Has verified gold": "✓" if b.has_gold() else "—",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", height=400)

    csv = df.to_csv(index=False)
    st.download_button("Export dataset index (CSV)", csv, "dataset_index.csv", "text/csv")


def parsing_page(bundles, spacy_nlp):
    st.subheader("Sentence Explorer & Parse Trees")
    query = st.text_input("Search sentences", placeholder="e.g. telescope, coordination, relative")
    filtered = filter_bundles(bundles, query)

    if not filtered:
        st.warning("No sentences match your search.")
        return

    options = {f"{b.sent_id}: {b.text[:70]}…" if len(b.text) > 70 else f"{b.sent_id}: {b.text}": b for b in filtered}
    choice = st.selectbox("Select sentence", list(options.keys()))
    bundle = options[choice]

    st.markdown(f"**Full text:** {bundle.text}")
    meta_cols = st.columns(3)
    meta_cols[0].caption(f"ID: `{bundle.sent_id}`")
    if bundle.metadata:
        meta_cols[1].caption(f"Genre: {bundle.metadata.get('genre', '—')}")
    disagreements = token_disagreements(bundle.spacy, bundle.stanza) if bundle.spacy and bundle.stanza else []
    meta_cols[2].caption(f"Parser disagreements: **{len(disagreements)}** tokens")

    tab_vis, tab_dep, tab_tok = st.tabs(["Tree visualization", "spaCy displaCy", "Token tables"])

    with tab_vis:
        if not bundle.spacy or not bundle.stanza:
            st.error("Parser outputs missing for this sentence. Reload the dataset from the sidebar.")
        else:
            disagree = _disagreement_indices(bundle.spacy, bundle.stanza)
            if disagree:
                st.info(
                    "Tokens highlighted in **red** are attachment disagreements between spaCy and Stanza."
                )

            panels: list[tuple[str, object, str, bool]] = [
                ("spaCy (transition-based)", bundle.spacy, PARSER_COLORS["spacy"], True),
                ("Stanza (graph-based)", bundle.stanza, PARSER_COLORS["stanza"], True),
            ]
            if bundle.has_gold():
                panels.append(("Gold (UD)", bundle.gold, PARSER_COLORS["gold"], False))

            cols = st.columns(len(panels))
            for col, (title, parse, color, use_highlight) in zip(cols, panels):
                with col:
                    st.markdown(f"**{title}**")
                    svg = render_dependency_svg(
                        parse,
                        title,
                        disagree if use_highlight else None,
                        color=color,
                    )
                    panel_h = tree_panel_height(parse)
                    st.components.v1.html(
                        wrap_html_fragment(svg, panel_h),
                        height=panel_h + 40,
                        scrolling=False,
                    )

            st.download_button(
                "Download comparison HTML",
                render_comparison_html(
                    bundle.spacy,
                    bundle.stanza,
                    bundle.gold if bundle.has_gold() else None,
                ),
                file_name=f"parse_{bundle.sent_id}.html",
                mime="text/html",
            )

    with tab_dep:
        st.components.v1.html(spacy_displacy_html(spacy_nlp, bundle.text), height=450, scrolling=True)

    with tab_tok:
        disagree_idx = {d["index"] for d in disagreements}
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            st.markdown("**spaCy**")
            st.markdown(token_table_html(bundle.spacy, disagree_idx), unsafe_allow_html=True)
        with tcol2:
            st.markdown("**Stanza**")
            st.markdown(token_table_html(bundle.stanza, disagree_idx), unsafe_allow_html=True)
        with tcol3:
            if bundle.has_gold():
                st.markdown("**Gold**")
                st.markdown(token_table_html(bundle.gold, disagree_idx), unsafe_allow_html=True)


def evaluation_page(bundles):
    st.subheader("Parser Comparison & UAS")
    uas_df = build_uas_dataframe(bundles)
    per_sent = build_per_sentence_uas(bundles)

    st.plotly_chart(uas_comparison_chart(uas_df), width="stretch")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(per_sentence_uas_chart(per_sent), width="stretch")
    with col2:
        st.dataframe(uas_df, width="stretch", hide_index=True)

    st.subheader("Per-sentence metrics")
    st.dataframe(per_sent, width="stretch", height=320)

    st.download_button(
        "Export UAS results (CSV)",
        uas_df.to_csv(index=False),
        "uas_scores.csv",
        "text/csv",
    )

    gold_bundles = [b for b in bundles if b.has_gold()]
    if gold_bundles:
        st.subheader("Attachment accuracy detail (gold subset)")
        detail = []
        for b in gold_bundles[:15]:
            s_uas, _, _ = uas(b.gold, b.spacy)
            t_uas, _, _ = uas(b.gold, b.stanza)
            detail.append({
                "Sentence": b.text[:60] + "…",
                "spaCy UAS": s_uas,
                "Stanza UAS": t_uas,
                "Δ": round(abs(s_uas - t_uas), 3),
            })
        st.dataframe(pd.DataFrame(detail), width="stretch")


def error_analysis_page(bundles):
    st.subheader("Structured Error Analysis")
    err_df = build_error_table(bundles)

    if err_df.empty:
        st.success("No parser disagreements found on this corpus.")
        return

    types = ["All"] + sorted(err_df["Error Type"].unique().tolist())
    selected_type = st.selectbox("Filter by error type", types)
    filtered = err_df if selected_type == "All" else err_df[err_df["Error Type"] == selected_type]

    sort_col = st.selectbox("Sort by", ["Disagreement Count", "Sentence", "Error Type"])
    ascending = st.checkbox("Ascending", value=False)
    filtered = filtered.sort_values(sort_col, ascending=ascending)

    st.plotly_chart(disagreement_frequency_chart(filtered), width="stretch")

    st.download_button(
        "Export error table (CSV)",
        filtered.to_csv(index=False),
        "error_analysis_table.csv",
        "text/csv",
    )

    for _, row in filtered.iterrows():
        with st.expander(
            f"{row['Error Type']} · {row['Disagreement Count']} tokens · "
            f"{row['Sentence'][:55]}…"
        ):
            st.markdown(f"**Sentence:** {row['Sentence']}")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**spaCy:** `{row['spaCy Parse']}`")
            c2.markdown(f"**Stanza:** `{row['Stanza Parse']}`")
            c3.markdown(f"**Gold:** `{row['Gold Standard']}`")
            st.caption(f"Tokens: {row['Disagreement Tokens']}")

            match = next((b for b in bundles if b.text == row["Sentence"]), None)
            if match and match.spacy and match.stanza:
                html_doc = render_comparison_html(
                    match.spacy,
                    match.stanza,
                    match.gold if match.has_gold() else None,
                )
                st.components.v1.html(html_doc, height=420, scrolling=True)


def ambiguity_page(spacy_nlp, stanza_nlp):
    st.subheader("Ambiguity Case Study")
    st.markdown(
        f'<div class="ambig-box"><strong>Target sentence:</strong> «{TELESCOPE_SENTENCE}»<br>'
        "Classic prepositional phrase attachment ambiguity.</div>",
        unsafe_allow_html=True,
    )

    gold = load_ambiguous_gold()
    bundle = parse_single(TELESCOPE_SENTENCE, spacy_nlp, stanza_nlp, gold=gold, sent_id="ambiguous-pp-001")

    spacy_pp = pp_attachment_target(bundle.spacy, "with")
    stanza_pp = pp_attachment_target(bundle.stanza, "with")
    gold_pp = pp_attachment_target(bundle.gold, "with") if bundle.gold else "saw"

    c1, c2, c3 = st.columns(3)
    c1.metric("spaCy attaches «with» to", spacy_pp or "—")
    c2.metric("Stanza attaches «with» to", stanza_pp or "—")
    c3.metric("Gold attaches «with» to", gold_pp or "—")

    st.markdown("### Linguistic readings")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### If PP attaches to **saw** (verb)")
        st.markdown(attachment_explanation("saw")["description"])
    with col_b:
        st.markdown("#### If PP attaches to **girl** (noun)")
        st.markdown(attachment_explanation("girl")["description"])

    st.markdown("### Parser interpretations")
    for name, parse, head in [("spaCy", bundle.spacy, spacy_pp), ("Stanza", bundle.stanza, stanza_pp)]:
        expl = attachment_explanation(head)
        st.markdown(f"**{name}** → {expl['reading']}: {expl['description']}")

    st.markdown("### Side-by-side parses")
    st.components.v1.html(
        render_comparison_html(bundle.spacy, bundle.stanza, bundle.gold),
        height=420,
        scrolling=True,
    )

    st.markdown("### Why parsers may differ")
    st.markdown(
        """
        - **Algorithm:** spaCy commits with local transitions; Stanza scores arcs globally.
        - **Training bias:** post-object PPs often attach to nouns in English treebanks.
        - **Labeling:** spaCy uses `prep`/`pobj`; Stanza uses UD `case`/`obl`/`nmod`.
        - **Semantics:** neither parser models world knowledge — only distributional patterns.
        """
    )

    st.components.v1.html(spacy_displacy_html(spacy_nlp, TELESCOPE_SENTENCE), height=420, scrolling=True)


def main():
    init_session_state()
    page = sidebar()
    header()

    try:
        bundles = get_bundles()
        spacy_nlp = load_spacy()
    except Exception as exc:
        st.error(f"Failed to initialize parsers: {exc}")
        st.markdown(
            """
            **Local setup**
            ```bash
            pip install -r requirements.txt
            streamlit run app.py
            ```
            The spaCy English model is installed from `requirements.txt` (no `spacy download` needed).

            **Streamlit Community Cloud** ([share.streamlit.io](https://share.streamlit.io/))
            - Main file: `app.py`
            - Python: 3.10–3.12
            - Commit `requirements.txt` with the `en-core-web-sm` wheel line (already included)
            - First launch downloads Stanza English models (~500MB); wait 2–5 minutes
            - Free tier needs ~1GB RAM; upgrade if the app crashes on load
            """
        )
        return

    pages = {
        "Overview": lambda: overview_page(bundles),
        "Dataset": lambda: dataset_page(bundles),
        "Parsing & Trees": lambda: parsing_page(bundles, spacy_nlp),
        "Evaluation": lambda: evaluation_page(bundles),
        "Error Analysis": lambda: error_analysis_page(bundles),
        "Ambiguity Case Study": lambda: ambiguity_page(spacy_nlp, load_stanza()),
    }
    pages[page]()


if __name__ == "__main__":
    main()

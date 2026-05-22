# Dependency Parser Comparative Analysis

This project compares two dependency parsing approaches on Universal Dependencies English data:

- **spaCy**: transition-based dependency parser
- **Stanza**: graph/neural biaffine-style dependency parser

It selects 50 sentences from the UD English EWT dataset, supports manual gold correction for 10 complex sentences, parses the sentences with both systems, calculates UAS, identifies parser disagreements, categorizes errors, and generates parse-tree visualizations.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import stanza; stanza.download('en')"
```

## Interactive Streamlit dashboard

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The dashboard provides:

- Sentence explorer with search and side-by-side parse trees (SVG + displaCy)
- UAS evaluation, disagreement charts, and error taxonomy filtering
- Ambiguity case study for *"The man saw the girl with the telescope."*
- CSV export for UAS and error analysis tables
- Automatic spaCy/Stanza model download on first run

## Run the CLI pipeline

```bash
python src/prepare_ud_data.py
python src/manual_gold_template.py
# Edit data/manual_gold_10.conllu if needed after inspection.
python src/run_parsers.py
python src/evaluate.py
python src/visualize_parses.py --sentence "The man saw the girl with the telescope."
```

## Main outputs

- `data/selected_50.conllu` — 50 selected UD English sentences
- `data/manual_gold_10.conllu` — 10 complex sentences for manual verification/correction
- `outputs/spacy_parses.conllu` — spaCy predicted heads/deprels
- `outputs/stanza_parses.conllu` — Stanza predicted heads/deprels
- `outputs/uas_scores.csv` — UAS comparison
- `outputs/error_analysis_table.csv` — disagreement and error taxonomy table
- `outputs/parse_tree_spacy.png` and `outputs/parse_tree_stanza.png` — visualizations
- `report/comparative_analysis_report.md` — report under 1000 words

## UAS definition used here

UAS = number of tokens with correct predicted syntactic head / total evaluated tokens.

Punctuation is excluded by default in `src/evaluate.py`, which is common in dependency parsing evaluation.

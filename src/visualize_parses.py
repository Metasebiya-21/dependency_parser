"""Generate dependency parse tree images for spaCy and Stanza.

Example:
python src/visualize_parses.py --sentence "The man saw the girl with the telescope."
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import spacy
import stanza

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def draw_tree(tokens, heads, labels, title, path):
    graph = nx.DiGraph()
    graph.add_node("ROOT")
    for i, tok in enumerate(tokens, 1):
        graph.add_node(f"{i}:{tok}")
    for i, (tok, head, dep) in enumerate(zip(tokens, heads, labels), 1):
        child = f"{i}:{tok}"
        parent = "ROOT" if head == 0 else f"{head}:{tokens[head-1]}"
        graph.add_edge(parent, child, label=dep)

    pos = nx.nx_pydot.graphviz_layout(graph, prog="dot")
    plt.figure(figsize=(12, 7))
    nx.draw(graph, pos, with_labels=True, node_size=2400, font_size=9, arrows=True)
    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def parse_spacy(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = [t.text for t in doc]
    heads = [0 if t.head.i == t.i else t.head.i + 1 for t in doc]
    labels = [t.dep_ for t in doc]
    return tokens, heads, labels


def parse_stanza(text):
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", tokenize_no_ssplit=True)
    doc = nlp(text)
    words = [w for sent in doc.sentences for w in sent.words]
    tokens = [w.text for w in words]
    heads = [w.head for w in words]
    labels = [w.deprel for w in words]
    return tokens, heads, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", required=True)
    args = parser.parse_args()

    s_tokens, s_heads, s_labels = parse_spacy(args.sentence)
    t_tokens, t_heads, t_labels = parse_stanza(args.sentence)
    draw_tree(s_tokens, s_heads, s_labels, "spaCy dependency parse", OUT_DIR / "parse_tree_spacy.png")
    draw_tree(t_tokens, t_heads, t_labels, "Stanza dependency parse", OUT_DIR / "parse_tree_stanza.png")
    print("Wrote outputs/parse_tree_spacy.png and outputs/parse_tree_stanza.png")


if __name__ == "__main__":
    main()

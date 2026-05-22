# Dependency Parser Comparative Analysis Report

This project compares two English dependency parsers on 50 sentences selected from the Universal Dependencies English EWT treebank. The comparison focuses on syntactic ambiguity and attachment decisions, especially prepositional phrase attachment, coordination scope, relative clause attachment, and noun compound parsing.

## Dataset and Gold Standard

The evaluation uses UD English EWT because it is a gold-standard English dependency corpus containing web text from reviews, email, weblogs, newsgroups, and question-answer data. The script `prepare_ud_data.py` downloads the UD development file and selects 50 sentences with higher syntactic complexity. The first 10 selected complex sentences are copied into `manual_gold_10.conllu` for human verification. The intended workflow is to inspect and correct HEAD and DEPREL columns before running final evaluation.

## Parser Approaches

spaCy represents the transition-based approach. A transition-based parser builds a dependency tree incrementally through a sequence of local actions such as shift, reduce, left-arc, and right-arc. This makes parsing fast and practical for production pipelines, but local decisions can affect later structure. Such parsers may struggle when an early attachment looks plausible but later context supports another analysis.

Stanza represents the graph-based approach. Graph-based parsing scores possible head-dependent arcs over the sentence and searches for a high-scoring tree. This makes it more globally informed: instead of committing only through local transitions, it can compare competing attachment structures. This can help with ambiguity, but errors still occur when lexical cues or training distributions favor the wrong head.

## Evaluation Method

The evaluation script calculates Unlabeled Attachment Score (UAS): the percentage of non-punctuation tokens whose predicted head matches the gold head. It also identifies sentences where spaCy and Stanza disagree and assigns an error category using dependency-label and construction-based rules. The generated table contains each sentence, a compact spaCy parse, a compact Stanza parse, the gold parse, disagreement tokens, and an error type.

## Expected Error Patterns

Prepositional phrase attachment is expected to produce many disagreements because a PP may modify a verb, noun, or entire clause. For example, in “The man saw the girl with the telescope,” the phrase “with the telescope” can describe the instrument used for seeing or the girl possessing the telescope. Coordination errors arise when the parser must decide whether a modifier or dependent applies to one conjunct or the whole coordination. Relative clause errors occur when a clause can attach to different possible nouns. Noun compound errors occur when multiword nominals can be bracketed in more than one way.

## Ambiguous Sentence Analysis

Sentence: “The man saw the girl with the telescope.”

There are two plausible attachments. If “with the telescope” attaches to “saw,” the interpretation is instrumental: the man used a telescope to see the girl. If it attaches to “girl,” the interpretation is nominal modification: the girl had or was associated with the telescope.

A transition-based parser such as spaCy may attach the PP according to local lexical and structural preferences learned from data. In many English parsing models, a PP immediately following an object noun is often attached to that noun, especially when the noun is a plausible possessor or descriptor. A graph-based parser such as Stanza may choose a different attachment if its global scoring gives higher probability to the verb-attachment analysis, particularly because “with” often marks instruments for perception/action verbs. The exact result should be confirmed by running `visualize_parses.py`, because model versions can differ.

## Conclusion

The project is designed to show that the two parsers can achieve similar overall UAS while disagreeing on linguistically important ambiguity cases. spaCy is likely faster and easier to integrate; Stanza may offer more globally optimized attachment decisions. The strongest analysis comes not only from aggregate UAS, but from the error table, because attachment ambiguities often affect sentence meaning even when the total number of incorrect heads is small.

"""
M6 -- Evaluation module for SignScribe.

Runs LOCALLY.  Implements the three evaluation metrics from the synopsis:

1. Classification metrics (Accuracy, Precision, Recall, F1) via sklearn
2. BLEU-4 (LLM fluency vs reference sentences) via sacrebleu
3. Hallucination Rate (custom metric)

Hallucination Rate definition (proposed):
    For each generated sentence, a word is "unsupported" if it cannot be
    traced back to any gloss token in the input.  Specifically:
    - We build a set of "supported" English words from the input glosses
      by expanding each gloss to its known English equivalents (verb forms,
      synonyms, articles/prepositions that the grammar naturally inserts).
    - Any word in the generated sentence NOT in this supported set is
      counted as a hallucination.
    - Hallucination Rate = count(unsupported words) / count(total words)

    Common function words (articles, prepositions, pronouns, auxiliaries)
    are whitelisted as always-supported, since any translation system is
    expected to insert them.

Usage:
    python -m src.evaluation.metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ── BLEU-4 ──────────────────────────────────────────────────────────

def compute_bleu(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute BLEU-4 score using sacrebleu.

    Parameters
    ----------
    predictions : list[str]
        Generated sentences (one per sample).
    references : list[str]
        Reference sentences (one per sample).

    Returns
    -------
    dict with keys: bleu, precisions (list of 4), brevity_penalty, ratio
    """
    import sacrebleu

    # sacrebleu expects references as list of lists (multiple refs per sample)
    refs_wrapped = [[r] for r in references]

    # Use sentence-level BLEU averaged across all samples
    scores = []
    for pred, ref_list in zip(predictions, refs_wrapped):
        score = sacrebleu.sentence_bleu(pred, ref_list)
        scores.append(score.score)

    # Also compute corpus-level BLEU
    corpus_bleu = sacrebleu.corpus_bleu(predictions, [references])

    return {
        "corpus_bleu4": corpus_bleu.score,
        "avg_sentence_bleu4": float(np.mean(scores)) if scores else 0.0,
        "num_samples": len(predictions),
    }


# ── Hallucination Rate ──────────────────────────────────────────────

# Function words that are always considered "supported" because any
# translation system is expected to insert them freely.
FUNCTION_WORDS = {
    # Articles
    "a", "an", "the",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "my", "your",
    "his", "her", "its", "our", "their", "mine", "yours",
    # Prepositions
    "to", "in", "on", "at", "for", "with", "from", "by", "of", "about",
    "into", "out", "up", "down", "over", "under", "between",
    # Auxiliaries & common verbs
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "don't", "doesn't", "didn't",
    "will", "would", "can", "could", "shall", "should", "may", "might",
    "have", "has", "had",
    "not", "no", "yes",
    # Conjunctions
    "and", "or", "but", "so", "if", "that", "because", "when", "while",
    # Punctuation-like
    ".", ",", "!", "?", "'s",
}

# Mapping from gloss tokens to the English words they can legitimately
# produce.  This is used to determine what words in the output are
# "supported" by the input glosses.
from src.grammar_mapper.rule_mapper import VERB_FORMS, FIXED_PHRASES


def _get_supported_words(gloss_tokens: list[str]) -> set[str]:
    """Build the set of English words that are 'supported' by the glosses.

    A word is supported if it can be traced back to a gloss token through:
    - Direct lowercasing of the gloss itself
    - Known verb conjugations
    - Known fixed phrase expansions
    """
    supported = set(FUNCTION_WORDS)  # always allowed

    for gloss in gloss_tokens:
        g = gloss.upper()

        # The gloss itself (lowercased)
        supported.add(g.lower())

        # Verb forms
        if g in VERB_FORMS:
            base, third, past = VERB_FORMS[g]
            supported.update({base, third, past})

        # Fixed phrases (split into words)
        if g in FIXED_PHRASES:
            phrase_words = FIXED_PHRASES[g].lower().rstrip(".!?").split()
            supported.update(phrase_words)

    return supported


def compute_hallucination_rate(
    predictions: list[str],
    gloss_inputs: list[str | list[str]],
) -> dict[str, float]:
    """Compute the hallucination rate for generated sentences.

    Parameters
    ----------
    predictions : list[str]
        Generated sentences.
    gloss_inputs : list[str or list[str]]
        The input gloss tokens for each prediction.

    Returns
    -------
    dict with keys:
        hallucination_rate: float (0.0 = no hallucinations, 1.0 = all hallucinated)
        total_words: int
        unsupported_words: int
        per_sample_rates: list[float]
    """
    total_words = 0
    unsupported_words = 0
    per_sample_rates: list[float] = []

    for pred, glosses in zip(predictions, gloss_inputs):
        # Parse glosses
        if isinstance(glosses, str):
            gloss_tokens = glosses.strip().upper().split()
        else:
            gloss_tokens = [g.upper() for g in glosses]

        supported = _get_supported_words(gloss_tokens)

        # Tokenize the prediction (simple whitespace + lowercase)
        pred_words = pred.lower().strip().rstrip(".!?").split()

        if not pred_words:
            per_sample_rates.append(0.0)
            continue

        sample_unsupported = 0
        for word in pred_words:
            # Strip punctuation from word edges
            clean = word.strip(".,!?;:'\"()-")
            if clean and clean not in supported:
                sample_unsupported += 1

        total_words += len(pred_words)
        unsupported_words += sample_unsupported

        rate = sample_unsupported / len(pred_words)
        per_sample_rates.append(rate)

    overall_rate = unsupported_words / max(total_words, 1)

    return {
        "hallucination_rate": overall_rate,
        "total_words": total_words,
        "unsupported_words": unsupported_words,
        "avg_per_sample_rate": float(np.mean(per_sample_rates)) if per_sample_rates else 0.0,
        "num_samples": len(predictions),
    }


# ── Classification Metrics ──────────────────────────────────────────

def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] | None = None,
) -> dict:
    """Compute standard classification metrics.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1,
                    per_class_report (str), confusion_matrix (list)
    """
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    result = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_samples": len(y_true),
    }

    if label_names:
        present_labels = sorted(set(y_true + y_pred))
        target_names = [label_names[i] for i in present_labels if i < len(label_names)]
        result["classification_report"] = classification_report(
            y_true, y_pred,
            target_names=target_names,
            zero_division=0,
        )
        result["confusion_matrix"] = confusion_matrix(
            y_true, y_pred
        ).tolist()

    return result


# ── Full Evaluation Report ──────────────────────────────────────────

def run_full_evaluation(
    classifier_true: list[int],
    classifier_pred: list[int],
    label_names: list[str],
    rule_predictions: list[str],
    llm_predictions: list[str],
    reference_sentences: list[str],
    gloss_inputs: list[str | list[str]],
    output_path: Path | None = None,
) -> dict:
    """Run the complete evaluation and optionally save results.

    Parameters
    ----------
    classifier_true, classifier_pred : list[int]
        True and predicted class labels for the classifier.
    label_names : list[str]
        Class label names.
    rule_predictions : list[str]
        Sentences generated by Path A (rule-based).
    llm_predictions : list[str]
        Sentences generated by Path B (LLM).
    reference_sentences : list[str]
        Ground-truth English sentences.
    gloss_inputs : list[str or list[str]]
        Input glosses corresponding to each prediction.
    output_path : Path or None
        If given, save the report as JSON.

    Returns
    -------
    dict
        Complete evaluation results.
    """
    print("=" * 60)
    print("  SignScribe - Full Evaluation Report")
    print("=" * 60)

    # 1. Classification metrics
    print("\n--- Classification Metrics ---")
    clf_metrics = compute_classification_metrics(
        classifier_true, classifier_pred, label_names
    )
    print(f"  Accuracy  : {clf_metrics['accuracy']:.4f}")
    print(f"  Precision : {clf_metrics['precision']:.4f}")
    print(f"  Recall    : {clf_metrics['recall']:.4f}")
    print(f"  F1 Score  : {clf_metrics['f1']:.4f}")
    if "classification_report" in clf_metrics:
        print(f"\n{clf_metrics['classification_report']}")

    # 2. BLEU-4 for both paths
    print("--- BLEU-4 Scores ---")
    rule_bleu = compute_bleu(rule_predictions, reference_sentences)
    llm_bleu = compute_bleu(llm_predictions, reference_sentences)
    print(f"  Path A (Rule-based):")
    print(f"    Corpus BLEU-4    : {rule_bleu['corpus_bleu4']:.2f}")
    print(f"    Avg Sentence BLEU: {rule_bleu['avg_sentence_bleu4']:.2f}")
    print(f"  Path B (LLM):")
    print(f"    Corpus BLEU-4    : {llm_bleu['corpus_bleu4']:.2f}")
    print(f"    Avg Sentence BLEU: {llm_bleu['avg_sentence_bleu4']:.2f}")

    # 3. Hallucination rate for both paths
    print("\n--- Hallucination Rate ---")
    rule_hall = compute_hallucination_rate(rule_predictions, gloss_inputs)
    llm_hall = compute_hallucination_rate(llm_predictions, gloss_inputs)
    print(f"  Path A (Rule-based):")
    print(f"    Rate: {rule_hall['hallucination_rate']:.4f} "
          f"({rule_hall['unsupported_words']}/{rule_hall['total_words']} words)")
    print(f"  Path B (LLM):")
    print(f"    Rate: {llm_hall['hallucination_rate']:.4f} "
          f"({llm_hall['unsupported_words']}/{llm_hall['total_words']} words)")

    # 4. Side-by-side comparison
    print("\n--- Side-by-Side Comparison (first 10 samples) ---")
    for i in range(min(10, len(reference_sentences))):
        gloss_str = gloss_inputs[i] if isinstance(gloss_inputs[i], str) else " ".join(gloss_inputs[i])
        print(f"\n  [{i+1}] Gloss    : {gloss_str}")
        print(f"      Reference : {reference_sentences[i]}")
        print(f"      Path A    : {rule_predictions[i]}")
        print(f"      Path B    : {llm_predictions[i]}")

    print("\n" + "=" * 60)

    # Compile full report
    report = {
        "classification": clf_metrics,
        "bleu4": {
            "rule_based": rule_bleu,
            "llm": llm_bleu,
        },
        "hallucination_rate": {
            "rule_based": rule_hall,
            "llm": llm_hall,
        },
    }

    # Remove non-serializable items
    if "confusion_matrix" in report["classification"]:
        pass  # already list
    if "classification_report" in report["classification"]:
        pass  # str is fine

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {output_path}")

    return report


# ── Demo / CLI ──────────────────────────────────────────────────────

def demo():
    """Run a small demo evaluation with synthetic data."""
    print("Running evaluation demo with synthetic data...\n")

    # Synthetic classifier results
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    y_pred = [0, 1, 2, 0, 1, 1, 0, 1, 2, 2]
    label_names = ["A", "B", "C"]

    # Synthetic translation results
    glosses = [
        "I SCHOOL GO",
        "HE FOOD EAT",
        "SHE BOOK READ",
        "I WATER DRINK FUTURE",
        "HELLO",
    ]
    references = [
        "I go to school.",
        "He eats food.",
        "She reads a book.",
        "I will drink water.",
        "Hello.",
    ]
    rule_preds = [
        "I go to school.",
        "He eat a food.",
        "She reads a book.",
        "I will drink a water.",
        "Hello.",
    ]
    llm_preds = [
        "I go to school every day.",
        "He eats food.",
        "She reads a book.",
        "I will drink water tomorrow.",
        "Hello there!",
    ]

    report = run_full_evaluation(
        classifier_true=y_true,
        classifier_pred=y_pred,
        label_names=label_names,
        rule_predictions=rule_preds,
        llm_predictions=llm_preds,
        reference_sentences=references,
        gloss_inputs=glosses,
    )


if __name__ == "__main__":
    demo()

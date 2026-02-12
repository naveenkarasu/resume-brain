"""Shared training metrics for NER and scoring models.

Provides unified metric computation for:
- Token-level / entity-level NER evaluation (via seqeval)
- Regression and ranking metrics (Spearman, NDCG)
- Text generation quality (BLEU, ROUGE)

All public functions accept plain Python lists or NumPy arrays and return
simple Python scalars or dicts so they can be used directly in training
loops, evaluation scripts, or HuggingFace Trainer ``compute_metrics``
callbacks.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NER / sequence-labelling metrics
# ---------------------------------------------------------------------------


def entity_level_f1(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    *,
    average: str = "micro",
    scheme: Optional[str] = None,
    zero_division: int = 0,
) -> Dict[str, float]:
    """Compute entity-level precision, recall, and F1 using *seqeval*.

    Parameters
    ----------
    true_labels:
        Ground-truth BIO/BIOES tag sequences.  Each inner list corresponds
        to one sentence / example and contains one string tag per token.
    pred_labels:
        Predicted tag sequences with the same shape as *true_labels*.
    average:
        Averaging strategy forwarded to ``seqeval.metrics``.  One of
        ``"micro"`` (default), ``"macro"``, or ``"weighted"``.
    scheme:
        Optional labelling scheme (``"IOB2"``, ``"IOBES"``, etc.).  When
        ``None``, seqeval will attempt automatic detection.
    zero_division:
        Value to return for metrics when there are no positive predictions
        or no positive labels (0 or 1).

    Returns
    -------
    dict
        ``{"precision": float, "recall": float, "f1": float}``

    Raises
    ------
    ImportError
        If *seqeval* is not installed.
    ValueError
        If the outer lengths of *true_labels* and *pred_labels* differ, or
        if any inner pair has mismatched lengths.
    """
    try:
        from seqeval.metrics import (
            precision_score,
            recall_score,
            f1_score,
        )
        from seqeval.scheme import IOB2, IOBES  # noqa: F401 – used via getattr
    except ImportError as exc:
        raise ImportError(
            "seqeval is required for entity-level evaluation.  "
            "Install it with:  pip install seqeval"
        ) from exc

    if len(true_labels) != len(pred_labels):
        raise ValueError(
            f"Outer sequence lengths differ: "
            f"{len(true_labels)} vs {len(pred_labels)}"
        )
    for idx, (t, p) in enumerate(zip(true_labels, pred_labels)):
        if len(t) != len(p):
            raise ValueError(
                f"Sequence {idx}: inner lengths differ ({len(t)} vs {len(p)})"
            )

    # Resolve optional scheme object for seqeval.
    scheme_cls = None
    if scheme is not None:
        import seqeval.scheme as _scheme_mod

        scheme_cls = getattr(_scheme_mod, scheme.upper(), None)

    kwargs: Dict[str, Any] = dict(
        y_true=true_labels,
        y_pred=pred_labels,
        average=average,
        zero_division=zero_division,
    )
    if scheme_cls is not None:
        kwargs["scheme"] = scheme_cls
        kwargs["mode"] = "strict"

    precision = precision_score(**kwargs)
    recall = recall_score(**kwargs)
    f1 = f1_score(**kwargs)

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


# ---------------------------------------------------------------------------
# Regression / ranking metrics
# ---------------------------------------------------------------------------

_ArrayLike = Union[Sequence[float], "np.ndarray"]


def spearman_correlation(
    y_true: _ArrayLike,
    y_pred: _ArrayLike,
) -> Dict[str, float]:
    """Compute Spearman rank-order correlation between two score vectors.

    Parameters
    ----------
    y_true:
        Ground-truth scores (continuous or ordinal).
    y_pred:
        Predicted scores with the same length.

    Returns
    -------
    dict
        ``{"spearman_rho": float, "p_value": float}``

    Raises
    ------
    ValueError
        If the input lengths do not match or are empty.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: {y_true_arr.shape} vs {y_pred_arr.shape}"
        )
    if y_true_arr.size == 0:
        raise ValueError("Inputs must not be empty")

    rho, p_value = spearmanr(y_true_arr, y_pred_arr)
    return {"spearman_rho": float(rho), "p_value": float(p_value)}


def ndcg_at_k(
    y_true: _ArrayLike,
    y_pred: _ArrayLike,
    k: int = 5,
) -> float:
    """Compute Normalised Discounted Cumulative Gain at rank *k*.

    Items are ranked by *y_pred* in descending order and relevance is taken
    from the corresponding *y_true* values.

    Parameters
    ----------
    y_true:
        Ground-truth relevance scores (non-negative).
    y_pred:
        Predicted scores used for ranking.
    k:
        Truncation depth.  Must be >= 1.

    Returns
    -------
    float
        NDCG@k in the range [0, 1].  Returns 0.0 when the ideal DCG is
        zero (i.e., all true relevances are zero).

    Raises
    ------
    ValueError
        If inputs are empty, have different lengths, or *k* < 1.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: {y_true_arr.shape} vs {y_pred_arr.shape}"
        )
    if y_true_arr.size == 0:
        raise ValueError("Inputs must not be empty")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    def _dcg(relevances: np.ndarray, topk: int) -> float:
        relevances = relevances[:topk]
        discounts = np.log2(np.arange(2, len(relevances) + 2))
        return float(np.sum(relevances / discounts))

    # Rank items by predicted score (descending).
    ranked_indices = np.argsort(y_pred_arr)[::-1]
    ranked_relevances = y_true_arr[ranked_indices]

    dcg = _dcg(ranked_relevances, k)

    # Ideal ranking: sort true relevances descending.
    ideal_relevances = np.sort(y_true_arr)[::-1]
    idcg = _dcg(ideal_relevances, k)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Text generation metrics (BLEU / ROUGE wrappers)
# ---------------------------------------------------------------------------


def compute_bleu(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    *,
    max_order: int = 4,
    smooth: bool = True,
) -> Dict[str, float]:
    """Compute corpus-level or sentence-level BLEU score.

    Wraps ``sacrebleu`` when available, falling back to ``nltk.translate``
    for basic sentence BLEU.

    Parameters
    ----------
    reference:
        A single reference string **or** a list of reference strings.
        When a list is provided, it is treated as multiple reference
        sentences (corpus-level evaluation) unless *hypothesis* is a
        single string.
    hypothesis:
        A single hypothesis string or a list of hypothesis strings.
    max_order:
        Maximum n-gram order (default 4).
    smooth:
        Whether to apply smoothing (relevant for short sentences).

    Returns
    -------
    dict
        ``{"bleu": float, "brevity_penalty": float}`` with scores in
        [0, 1].
    """
    # ------------------------------------------------------------------
    # Try sacrebleu first (preferred for reproducible corpus BLEU).
    # ------------------------------------------------------------------
    try:
        import sacrebleu

        refs = [reference] if isinstance(reference, str) else reference
        hyps = [hypothesis] if isinstance(hypothesis, str) else hypothesis

        # sacrebleu expects refs as list-of-list (one list per reference set).
        bleu_result = sacrebleu.corpus_bleu(
            hyps,
            [refs],
            smooth_method="exp" if smooth else "none",
        )
        return {
            "bleu": bleu_result.score / 100.0,
            "brevity_penalty": float(bleu_result.bp),
        }
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # Fallback: nltk sentence_bleu (per-sentence, averaged).
    # ------------------------------------------------------------------
    try:
        from nltk.translate.bleu_score import (
            SmoothingFunction,
            sentence_bleu,
        )
    except ImportError as exc:
        raise ImportError(
            "Either sacrebleu or nltk is required for BLEU computation.  "
            "Install with:  pip install sacrebleu   or   pip install nltk"
        ) from exc

    smoother = SmoothingFunction().method1 if smooth else None
    weights = tuple(1.0 / max_order for _ in range(max_order))

    refs = [reference] if isinstance(reference, str) else reference
    hyps = [hypothesis] if isinstance(hypothesis, str) else hypothesis

    scores: list[float] = []
    for ref, hyp in zip(refs, hyps):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        score = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            weights=weights,
            smoothing_function=smoother,
        )
        scores.append(float(score))

    avg_bleu = sum(scores) / len(scores) if scores else 0.0

    # nltk does not directly expose BP per call, so we approximate it.
    ref_len = sum(len(r.split()) for r in refs)
    hyp_len = sum(len(h.split()) for h in hyps)
    bp = math.exp(min(0, 1 - ref_len / hyp_len)) if hyp_len > 0 else 0.0

    return {"bleu": avg_bleu, "brevity_penalty": bp}


def compute_rouge(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    *,
    rouge_types: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE scores between reference and hypothesis texts.

    Wraps the ``rouge-score`` library (Google implementation).

    Parameters
    ----------
    reference:
        A single reference string or a list of reference strings.
    hypothesis:
        A single hypothesis string or a list of hypothesis strings.
    rouge_types:
        Which ROUGE variants to compute.  Defaults to
        ``["rouge1", "rouge2", "rougeL"]``.

    Returns
    -------
    dict
        Nested dict keyed by ROUGE type, each containing
        ``{"precision": float, "recall": float, "fmeasure": float}``.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError(
            "rouge-score is required for ROUGE computation.  "
            "Install it with:  pip install rouge-score"
        ) from exc

    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    refs = [reference] if isinstance(reference, str) else reference
    hyps = [hypothesis] if isinstance(hypothesis, str) else hypothesis

    if len(refs) != len(hyps):
        raise ValueError(
            f"Number of references ({len(refs)}) and hypotheses "
            f"({len(hyps)}) must match"
        )

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    # Accumulate per-type scores.
    totals: Dict[str, Dict[str, float]] = {
        rt: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
        for rt in rouge_types
    }

    for ref, hyp in zip(refs, hyps):
        result = scorer.score(ref, hyp)
        for rt in rouge_types:
            totals[rt]["precision"] += result[rt].precision
            totals[rt]["recall"] += result[rt].recall
            totals[rt]["fmeasure"] += result[rt].fmeasure

    n = len(refs)
    return {
        rt: {k: v / n for k, v in totals[rt].items()}
        for rt in rouge_types
    }

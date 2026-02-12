"""NER-specific utilities for BIO tagging, label alignment, and evaluation.

Covers the full lifecycle of token-classification data preparation:

1. **Label alignment** -- mapping word-level BIO labels to subword tokens
   produced by HuggingFace tokenizers.
2. **BIO decoding** -- converting tag sequences back into typed entity spans.
3. **Validation** -- checking that BIO transitions are well-formed.
4. **Chunking** -- splitting long documents into training-friendly windows
   while preserving label alignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Sentinel used for subword tokens that should be ignored during loss
# computation (e.g., [CLS], [SEP], continuation subwords).
IGNORE_LABEL_ID: int = -100


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """A single named-entity span extracted from a BIO-tagged sequence."""

    entity_type: str
    start: int  # inclusive token index
    end: int  # exclusive token index
    tokens: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        """Reconstruct surface text by joining tokens with spaces."""
        return " ".join(self.tokens)

    def __repr__(self) -> str:
        return (
            f"Entity(type={self.entity_type!r}, "
            f"span=[{self.start}:{self.end}), "
            f"text={self.text!r})"
        )


# ---------------------------------------------------------------------------
# Label alignment
# ---------------------------------------------------------------------------


def align_labels_with_tokens(
    labels: List[str],
    word_ids: List[Optional[int]],
    *,
    label_all_subwords: bool = False,
    ignore_label: str = "O",
) -> List[str]:
    """Map word-level BIO labels to subword-level tokens.

    HuggingFace *fast* tokenizers return a ``word_ids`` list that maps each
    subword position to its original word index (or ``None`` for special
    tokens like ``[CLS]`` / ``[SEP]``).  This function produces a label
    for every subword position.

    Parameters
    ----------
    labels:
        BIO labels aligned with the **original** words.
    word_ids:
        Output of ``tokenizer(...).word_ids()``.  Has the same length as
        the subword token sequence.
    label_all_subwords:
        If ``True``, every continuation subword receives the same label as
        the first subword of that word.  If ``False`` (default), only the
        first subword receives the real label; continuations are assigned
        *ignore_label* which is later replaced by ``IGNORE_LABEL_ID`` in
        the collator / training loop.
    ignore_label:
        The placeholder label used for special tokens and (optionally)
        continuation subwords.  Defaults to ``"O"``.

    Returns
    -------
    list[str]
        Subword-aligned labels with the same length as *word_ids*.

    Raises
    ------
    IndexError
        If a ``word_id`` references an index outside *labels*.
    """
    aligned: List[str] = []
    previous_word_id: Optional[int] = None

    for word_id in word_ids:
        if word_id is None:
            # Special token -- ignored during loss computation.
            aligned.append(ignore_label)
        elif word_id != previous_word_id:
            # First subword of a new word -- use the real label.
            aligned.append(labels[word_id])
        else:
            # Continuation subword.
            if label_all_subwords:
                # Propagate I- tag (convert B- to I- for continuations).
                original = labels[word_id]
                if original.startswith("B-"):
                    aligned.append("I-" + original[2:])
                else:
                    aligned.append(original)
            else:
                aligned.append(ignore_label)
        previous_word_id = word_id

    return aligned


# ---------------------------------------------------------------------------
# BIO decoding
# ---------------------------------------------------------------------------


def bio_tags_to_entities(
    tags: Sequence[str],
    tokens: Optional[Sequence[str]] = None,
) -> List[Entity]:
    """Extract typed entity spans from a BIO-tagged sequence.

    Supports both **IOB2** (``B-`` always starts an entity) and the
    lenient convention where ``I-`` at the beginning of a sequence or after
    ``O`` implicitly starts a new entity.

    Parameters
    ----------
    tags:
        BIO tag for each token position.
    tokens:
        Optional surface-form tokens (same length as *tags*).  When
        provided, the extracted ``Entity`` objects will carry the
        corresponding token strings.

    Returns
    -------
    list[Entity]
        Entities in the order they appear in the sequence.
    """
    if tokens is not None and len(tokens) != len(tags):
        raise ValueError(
            f"tokens length ({len(tokens)}) != tags length ({len(tags)})"
        )

    entities: List[Entity] = []
    current_type: Optional[str] = None
    current_start: int = 0
    current_tokens: List[str] = []

    def _flush() -> None:
        if current_type is not None:
            entities.append(
                Entity(
                    entity_type=current_type,
                    start=current_start,
                    end=len(current_tokens) + current_start,
                    tokens=tuple(current_tokens),
                )
            )

    for idx, tag in enumerate(tags):
        if tag.startswith("B-"):
            _flush()
            current_type = tag[2:]
            current_start = idx
            current_tokens = [tokens[idx]] if tokens is not None else []
        elif tag.startswith("I-"):
            etype = tag[2:]
            if current_type == etype:
                # Continue the current entity.
                if tokens is not None:
                    current_tokens.append(tokens[idx])
            else:
                # Type mismatch or no open entity -- start new (lenient).
                _flush()
                current_type = etype
                current_start = idx
                current_tokens = [tokens[idx]] if tokens is not None else []
        else:
            # O tag or any unrecognised tag -- close current entity.
            _flush()
            current_type = None
            current_tokens = []

    # Flush any trailing entity.
    _flush()

    return entities


# ---------------------------------------------------------------------------
# BIO validation
# ---------------------------------------------------------------------------


def validate_bio_sequence(tags: Sequence[str]) -> List[Dict[str, object]]:
    """Check a BIO tag sequence for invalid transitions.

    Valid IOB2 transitions:
    - ``O``  -> ``B-X`` or ``O``
    - ``B-X`` -> ``I-X``, ``B-Y``, or ``O``
    - ``I-X`` -> ``I-X``, ``B-Y``, or ``O``

    An ``I-X`` tag is invalid after ``O`` or after ``B-Y`` / ``I-Y``
    where ``Y != X``.

    Parameters
    ----------
    tags:
        BIO tag sequence.

    Returns
    -------
    list[dict]
        A (possibly empty) list of violation dicts, each containing:
        ``{"position": int, "tag": str, "previous_tag": str, "reason": str}``.
    """
    violations: List[Dict[str, object]] = []
    prev_tag = "O"

    for idx, tag in enumerate(tags):
        if tag == "O":
            prev_tag = tag
            continue

        if tag.startswith("B-"):
            # B- is always valid.
            prev_tag = tag
            continue

        if tag.startswith("I-"):
            etype = tag[2:]
            if prev_tag == "O":
                violations.append(
                    {
                        "position": idx,
                        "tag": tag,
                        "previous_tag": prev_tag,
                        "reason": "I- tag follows O without a preceding B- tag",
                    }
                )
            elif prev_tag.startswith("B-") or prev_tag.startswith("I-"):
                prev_etype = prev_tag[2:]
                if prev_etype != etype:
                    violations.append(
                        {
                            "position": idx,
                            "tag": tag,
                            "previous_tag": prev_tag,
                            "reason": (
                                f"I-{etype} follows {prev_tag} "
                                f"(entity type mismatch)"
                            ),
                        }
                    )
            prev_tag = tag
            continue

        # Completely unrecognised tag format.
        violations.append(
            {
                "position": idx,
                "tag": tag,
                "previous_tag": prev_tag,
                "reason": "Tag does not match B-/I-/O pattern",
            }
        )
        prev_tag = tag

    return violations


# ---------------------------------------------------------------------------
# Chunking for long documents
# ---------------------------------------------------------------------------


def chunk_for_training(
    texts: List[List[str]],
    labels: List[List[str]],
    max_length: int = 512,
    *,
    stride: int = 0,
    ensure_bio_consistency: bool = True,
) -> List[Tuple[List[str], List[str]]]:
    """Split long token sequences into fixed-length chunks for training.

    Parameters
    ----------
    texts:
        List of documents, each represented as a list of word tokens.
    labels:
        Parallel list of BIO label sequences (same shape as *texts*).
    max_length:
        Maximum number of tokens per chunk.  Must be >= 1.
    stride:
        Number of overlapping tokens between consecutive chunks.  When 0
        (default), chunks are non-overlapping.  Must be < *max_length*.
    ensure_bio_consistency:
        If ``True`` (default), chunks that would start with an ``I-`` tag
        have that tag converted to the corresponding ``B-`` tag to keep
        the BIO sequence self-consistent.

    Returns
    -------
    list[tuple[list[str], list[str]]]
        Pairs of ``(token_chunk, label_chunk)``.

    Raises
    ------
    ValueError
        If *texts* and *labels* have different outer or inner lengths, or
        if parameter constraints are violated.
    """
    if len(texts) != len(labels):
        raise ValueError(
            f"texts ({len(texts)}) and labels ({len(labels)}) "
            f"must have the same number of documents"
        )
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}")
    if stride < 0 or stride >= max_length:
        raise ValueError(
            f"stride must be in [0, max_length), got stride={stride}, "
            f"max_length={max_length}"
        )

    step = max_length - stride if stride > 0 else max_length
    chunks: List[Tuple[List[str], List[str]]] = []

    for doc_idx, (doc_tokens, doc_labels) in enumerate(zip(texts, labels)):
        if len(doc_tokens) != len(doc_labels):
            raise ValueError(
                f"Document {doc_idx}: token length ({len(doc_tokens)}) != "
                f"label length ({len(doc_labels)})"
            )

        if len(doc_tokens) == 0:
            continue

        for start in range(0, len(doc_tokens), step):
            end = min(start + max_length, len(doc_tokens))
            tok_chunk = list(doc_tokens[start:end])
            lbl_chunk = list(doc_labels[start:end])

            if ensure_bio_consistency and lbl_chunk:
                first = lbl_chunk[0]
                if first.startswith("I-"):
                    lbl_chunk[0] = "B-" + first[2:]

            chunks.append((tok_chunk, lbl_chunk))

            # If we've reached the end, no need to continue stepping.
            if end == len(doc_tokens):
                break

    return chunks

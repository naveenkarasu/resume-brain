"""Training utility package -- metrics, NER helpers, and taxonomy loaders."""

from .metrics import (
    compute_bleu,
    compute_rouge,
    entity_level_f1,
    ndcg_at_k,
    spearman_correlation,
)
from .ner_utils import (
    IGNORE_LABEL_ID,
    Entity,
    align_labels_with_tokens,
    bio_tags_to_entities,
    chunk_for_training,
    validate_bio_sequence,
)
from .taxonomy import (
    Skill,
    TaxonomyGraph,
    build_skill_hierarchy,
    get_related_skills,
    load_esco_taxonomy,
    load_onet_taxonomy,
)

__all__ = [
    # metrics
    "entity_level_f1",
    "spearman_correlation",
    "ndcg_at_k",
    "compute_bleu",
    "compute_rouge",
    # ner_utils
    "IGNORE_LABEL_ID",
    "Entity",
    "align_labels_with_tokens",
    "bio_tags_to_entities",
    "validate_bio_sequence",
    "chunk_for_training",
    # taxonomy
    "Skill",
    "TaxonomyGraph",
    "load_esco_taxonomy",
    "load_onet_taxonomy",
    "build_skill_hierarchy",
    "get_related_skills",
]

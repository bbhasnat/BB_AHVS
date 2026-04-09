"""Duplicate detection & removal module — lexical, semantic, and hybrid dedup.

Three modes (configured via ``dedup_config.yaml`` or ``params``):

* **lexical** — MinHash LSH on whitespace tokens (Jaccard similarity).
  Fast, catches near-identical texts (retweets, copy-paste).
* **semantic** — Sentence-transformer embeddings + DBSCAN (cosine distance).
  Slower, catches paraphrases and semantic near-duplicates.
* **hybrid** — Lexical first (cheap), then semantic on survivors (thorough).

When ``deduplicate`` is True (the default), duplicates are removed and
``transformed_df`` is set so downstream modules operate on clean data.
A mapping of duplicate groups is saved for auditability.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ahvs.data_analyst.models import ModuleInput, ModuleResult

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "dedup_config.yaml"

# Minimum non-whitespace characters for a text to be eligible for dedup.
# Rows below this threshold are excluded from hashing/embedding to avoid
# false matches on empty/blank strings.
_MIN_TEXT_LEN = 1


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict[str, Any]:
    """Load default config from dedup_config.yaml.

    Returns hard-coded defaults on missing file, parse errors, or non-dict YAML.
    """
    try:
        if _CONFIG_PATH.exists():
            import yaml

            with open(_CONFIG_PATH) as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                return loaded
            logger.warning(
                "dedup_config.yaml is not a mapping (got %s) — using defaults.",
                type(loaded).__name__,
            )
    except Exception as exc:
        logger.warning("Failed to load dedup_config.yaml: %s — using defaults.", exc)
    return {}


def _merge_params(config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Overlay user params on top of file config (params win).

    Supports both flat keys (``lsh_threshold``) and nested dicts
    (``{"lexical": {"lsh_threshold": 0.7}}``).
    """
    merged = dict(config)

    # Top-level overrides
    for key in ("dedup_mode", "deduplicate"):
        if key in params:
            merged[key] = params[key]

    # Build section dicts, ensuring they are actual dicts
    lex = dict(merged.get("lexical", {})) if isinstance(merged.get("lexical"), dict) else {}
    sem = dict(merged.get("semantic", {})) if isinstance(merged.get("semantic"), dict) else {}
    hyb = dict(merged.get("hybrid", {})) if isinstance(merged.get("hybrid"), dict) else {}

    # Nested dict overrides (e.g. params={"lexical": {"lsh_threshold": 0.7}})
    if isinstance(params.get("lexical"), dict):
        lex.update(params["lexical"])
    if isinstance(params.get("semantic"), dict):
        sem.update(params["semantic"])
    if isinstance(params.get("hybrid"), dict):
        hyb.update(params["hybrid"])

    # Flat key overrides (convenience shorthand)
    for key in ("lsh_threshold", "lsh_num_perm"):
        if key in params:
            lex[key] = params[key]
    for key in ("embedding_model", "eps", "min_samples", "batch_size", "device",
                "max_cluster_size", "split_eps_factor"):
        if key in params:
            sem[key] = params[key]
    for key in ("skip_semantic_if_lexical_removed_pct",):
        if key in params:
            hyb[key] = params[key]

    merged["lexical"] = lex
    merged["semantic"] = sem
    merged["hybrid"] = hyb
    return merged


def _resolve_text_cols(df: pd.DataFrame, input_cols: list[str]) -> list[str]:
    """Return input columns that are text-like (object or string dtype)."""
    text_cols = []
    for c in input_cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            text_cols.append(c)
    return text_cols


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(inp: ModuleInput) -> ModuleResult:
    """Detect (and optionally remove) duplicates in the dataset."""
    df = inp.df
    params = inp.params
    output_dir = inp.output_dir / "duplicates"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _merge_params(_load_config(), params)
    mode = cfg.get("dedup_mode", "lexical")
    deduplicate = cfg.get("deduplicate", True)
    user_forced = params.get("_user_forced_dedup", False)

    # GPU guard: semantic/hybrid requires GPU unless user explicitly forced it.
    if mode in ("semantic", "hybrid") and not user_forced:
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False
        if not gpu_available:
            original_mode = mode
            mode = "lexical"
            logger.warning(
                "GPU unavailable — downgrading dedup from '%s' to 'lexical'. "
                "Use --dedup-mode %s to force CPU execution.",
                original_mode, original_mode,
            )

    # Use only the first text column for dedup to avoid cross-column union
    # removing unique rows because a low-cardinality auxiliary column repeats.
    all_text_cols = _resolve_text_cols(df, inp.input_cols)
    if not all_text_cols:
        return ModuleResult(
            module_name="duplicates",
            status="skipped",
            summary={"dedup_mode": mode, "dedup_applied": False},
            narrative="No text columns found for deduplication.",
        )
    text_cols = all_text_cols[:1]  # primary text column only

    summary: dict[str, Any] = {"dedup_mode": mode}
    warnings: list[str] = []
    artifacts: list[Path] = []
    narrative_parts: list[str] = []

    if len(all_text_cols) > 1:
        warnings.append(
            f"Multiple text columns detected ({all_text_cols}). "
            f"Deduplicating on primary column '{text_cols[0]}' only."
        )

    # ------------------------------------------------------------------
    # Exact duplicates on primary text column (always, regardless of mode)
    # ------------------------------------------------------------------
    all_dup_indices: set[int] = set()
    exact_mask = df.duplicated(subset=text_cols, keep="first")
    exact_dups = int(exact_mask.sum())
    summary["exact_duplicates"] = exact_dups
    if exact_dups:
        all_dup_indices.update(df.index[exact_mask].tolist())
        narrative_parts.append(f"{exact_dups} exact duplicate rows.")

    # ------------------------------------------------------------------
    # Mode dispatch
    # ------------------------------------------------------------------
    duplicate_groups: dict[str, list[list[int]]] = {}

    if mode == "lexical":
        lex_dups, lex_groups, lex_warnings = _run_lexical(
            df, text_cols, cfg.get("lexical", {}),
        )
        all_dup_indices.update(df.index[list(lex_dups)].tolist())
        duplicate_groups.update(lex_groups)
        warnings.extend(lex_warnings)
        col = text_cols[0]
        n_col = len(lex_dups)
        pct_col = round(n_col / len(df) * 100, 2) if len(df) else 0.0
        narrative_parts.append(
            f"Lexical: column '{col}' — {n_col} fuzzy duplicates ({pct_col}%)."
        )
        summary["lexical"] = {"fuzzy_duplicates": n_col, "pct": pct_col}
        # Backward-compat summary key
        summary["fuzzy_duplicates"] = {
            col: {"fuzzy_duplicates": n_col, "fuzzy_duplicate_pct": pct_col,
                  "threshold": cfg.get("lexical", {}).get("lsh_threshold", 0.85)}
        }

    elif mode == "semantic":
        sem_dups, sem_groups, sem_warnings, sem_ran = _run_semantic(
            df, text_cols, cfg.get("semantic", {}), output_dir,
        )
        all_dup_indices.update(df.index[list(sem_dups)].tolist())
        duplicate_groups.update({f"{k}_semantic": v for k, v in sem_groups.items()})
        warnings.extend(sem_warnings)
        if sem_ran:
            col = text_cols[0]
            n_col = len(sem_dups)
            pct_col = round(n_col / len(df) * 100, 2) if len(df) else 0.0
            narrative_parts.append(
                f"Semantic: column '{col}' — {n_col} duplicates ({pct_col}%)."
            )
            summary["semantic"] = {"duplicates": n_col, "pct": pct_col}
        else:
            narrative_parts.append("Semantic dedup skipped (missing dependencies).")

    elif mode == "hybrid":
        # --- Pass 1: lexical ---
        lex_dups, lex_groups, lex_warnings = _run_lexical(
            df, text_cols, cfg.get("lexical", {}),
        )
        all_dup_indices.update(df.index[list(lex_dups)].tolist())
        duplicate_groups.update(lex_groups)
        warnings.extend(lex_warnings)
        lex_n = len(lex_dups)
        lex_pct = round(lex_n / len(df) * 100, 2) if len(df) else 0.0
        narrative_parts.append(
            f"Hybrid pass 1 (lexical): {lex_n} duplicates ({lex_pct}%)."
        )
        summary["lexical"] = {"fuzzy_duplicates": lex_n, "pct": lex_pct}

        # --- Check skip threshold ---
        hyb_cfg = cfg.get("hybrid", {})
        skip_pct = hyb_cfg.get("skip_semantic_if_lexical_removed_pct", 100)
        if lex_pct >= skip_pct:
            narrative_parts.append(
                f"Skipping semantic pass (lexical removed {lex_pct}% >= threshold {skip_pct}%)."
            )
        else:
            # Build survivor df — preserve original index order for remapping
            survivors_df = df.drop(index=list(all_dup_indices))
            # Keep ordered mapping: survivor position → original index label
            survivor_orig_index = survivors_df.index.tolist()
            survivors_df = survivors_df.reset_index(drop=True)

            sem_dups, sem_groups, sem_warnings, sem_ran = _run_semantic(
                survivors_df, text_cols, cfg.get("semantic", {}), output_dir,
            )
            warnings.extend(sem_warnings)

            if sem_ran and sem_dups:
                # Remap survivor positions back to original index labels
                sem_original_idx = {survivor_orig_index[i] for i in sem_dups}
                all_dup_indices.update(sem_original_idx)

                # Remap group member positions to original index labels
                remapped_groups: dict[str, list[list[int]]] = {}
                for k, grp_list in sem_groups.items():
                    remapped = []
                    for grp in grp_list:
                        remapped.append([survivor_orig_index[i] for i in grp])
                    remapped_groups[f"{k}_semantic"] = remapped
                duplicate_groups.update(remapped_groups)

                sem_n = len(sem_dups)
                sem_pct = round(sem_n / len(survivors_df) * 100, 2) if len(survivors_df) else 0.0
                narrative_parts.append(
                    f"Hybrid pass 2 (semantic): {sem_n} additional duplicates "
                    f"({sem_pct}% of survivors)."
                )
                summary["semantic"] = {"duplicates": sem_n, "pct": sem_pct}
            elif sem_ran:
                narrative_parts.append("Hybrid pass 2 (semantic): 0 additional duplicates.")
            else:
                narrative_parts.append(
                    "Hybrid pass 2 skipped (semantic dependencies unavailable)."
                )
    else:
        warnings.append(f"Unknown dedup_mode '{mode}', falling back to lexical.")
        lex_dups, lex_groups, lex_warnings = _run_lexical(
            df, text_cols, cfg.get("lexical", {}),
        )
        all_dup_indices.update(df.index[list(lex_dups)].tolist())
        duplicate_groups.update(lex_groups)
        warnings.extend(lex_warnings)

    # ------------------------------------------------------------------
    # Save duplicate group mapping (including exact-dup info)
    # ------------------------------------------------------------------
    mapping_data: dict[str, Any] = {
        "dedup_mode": mode,
        "config": {
            k: v for k, v in cfg.items()
            if k in ("lexical", "semantic", "hybrid")
        },
        "exact_duplicates": exact_dups,
    }
    if duplicate_groups:
        mapping_data["columns"] = {
            col: {"n_groups": len(grps), "groups": grps}
            for col, grps in duplicate_groups.items()
        }
    mapping_path = output_dir / "duplicate_groups.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping_data, f, indent=2)
    artifacts.append(mapping_path)

    # ------------------------------------------------------------------
    # Apply deduplication
    # ------------------------------------------------------------------
    transformed_df = None
    if deduplicate and all_dup_indices:
        df_clean = df.drop(index=list(all_dup_indices)).reset_index(drop=True)
        transformed_df = df_clean

        n_removed = len(df) - len(df_clean)
        summary["rows_before"] = len(df)
        summary["rows_after"] = len(df_clean)
        summary["rows_removed"] = n_removed
        summary["dedup_applied"] = True
        narrative_parts.append(
            f"Deduplicated: {len(df)} -> {len(df_clean)} rows "
            f"({n_removed} removed, {round(n_removed / len(df) * 100, 1)}% reduction)."
        )

        dedup_path = output_dir / "deduplicated.parquet"
        df_clean.to_parquet(dedup_path, index=False)
        artifacts.append(dedup_path)
        logger.info(
            "Dedup complete (%s): %d -> %d rows. Saved to %s",
            mode, len(df), len(df_clean), dedup_path,
        )
    elif deduplicate:
        summary["dedup_applied"] = False
        narrative_parts.append("No duplicates to remove.")
    else:
        summary["dedup_applied"] = False
        narrative_parts.append("Deduplication disabled (report only).")

    if not narrative_parts:
        narrative_parts.append("No duplicates detected.")

    return ModuleResult(
        module_name="duplicates",
        status="success",
        summary=summary,
        narrative=" ".join(narrative_parts),
        artifacts=artifacts,
        warnings=warnings,
        transformed_df=transformed_df,
    )


# =========================================================================
# Lexical dedup (MinHash LSH)
# =========================================================================

def _run_lexical(
    df: pd.DataFrame,
    text_cols: list[str],
    cfg: dict[str, Any],
) -> tuple[set[int], dict[str, list[list[int]]], list[str]]:
    """Run MinHash LSH dedup.  Returns (dup_indices, groups_per_col, warnings)."""
    threshold = cfg.get("lsh_threshold", 0.85)
    num_perm = cfg.get("lsh_num_perm", 128)
    all_dups: set[int] = set()
    all_groups: dict[str, list[list[int]]] = {}
    warnings: list[str] = []

    for col in text_cols:
        try:
            dup_idx, groups = _lsh_duplicates(df, col, threshold, num_perm)
            all_dups.update(dup_idx)
            all_groups[col] = groups
            pct = round(len(dup_idx) / len(df) * 100, 2) if len(df) else 0.0
            if pct > 10:
                warnings.append(
                    f"Column '{col}' has {pct}% fuzzy duplicates (lexical)."
                )
        except ImportError:
            warnings.append(
                f"datasketch not installed — skipping lexical dedup for '{col}'."
            )
        except Exception as exc:
            warnings.append(f"Lexical dedup failed for '{col}': {exc}")

    return all_dups, all_groups, warnings


def _lsh_duplicates(
    df: pd.DataFrame, col: str, threshold: float, num_perm: int,
) -> tuple[set[int], list[list[int]]]:
    """Return (duplicate_positional_indices, duplicate_groups) via MinHash LSH.

    Rows where the text is blank/whitespace-only are excluded from hashing
    to avoid collapsing all empty strings into one duplicate group.
    """
    from datasketch import MinHash, MinHashLSH

    texts = df[col].fillna("").astype(str).tolist()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    duplicate_indices: set[int] = set()
    group_map: dict[int, list[int]] = {}

    for idx, text in enumerate(texts):
        stripped = text.strip()
        if len(stripped) < _MIN_TEXT_LEN:
            continue  # skip blank/empty — not eligible for dedup

        m = MinHash(num_perm=num_perm)
        for word in stripped.lower().split():
            m.update(word.encode("utf8"))

        candidates = lsh.query(m)
        if candidates:
            duplicate_indices.add(idx)
            representative = min(candidates)
            group_map.setdefault(representative, [representative]).append(idx)

        try:
            lsh.insert(idx, m)
        except ValueError:
            pass

    groups = [members for members in group_map.values() if len(members) > 1]
    return duplicate_indices, groups


# =========================================================================
# Semantic dedup (sentence-transformers + DBSCAN)
# =========================================================================

def _run_semantic(
    df: pd.DataFrame,
    text_cols: list[str],
    cfg: dict[str, Any],
    output_dir: Path,
) -> tuple[set[int], dict[str, list[list[int]]], list[str], bool]:
    """Run embedding-based dedup.

    Returns (dup_indices, groups_per_col, warnings, actually_ran).
    ``actually_ran`` is False when dependencies are missing.
    """
    warnings: list[str] = []

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import DBSCAN
    except ImportError:
        warnings.append(
            "sentence-transformers or scikit-learn not installed — "
            "skipping semantic dedup.  "
            "Install with: pip install sentence-transformers scikit-learn"
        )
        return set(), {}, warnings, False

    import torch

    model_name = cfg.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
    eps = cfg.get("eps", 0.15)
    min_samples = cfg.get("min_samples", 2)
    batch_size = cfg.get("batch_size", 64)
    device_cfg = cfg.get("device", "auto")
    max_cluster_size = cfg.get("max_cluster_size")
    split_eps_factor = cfg.get("split_eps_factor", 0.7)

    # Resolve device
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg
    logger.info("Semantic dedup: model=%s, eps=%s, device=%s", model_name, eps, device)

    # Guard model initialization
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as exc:
        warnings.append(f"Failed to load embedding model '{model_name}': {exc}")
        logger.error("Embedding model init failed: %s", exc, exc_info=True)
        return set(), {}, warnings, False

    all_dups: set[int] = set()
    all_groups: dict[str, list[list[int]]] = {}

    for col in text_cols:
        try:
            raw_texts = df[col].fillna("").astype(str).tolist()

            # Filter out blank texts — don't embed or cluster them
            eligible_mask = [len(t.strip()) >= _MIN_TEXT_LEN for t in raw_texts]
            eligible_idx = [i for i, ok in enumerate(eligible_mask) if ok]
            eligible_texts = [raw_texts[i] for i in eligible_idx]

            if len(eligible_texts) < 2:
                continue  # need at least 2 texts for DBSCAN

            # Encode
            logger.info("Encoding %d texts (batch_size=%d)...", len(eligible_texts), batch_size)
            embeddings = model.encode(
                eligible_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # DBSCAN
            logger.info("Running DBSCAN (eps=%s, min_samples=%s)...", eps, min_samples)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
            labels = dbscan.fit_predict(embeddings)

            # Recursive split of oversized clusters
            if max_cluster_size is not None:
                labels = _split_large_clusters(
                    embeddings, labels, max_cluster_size,
                    eps * split_eps_factor, min_samples,
                )

            # Convert clusters to duplicate groups (in eligible-local positions)
            local_dup_indices, local_groups = _clusters_to_dedup(labels)

            # Map local positions back to DataFrame positions
            for local_idx in local_dup_indices:
                all_dups.add(eligible_idx[local_idx])
            remapped_groups = []
            for grp in local_groups:
                remapped_groups.append([eligible_idx[i] for i in grp])
            all_groups[col] = remapped_groups

            pct = round(len(local_dup_indices) / len(df) * 100, 2) if len(df) else 0.0
            if pct > 10:
                warnings.append(
                    f"Column '{col}' has {pct}% semantic duplicates."
                )

            # Save embeddings for potential reuse
            emb_path = output_dir / f"embeddings_{col}.npy"
            np.save(emb_path, embeddings)
            logger.info("Saved embeddings to %s", emb_path)

        except Exception as exc:
            warnings.append(f"Semantic dedup failed for '{col}': {exc}")
            logger.error("Semantic dedup failed for '%s': %s", col, exc, exc_info=True)

    return all_dups, all_groups, warnings, True


def _split_large_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_size: int,
    tighter_eps: float,
    min_samples: int,
) -> np.ndarray:
    """Recursively split clusters that exceed max_size."""
    from sklearn.cluster import DBSCAN

    labels = labels.copy()
    unique, counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique, counts):
        if label == -1 or count <= max_size:
            continue
        mask = labels == label
        sub_emb = embeddings[mask]
        sub_db = DBSCAN(eps=tighter_eps, min_samples=min_samples, metric="cosine")
        sub_labels = sub_db.fit_predict(sub_emb)
        max_label = labels.max()
        sub_labels[sub_labels != -1] += max_label + 1
        labels[mask] = sub_labels

    return labels


def _clusters_to_dedup(labels: np.ndarray) -> tuple[set[int], list[list[int]]]:
    """Convert DBSCAN cluster labels to dedup indices.

    For each cluster (label != -1), keep the first member and flag the rest
    as duplicates.
    """
    from collections import defaultdict

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            clusters[label].append(idx)

    dup_indices: set[int] = set()
    groups: list[list[int]] = []

    for members in clusters.values():
        if len(members) > 1:
            groups.append(members)
            dup_indices.update(members[1:])

    return dup_indices, groups

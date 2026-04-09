"""Clustering module — sentence-transformer embeddings + DBSCAN.

Adds a ``cluster_label`` column to the DataFrame so the downstream
``subsample`` module can use ``strategy=diversity`` for cluster-proportional
sampling.

Adapted from KD ``data_collector/clusterer.py`` with these additions:
* Reuses pre-computed embeddings from the ``duplicates`` module when available.
* Same GPU policy as semantic dedup: requires GPU unless user forces CPU.
* Recursive splitting of oversized clusters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ahvs.data_analyst.models import ModuleInput, ModuleResult

logger = logging.getLogger(__name__)

# Default parameters (overridable via module params)
_DEFAULTS: dict[str, Any] = {
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "eps": 0.3,
    "min_samples": 2,
    "batch_size": 64,
    "device": "auto",
    "max_cluster_size": None,
    "split_eps_factor": 0.7,
}

# Minimum non-whitespace characters for a text to be eligible for clustering.
_MIN_TEXT_LEN = 1


def run(inp: ModuleInput) -> ModuleResult:
    """Cluster texts using sentence-transformer embeddings + DBSCAN."""
    df = inp.df
    params = inp.params
    output_dir = inp.output_dir / "cluster"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    warnings: list[str] = []
    artifacts: list[Path] = []

    # Resolve text columns
    text_cols = _resolve_text_cols(df, inp.input_cols)
    if not text_cols:
        df = df.copy()
        df["cluster_label"] = -1
        return ModuleResult(
            module_name="cluster",
            status="skipped",
            summary={"reason": "no_text_columns"},
            narrative="No text columns found for clustering.",
            transformed_df=df,
        )
    col = text_cols[0]

    # GPU check — skip if no GPU (unless user explicitly set device)
    user_forced_device = "device" in params
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False

    if not gpu_available and not user_forced_device:
        logger.warning(
            "GPU unavailable — skipping embedding clustering. "
            "Pass device=cpu in params to force CPU execution."
        )
        df = df.copy()
        df["cluster_label"] = -1
        return ModuleResult(
            module_name="cluster",
            status="skipped",
            summary={"reason": "no_gpu"},
            narrative="Clustering skipped: GPU unavailable. Use device=cpu to force.",
            warnings=["GPU unavailable — skipping clustering. "
                       "Subsample will fall back to stratified/random."],
            transformed_df=df,
        )

    # Check dependencies
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import DBSCAN
    except ImportError:
        df = df.copy()
        df["cluster_label"] = -1
        return ModuleResult(
            module_name="cluster",
            status="skipped",
            summary={"reason": "missing_deps"},
            narrative="Clustering skipped: sentence-transformers or scikit-learn not installed.",
            warnings=["Install with: pip install sentence-transformers scikit-learn"],
            transformed_df=df,
        )

    import torch

    # Resolve params
    model_name = params.get("embedding_model", _DEFAULTS["embedding_model"])
    eps = params.get("eps", _DEFAULTS["eps"])
    min_samples = params.get("min_samples", _DEFAULTS["min_samples"])
    batch_size = params.get("batch_size", _DEFAULTS["batch_size"])
    device_cfg = params.get("device", _DEFAULTS["device"])
    max_cluster_size = params.get("max_cluster_size", _DEFAULTS["max_cluster_size"])
    split_eps_factor = params.get("split_eps_factor", _DEFAULTS["split_eps_factor"])

    device = "cuda" if (device_cfg == "auto" and gpu_available) else (
        device_cfg if device_cfg != "auto" else "cpu"
    )
    logger.info("Clustering: model=%s, eps=%s, device=%s", model_name, eps, device)

    # ------------------------------------------------------------------
    # Try to reuse embeddings from dedup module
    # ------------------------------------------------------------------
    dedup_emb_path = inp.output_dir / "duplicates" / f"embeddings_{col}.npy"
    embeddings: np.ndarray | None = None
    reused_embeddings = False

    if dedup_emb_path.exists():
        try:
            saved = np.load(dedup_emb_path)
            if saved.shape[0] == len(df):
                embeddings = saved
                reused_embeddings = True
                logger.info(
                    "Reusing %d embeddings from dedup module (%s).",
                    len(embeddings), dedup_emb_path,
                )
            else:
                logger.info(
                    "Dedup embeddings shape mismatch (%d vs %d rows) — encoding fresh.",
                    saved.shape[0], len(df),
                )
        except Exception as exc:
            logger.warning("Failed to load dedup embeddings: %s — encoding fresh.", exc)

    # ------------------------------------------------------------------
    # Encode if needed
    # ------------------------------------------------------------------
    if embeddings is None:
        try:
            model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            df = df.copy()
            df["cluster_label"] = -1
            return ModuleResult(
                module_name="cluster",
                status="error",
                error_message=f"Failed to load model '{model_name}': {exc}",
                transformed_df=df,
            )

        raw_texts = df[col].fillna("").astype(str).tolist()
        # Filter blanks
        eligible_mask = [len(t.strip()) >= _MIN_TEXT_LEN for t in raw_texts]
        eligible_texts = [raw_texts[i] for i, ok in enumerate(eligible_mask) if ok]
        eligible_idx = [i for i, ok in enumerate(eligible_mask) if ok]

        if len(eligible_texts) < min_samples:
            df = df.copy()
            df["cluster_label"] = -1
            return ModuleResult(
                module_name="cluster",
                status="skipped",
                summary={"reason": "too_few_texts"},
                narrative=f"Too few eligible texts ({len(eligible_texts)}) for clustering.",
                transformed_df=df,
            )

        logger.info("Encoding %d texts (batch_size=%d)...", len(eligible_texts), batch_size)
        emb_eligible = model.encode(
            eligible_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Build full-size embedding array (zeros for blanks — they get noise label)
        embeddings = np.zeros((len(df), emb_eligible.shape[1]), dtype=np.float32)
        for local_i, orig_i in enumerate(eligible_idx):
            embeddings[orig_i] = emb_eligible[local_i]

        # Save embeddings
        emb_path = output_dir / f"embeddings_{col}.npy"
        np.save(emb_path, embeddings)
        artifacts.append(emb_path)
        logger.info("Saved embeddings to %s", emb_path)

    # ------------------------------------------------------------------
    # DBSCAN clustering
    # ------------------------------------------------------------------
    logger.info("Running DBSCAN (eps=%s, min_samples=%s)...", eps, min_samples)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = dbscan.fit_predict(embeddings)

    # Recursive split of oversized clusters
    if max_cluster_size is not None:
        labels = _split_large_clusters(
            embeddings, labels, max_cluster_size,
            eps * split_eps_factor, min_samples,
        )

    # Assign labels
    df = df.copy()
    df["cluster_label"] = labels

    # Stats
    unique_labels = np.unique(labels)
    n_clusters = int((unique_labels != -1).sum())
    n_noise = int((labels == -1).sum())
    cluster_sizes = []
    for lbl in unique_labels:
        if lbl != -1:
            cluster_sizes.append(int((labels == lbl).sum()))

    summary = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "avg_cluster_size": round(np.mean(cluster_sizes), 1) if cluster_sizes else 0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
        "eps": eps,
        "model": model_name,
        "device": device,
        "embeddings_reused": reused_embeddings,
    }

    narrative = (
        f"Clustered {len(df)} texts into {n_clusters} clusters "
        f"({n_noise} noise points). "
        f"Avg cluster size: {summary['avg_cluster_size']}."
    )

    logger.info("Clustering complete: %d clusters, %d noise.", n_clusters, n_noise)

    return ModuleResult(
        module_name="cluster",
        status="success",
        summary=summary,
        narrative=narrative,
        artifacts=artifacts,
        warnings=warnings,
        transformed_df=df,
    )


def _resolve_text_cols(df: pd.DataFrame, input_cols: list[str]) -> list[str]:
    """Return input columns that are text-like."""
    text_cols = []
    for c in input_cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            text_cols.append(c)
    return text_cols


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

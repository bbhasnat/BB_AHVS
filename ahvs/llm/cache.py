"""Content-addressable LLM response cache — eliminates redundant API calls.

Uses SHA-256 of the full API call parameters (model, messages, system,
max_tokens, temperature) as cache key. Stores responses in a per-project
SQLite database with WAL mode for concurrent read safety.

Design from: hackathon_knowledge_distillation/LLM_cost_optimization.md

Environment variables:
    LLM_CACHE_ENABLED       — "true" (default) or "false"
    LLM_CACHE_DIR           — cache directory (default: <repo>/.ahvs/.llm_cache)
    LLM_CACHE_TTL_HOURS     — entry TTL in hours (default: no expiry)
    LLM_CACHE_STORE_MESSAGES — "true" to store input messages (default: false, PII safety)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _is_cache_enabled() -> bool:
    return os.environ.get("LLM_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")


class LLMCache:
    """SQLite-backed LLM response cache with content-addressable keys."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._dir = Path(cache_dir) if cache_dir else Path(".llm_cache")
        self._db_path = self._dir / "responses.db"
        self._conn: sqlite3.Connection | None = None
        self._store_messages = os.environ.get(
            "LLM_CACHE_STORE_MESSAGES", "false"
        ).lower() in ("true", "1", "yes")
        self._ttl_hours: int | None = None
        ttl_env = os.environ.get("LLM_CACHE_TTL_HOURS")
        if ttl_env:
            try:
                self._ttl_hours = int(ttl_env)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Connection / schema
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self._dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key          TEXT PRIMARY KEY,
                model        TEXT NOT NULL,
                response     TEXT NOT NULL,
                tokens_in    INTEGER,
                tokens_out   INTEGER,
                created_at   TEXT NOT NULL,
                expires_at   TEXT,
                messages     TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_model ON cache(model)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)"
        )
        self._conn.commit()
        return self._conn

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(
        *,
        model: str,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        """SHA-256 of the full call parameters — any change produces a new key."""
        blob: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "system": system,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "json_mode": json_mode,
        }
        raw = json.dumps(blob, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Get / put
    # ------------------------------------------------------------------

    def get(self, key: str) -> dict[str, Any] | None:
        """Look up a cached response. Returns None on miss or expired entry."""
        conn = self._connect()
        row = conn.execute(
            "SELECT response, model, tokens_in, tokens_out, expires_at "
            "FROM cache WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None

        response, model, tokens_in, tokens_out, expires_at = row

        # Check TTL
        if expires_at:
            try:
                exp = datetime.fromisoformat(expires_at)
                if datetime.now(timezone.utc) > exp:
                    # Expired — treat as miss
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
                    return None
            except ValueError:
                pass

        return {
            "content": response,
            "model": model,
            "tokens_in": tokens_in or 0,
            "tokens_out": tokens_out or 0,
        }

    def put(
        self,
        key: str,
        *,
        content: str,
        model: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        messages: list[dict[str, str]] | None = None,
    ) -> None:
        """Store a validated response in the cache."""
        conn = self._connect()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        expires_at: str | None = None
        if self._ttl_hours:
            expires_at = (
                datetime.now(timezone.utc) + timedelta(hours=self._ttl_hours)
            ).isoformat(timespec="seconds")

        msg_json: str | None = None
        if self._store_messages and messages:
            msg_json = json.dumps(messages, ensure_ascii=False)

        conn.execute(
            """
            INSERT OR REPLACE INTO cache
                (key, model, response, tokens_in, tokens_out, created_at, expires_at, messages)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (key, model, content, tokens_in, tokens_out, now, expires_at, msg_json),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def evict_expired(self) -> int:
        """Remove expired entries. Returns count of deleted rows."""
        conn = self._connect()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cur = conn.execute(
            "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,),
        )
        conn.commit()
        return cur.rowcount

    def clear(self, *, model: str | None = None) -> int:
        """Clear cache entries. Optionally filter by model."""
        conn = self._connect()
        if model:
            cur = conn.execute("DELETE FROM cache WHERE model = ?", (model,))
        else:
            cur = conn.execute("DELETE FROM cache")
        conn.commit()
        return cur.rowcount

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        conn = self._connect()
        total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        models: dict[str, int] = {}
        for row in conn.execute(
            "SELECT model, COUNT(*) FROM cache GROUP BY model"
        ):
            models[row[0]] = row[1]

        size_mb = 0.0
        if self._db_path.exists():
            size_mb = self._db_path.stat().st_size / (1024 * 1024)

        oldest = conn.execute(
            "SELECT MIN(created_at) FROM cache"
        ).fetchone()[0]

        return {
            "enabled": True,
            "entries": total,
            "size_mb": round(size_mb, 2),
            "db_path": str(self._db_path),
            "oldest_entry": oldest,
            "models": models,
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class CachedClientWrapper:
    """Transparent caching wrapper around any LLM client with a .chat() method.

    Intercepts .chat() calls, checks the cache, and only forwards to the
    real client on cache miss. All other attributes are proxied to the
    underlying client.
    """

    def __init__(self, client: Any, cache: LLMCache) -> None:
        self._client = client
        self._cache = cache
        self._hits = 0
        self._misses = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        system: str | None = None,
        strip_thinking: bool = False,
    ) -> Any:
        from ahvs.llm.client import LLMResponse

        # Resolve model for cache key
        resolved_model = model or getattr(
            getattr(self._client, "config", None), "primary_model", "unknown"
        )

        key = LLMCache.make_key(
            model=resolved_model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )

        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            logger.info(
                "Cache HIT  key=%s...  model=%s  (saved 1 API call)",
                key[:12],
                cached["model"],
            )
            return LLMResponse(
                content=cached["content"],
                model=cached["model"],
                prompt_tokens=cached["tokens_in"],
                completion_tokens=cached["tokens_out"],
                finish_reason="stop",
            )

        # Cache miss — call the real client
        self._misses += 1
        logger.info("Cache MISS key=%s...  model=%s", key[:12], resolved_model)

        response = self._client.chat(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            system=system,
            strip_thinking=strip_thinking,
        )

        # Validate before caching: only cache successful, non-empty responses
        if (
            response.content
            and response.content.strip()
            and response.finish_reason != "length"
        ):
            self._cache.put(
                key,
                content=response.content,
                model=response.model,
                tokens_in=response.prompt_tokens,
                tokens_out=response.completion_tokens,
                messages=messages if system is None else (
                    [{"role": "system", "content": system}] + messages
                ),
            )
        else:
            logger.info(
                "Cache SKIP key=%s...  (finish_reason=%s, empty=%s)",
                key[:12],
                response.finish_reason,
                not bool(response.content and response.content.strip()),
            )

        return response

    def log_session_stats(self) -> None:
        """Log cache hit/miss stats for this session."""
        total = self._hits + self._misses
        if total == 0:
            return
        hit_rate = (self._hits / total) * 100
        logger.info(
            "Cache session: %d calls, %d hits, %d misses (%.0f%% hit rate)",
            total,
            self._hits,
            self._misses,
            hit_rate,
        )

    # Proxy everything else to the underlying client
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

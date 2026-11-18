"""Append-only, hash-chained trial ledger (JSONL).

Every evaluation a search makes is one row, including repeats of a variant
that was already scored (``cached: true``; the evaluator is not called again)
and evaluations that raised (``error`` set, no score). Rows are never edited
or removed, so the counts are exact:

* ``n_trials``: every row, i.e. every proposal, duplicates included.
* ``n_evaluator_calls``: rows where the evaluator actually ran.
* ``n_unique``: distinct variants tried (per split and overall).

Each row carries ``prev`` (the previous row's hash) and ``hash`` (SHA-256
of the row's canonical JSON without ``hash``), starting from 64 zeros, so
:meth:`TrialLedger.verify` (or :func:`verify_ledger_file`) detects any edit,
deletion, reordering or truncation in the middle of a file. It is tamper
*evidence* for an honest audit trail, not protection against someone who
rewrites the whole file and every later hash. Removing rows from the *end*
cannot be detected from the file alone: keep the ``head`` hash a summary
reports and pass it as ``expected_head`` to check later. Use one writer
(process) per ledger file.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from .evaluator import SPLITS, TRAIN, EvaluationResult

GENESIS = "0" * 64


class LedgerIntegrityError(RuntimeError):
    """The ledger file's hash chain does not verify."""


def variant_key(variant: Mapping[str, Any]) -> str:
    """Canonical string for a variant (key order does not matter)."""
    return json.dumps(dict(variant), sort_keys=True, separators=(",", ":"), default=str)


def _canonical(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def _row_hash(row: Mapping[str, Any]) -> str:
    body = {k: v for k, v in row.items() if k != "hash"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def _check_rows(rows: Iterable[Mapping[str, Any]]) -> Optional[str]:
    prev = GENESIS
    for i, row in enumerate(rows):
        if row.get("seq") != i:
            return f"row {i}: seq is {row.get('seq')!r}, expected {i}"
        if row.get("prev") != prev:
            return f"row {i}: prev hash does not match row {i - 1}"
        if row.get("hash") != _row_hash(row):
            return f"row {i}: hash mismatch (row was modified)"
        prev = row["hash"]
    return None


def verify_ledger_file(path: Union[str, Path], expected_head: Optional[str] = None) -> Dict[str, Any]:
    """Check a ledger file: ``{"ok", "rows", "error", "head"}``.

    With ``expected_head`` (a hash recorded earlier), also fail if that row
    is no longer in the chain, which catches rows removed from the end.
    """
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return {"ok": True, "rows": 0, "error": None, "head": GENESIS}
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                torn = not line.endswith("\n")
                what = "incomplete last line (interrupted write?)" if torn else "not valid JSON"
                return {"ok": False, "rows": len(rows), "error": f"line {lineno}: {what}",
                        "head": rows[-1]["hash"] if rows else GENESIS}
    error = _check_rows(rows)
    if error is None and expected_head is not None and expected_head != GENESIS:
        if not any(r.get("hash") == expected_head for r in rows):
            error = f"expected head {expected_head[:12]}... is not in the chain (rows removed?)"
    return {"ok": error is None, "rows": len(rows), "error": error,
            "head": rows[-1].get("hash") if rows else GENESIS}


class TrialLedger:
    """Thread-safe append-only trial ledger, in memory and optionally on disk.

    Opening an existing file loads and verifies it (``LedgerIntegrityError``
    if the chain is broken) and appends after its last row, so counts carry
    across processes.
    """

    def __init__(self, path: Optional[Union[str, Path]] = None, *, fsync: bool = False):
        self.path = Path(path) if path is not None else None
        self.fsync = fsync
        self._lock = threading.Lock()
        self._rows: List[Dict[str, Any]] = []
        # (split, evaluator, variant key) -> first successful result.
        self._cache: Dict[tuple, Dict[str, Any]] = {}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self._load()

    def _load(self) -> None:
        assert self.path is not None
        report = verify_ledger_file(self.path)
        if not report["ok"]:
            raise LedgerIntegrityError(f"{self.path.name}: {report['error']}")
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    self._index(json.loads(line))

    def _index(self, row: Dict[str, Any]) -> None:
        self._rows.append(row)
        if row.get("result") is not None and not row.get("error"):
            self._cache.setdefault((row["split"], row.get("evaluator"), row["key"]), row["result"])

    # ----- writing -----

    def lookup(
        self, variant: Mapping[str, Any], split: str = TRAIN, evaluator: Optional[str] = None
    ) -> Optional[EvaluationResult]:
        """The first successful result for this variant, split and evaluator name."""
        with self._lock:
            hit = self._cache.get((split, evaluator, variant_key(variant)))
        if hit is None:
            return None
        return EvaluationResult(score=hit["score"], n=hit.get("n"), metadata=dict(hit.get("metadata") or {}))

    def record(
        self,
        *,
        source: str,
        variant: Mapping[str, Any],
        split: str = TRAIN,
        result: Optional[EvaluationResult] = None,
        error: Optional[str] = None,
        cached: bool = False,
        round_number: Optional[int] = None,
        evaluator: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append one row and return it (as stored)."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
        if (result is None) == (error is None):
            raise ValueError("pass exactly one of result or error")
        with self._lock:
            seq = len(self._rows)
            row: Dict[str, Any] = {
                "seq": seq,
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "source": source,
                "round": round_number,
                "split": split,
                "evaluator": evaluator,
                "key": variant_key(variant),
                "variant": dict(variant),
                "result": result.to_dict() if result is not None else None,
                "error": error,
                "cached": bool(cached),
                "prev": self._rows[-1]["hash"] if self._rows else GENESIS,
            }
            # Round-trip so the in-memory row is exactly what is on disk.
            row = json.loads(_canonical(row))
            row["hash"] = _row_hash(row)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(_canonical(row) + "\n")
                    if self.fsync:
                        fh.flush()
                        os.fsync(fh.fileno())
            self._index(row)
            return row

    # ----- reading -----

    def rows(self, offset: int = 0, limit: Optional[int] = None, split: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            items = [r for r in self._rows if split is None or r["split"] == split]
        end = None if limit is None else offset + limit
        return [dict(r) for r in items[offset:end]]

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def head(self) -> str:
        with self._lock:
            return self._rows[-1]["hash"] if self._rows else GENESIS

    def verify(self, expected_head: Optional[str] = None) -> Dict[str, Any]:
        """Verify the in-memory chain, and the file too if there is one.

        Holds the lock throughout so a concurrent append cannot make memory
        and file disagree mid-check.
        """
        with self._lock:
            error = _check_rows(self._rows)
            n = len(self._rows)
            head = self._rows[-1]["hash"] if self._rows else GENESIS
            if error is None and expected_head not in (None, GENESIS):
                if not any(r["hash"] == expected_head for r in self._rows):
                    error = f"expected head {expected_head[:12]}... is not in the chain"
            report: Dict[str, Any] = {"ok": error is None, "rows": n, "error": error}
            if self.path is not None:
                disk = verify_ledger_file(self.path)
                report["file"] = disk
                if error is None and (not disk["ok"] or disk["rows"] != n or disk["head"] != head):
                    report["ok"] = False
                    report["error"] = disk["error"] or f"file has {disk['rows']} rows, memory has {n}"
        return report

    def counts(self) -> Dict[str, Any]:
        with self._lock:
            rows = list(self._rows)
        by_split: Dict[str, Dict[str, int]] = {}
        for split in SPLITS:
            sel = [r for r in rows if r["split"] == split]
            by_split[split] = {
                "trials": len(sel),
                "evaluator_calls": sum(1 for r in sel if not r["cached"]),
                "cached": sum(1 for r in sel if r["cached"]),
                "errors": sum(1 for r in sel if r["error"]),
                "unique_variants": len({r["key"] for r in sel}),
            }
        return {
            "n_trials": len(rows),
            "n_evaluator_calls": sum(1 for r in rows if not r["cached"]),
            "n_unique": len({r["key"] for r in rows}),
            "by_split": by_split,
        }

    @property
    def n_trials(self) -> int:
        return len(self._rows)

    @property
    def n_evaluator_calls(self) -> int:
        return self.counts()["n_evaluator_calls"]

    @property
    def n_unique(self) -> int:
        return self.counts()["n_unique"]

    def _scored(self, split: str, min_n: int) -> List[Dict[str, Any]]:
        with self._lock:
            rows = list(self._rows)
        out: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            res = r["result"]
            if r["split"] != split or res is None or r["error"]:
                continue
            if min_n and (res.get("n") is None or res["n"] < min_n):
                continue
            out.setdefault(r["key"], r)  # first scoring row per variant
        return list(out.values())

    def best(self, split: str = TRAIN, min_n: int = 0) -> Optional[Dict[str, Any]]:
        """Highest-scoring row on ``split`` (ties: the earliest)."""
        top = self.top(1, split=split, min_n=min_n)
        return top[0] if top else None

    def top(self, k: int = 5, split: str = TRAIN, min_n: int = 0) -> List[Dict[str, Any]]:
        """Top ``k`` distinct variants by score on ``split``."""
        scored = self._scored(split, min_n)
        scored.sort(key=lambda r: (-r["result"]["score"], r["seq"]))
        return [dict(r) for r in scored[: max(0, k)]]

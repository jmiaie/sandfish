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

Two optional additions, both inert unless used:

* **Event rows** (``record_event``) sit in the same chain but are not trials:
  they never count towards N and never appear in ``rows()``. They log the
  holdout guard (``lock_holdout``, ``holdout_access``) and diagnostics such as
  walk-forward runs. A ledger without events is byte-for-byte what it was.
* **Content head** (``content_head()``): a SHA-256 chain over the rows with
  the timestamp, run id and chain hashes left out (and swarm agent ids
  normalised to ``agent_<n>``), so two identical, deterministic runs give
  identical content heads even though their ``head`` hashes differ.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

from .evaluator import SPLITS, TEST, TRAIN, EvaluationResult

GENESIS = "0" * 64

# Holdout guard states: none (no guard), locked, open (validate_holdout is
# running), consumed (validate_holdout has run).
GUARD_NONE, GUARD_LOCKED, GUARD_OPEN, GUARD_CONSUMED = "none", "locked", "open", "consumed"


class LedgerIntegrityError(RuntimeError):
    """The ledger file's hash chain does not verify."""


class HoldoutLockedError(RuntimeError):
    """The held-out test split is locked: only ``validate_holdout`` may score it."""


def variant_key(variant: Mapping[str, Any]) -> str:
    """Canonical string for a variant (key order does not matter)."""
    return json.dumps(dict(variant), sort_keys=True, separators=(",", ":"), default=str)


def _canonical(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


_AGENT_RE = re.compile(r"^.*_agent_(\d+)$")


def _content_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The part of a row that is the same across identical runs."""
    if "event" in row:
        return {"event": row["event"], "data": _normalise(row.get("data"))}
    source = row.get("source")
    if isinstance(source, str):
        m = _AGENT_RE.match(source)
        if m:
            source = f"agent_{m.group(1)}"
    return {
        "source": source, "round": row.get("round"), "split": row.get("split"),
        "evaluator": row.get("evaluator"), "key": row.get("key"), "variant": row.get("variant"),
        "result": row.get("result"), "error": row.get("error"), "cached": row.get("cached"),
    }


def _normalise(obj: Any) -> Any:
    """Strip run-specific agent id prefixes inside logged event data."""
    if isinstance(obj, dict):
        return {k: _normalise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalise(v) for v in obj]
    if isinstance(obj, str):
        m = _AGENT_RE.match(obj)
        return f"agent_{m.group(1)}" if m else obj
    return obj


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
        self._rows: List[Dict[str, Any]] = []    # every row, trials and events (the chain)
        self._trials: List[Dict[str, Any]] = []  # trial rows only
        # (split, evaluator, variant key) -> first successful result.
        self._cache: Dict[tuple, Dict[str, Any]] = {}
        self._tls = threading.local()
        self._guard: Dict[str, Any] = {
            "state": GUARD_NONE, "locked_seq": None, "test_rows_before_lock": None,
            "denied_attempts": 0, "validations": 0,
        }
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
        if self._guard["state"] == GUARD_OPEN:
            # A validate_holdout died part-way: the test split may have been seen.
            self._guard["state"] = GUARD_CONSUMED

    def _index(self, row: Dict[str, Any]) -> None:
        self._rows.append(row)
        if "event" in row:
            self._apply_event(row)
            return
        self._trials.append(row)
        if row.get("result") is not None and not row.get("error"):
            self._cache.setdefault((row["split"], row.get("evaluator"), row["key"]), row["result"])

    def _apply_event(self, row: Dict[str, Any]) -> None:
        g, name, data = self._guard, row["event"], row.get("data") or {}
        if name == "holdout_locked":
            g.update(state=GUARD_LOCKED, locked_seq=row["seq"],
                     test_rows_before_lock=data.get("test_rows_before_lock"))
        elif name in ("holdout_unlocked", "holdout_reopened"):
            g["state"] = GUARD_OPEN
        elif name == "holdout_consumed":
            g.update(state=GUARD_CONSUMED, validations=g["validations"] + 1)
        elif name == "holdout_denied":
            g["denied_attempts"] += 1

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

    def _append(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Append one row (caller holds the lock) and return it as stored."""
        body["seq"] = len(self._rows)
        body["prev"] = self._rows[-1]["hash"] if self._rows else GENESIS
        # Round-trip so the in-memory row is exactly what is on disk.
        row = json.loads(_canonical(body))
        row["hash"] = _row_hash(row)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(_canonical(row) + "\n")
                if self.fsync:
                    fh.flush()
                    os.fsync(fh.fileno())
        self._index(row)
        return row

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
        if split == TEST and self._guard["state"] != GUARD_NONE and not self._holdout_is_open():
            raise HoldoutLockedError("the holdout (test) split is locked; use validate_holdout")
        with self._lock:
            return self._append({
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
            })

    def record_event(self, name: str, data: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Append a non-trial event row to the chain (never counted in N)."""
        if not name or not isinstance(name, str):
            raise ValueError("event name must be a non-empty string")
        with self._lock:
            return self._append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": name,
                "data": dict(data or {}),
            })

    # ----- holdout guard -----

    def _holdout_is_open(self) -> bool:
        return bool(getattr(self._tls, "open", False))

    def lock_holdout(self, reason: Optional[str] = None) -> Dict[str, Any]:
        """Lock the test split: from now on only :func:`validate_holdout`
        (via :meth:`holdout_access`) may score it. Logged in the chain.

        Locking is idempotent. If test rows were already recorded, the lock
        still happens but the log says so (``test_rows_before_lock``) and
        :meth:`holdout_status` will not call the holdout clean.
        """
        with self._lock:
            state = self._guard["state"]
            if state in (GUARD_LOCKED, GUARD_OPEN):
                return self._status_unlocked()
            if state == GUARD_CONSUMED:
                raise HoldoutLockedError("holdout was already consumed by validate_holdout")
            prior = sum(1 for r in self._trials if r["split"] == TEST)
            self._append({
                "ts": datetime.now(timezone.utc).isoformat(), "event": "holdout_locked",
                "data": {"test_rows_before_lock": prior, "reason": reason},
            })
            return self._status_unlocked()

    def check_holdout(self, split: str, source: str, variant: Mapping[str, Any]) -> None:
        """Raise :class:`HoldoutLockedError` (and log the attempt) if ``split``
        is the locked test split. Called before an evaluator is invoked."""
        if split != TEST or self._guard["state"] == GUARD_NONE or self._holdout_is_open():
            return
        self.record_event("holdout_denied", {"source": source, "key": variant_key(variant)})
        raise HoldoutLockedError(
            "the holdout (test) split is locked; only validate_holdout may evaluate it"
        )

    @contextmanager
    def holdout_access(
        self, finalists: Sequence[Mapping[str, Any]] = (), *, allow_repeat: bool = False
    ) -> Iterator[None]:
        """Open the locked test split for the calling thread (used by
        :func:`validate_holdout`). A no-op for an unguarded ledger. Logs
        ``holdout_unlocked`` / ``holdout_consumed`` events with the finalists.

        A second use raises unless ``allow_repeat=True``, which logs
        ``holdout_reopened``, so re-selecting after peeking stays visible.
        """
        with self._lock:
            state = self._guard["state"]
            if state == GUARD_NONE:
                guarded = False
            else:
                guarded = True
                if state == GUARD_OPEN:
                    raise HoldoutLockedError("validate_holdout is already running on this ledger")
                keys = [variant_key(v) for v in finalists]
                if state == GUARD_CONSUMED:
                    if not allow_repeat:
                        raise HoldoutLockedError(
                            "the holdout was already consumed; pass allow_repeat=True to reopen "
                            "it (the reopening is logged)")
                    name = "holdout_reopened"
                else:
                    name = "holdout_unlocked"
                self._append({"ts": datetime.now(timezone.utc).isoformat(), "event": name,
                              "data": {"finalists": keys}})
        if not guarded:
            yield
            return
        self._tls.open = True
        try:
            yield
        finally:
            self._tls.open = False
            with self._lock:
                n = sum(1 for r in self._trials if r["split"] == TEST)
                self._append({"ts": datetime.now(timezone.utc).isoformat(),
                              "event": "holdout_consumed", "data": {"test_rows": n}})

    def _status_unlocked(self) -> Dict[str, Any]:
        g = dict(self._guard)
        g["clean"] = (
            g["state"] != GUARD_NONE and g["test_rows_before_lock"] == 0
        )
        return g

    def holdout_status(self) -> Dict[str, Any]:
        """Guard state: ``state`` (none/locked/open/consumed), where it was
        locked, test rows seen before the lock, denied attempts, validations,
        and ``clean`` (locked before any test row was recorded)."""
        with self._lock:
            return self._status_unlocked()

    # ----- reading -----

    def rows(self, offset: int = 0, limit: Optional[int] = None, split: Optional[str] = None) -> List[Dict[str, Any]]:
        """Trial rows (oldest first). Event rows are not trials: see :meth:`events`."""
        with self._lock:
            items = [r for r in self._trials if split is None or r["split"] == split]
        end = None if limit is None else offset + limit
        return [dict(r) for r in items[offset:end]]

    def events(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Event rows (oldest first), optionally only those called ``name``."""
        with self._lock:
            return [dict(r) for r in self._rows if "event" in r and (name is None or r["event"] == name)]

    def __len__(self) -> int:
        return len(self._trials)

    @property
    def head(self) -> str:
        with self._lock:
            return self._rows[-1]["hash"] if self._rows else GENESIS

    def content_head(self) -> str:
        """Deterministic head: a SHA-256 chain over each row's content only
        (no timestamp, run id or chain hash; swarm agent ids as ``agent_<n>``).

        Two identical runs with a deterministic evaluator and seed have equal
        content heads; any change to a variant, score, error, order or logged
        event changes it. It does not replace :attr:`head` and :meth:`verify`.
        """
        with self._lock:
            rows = list(self._rows)
        h = GENESIS
        for row in rows:
            h = hashlib.sha256((h + _canonical(_content_row(row))).encode("utf-8")).hexdigest()
        return h

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
            rows = list(self._trials)
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
            "n_not_scorable": sum(1 for r in rows if _unscorable(r)),
            "not_scorable_by_split": {
                sp: sum(1 for r in rows if r["split"] == sp and _unscorable(r)) for sp in SPLITS},
            "by_split": by_split,
        }

    @property
    def n_trials(self) -> int:
        return len(self._trials)

    @property
    def n_evaluator_calls(self) -> int:
        return self.counts()["n_evaluator_calls"]

    @property
    def n_unique(self) -> int:
        return self.counts()["n_unique"]

    def _scored(self, split: str, min_n: int) -> List[Dict[str, Any]]:
        with self._lock:
            rows = list(self._trials)
        out: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            res = r["result"]
            if r["split"] != split or res is None or r["error"] or res.get("score") is None:
                continue  # errors and not-scorable results are counted but never ranked
            if min_n and (res.get("n") is None or res["n"] < min_n):
                continue
            out.setdefault(r["key"], r)  # first scoring row per variant
        return list(out.values())

    def best(self, split: str = TRAIN, min_n: int = 0) -> Optional[Dict[str, Any]]:
        """Highest-scoring row on ``split`` (ties: the earliest). Not-scorable
        results are never the best."""
        top = self.top(1, split=split, min_n=min_n)
        return top[0] if top else None

    def top(self, k: int = 5, split: str = TRAIN, min_n: int = 0) -> List[Dict[str, Any]]:
        """Top ``k`` distinct variants by score on ``split`` (scorable only)."""
        scored = self._scored(split, min_n)
        scored.sort(key=lambda r: (-r["result"]["score"], r["seq"]))
        return [dict(r) for r in scored[: max(0, k)]]


def _unscorable(row: Mapping[str, Any]) -> bool:
    res = row.get("result")
    return res is not None and not row.get("error") and res.get("score") is None

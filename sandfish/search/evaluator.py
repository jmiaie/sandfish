"""External evaluators: the opt-in plugin interface for parameter search.

SandFish does not score anything itself. An *evaluator* is caller-supplied
code that takes one proposed variant (a dict of parameter values) and a data
split name (``"train"`` or ``"test"``) and returns a score plus metadata.
Whatever data the evaluator reads, and how it keeps the test split unseen, is
entirely up to the evaluator.

Two ways to provide one:

* In Python, pass any object with ``evaluate(variant, split)`` or a plain
  function ``fn(variant, split)`` to :func:`as_evaluator` (the search helpers
  call it for you).
* For the MCP server and other by-name lookups, register a factory with
  :func:`register_evaluator`, or ship it in an installed package under the
  ``sandfish.evaluators`` entry-point group::

      [project.entry-points."sandfish.evaluators"]
      my_eval = "my_pkg.evaluators:make_evaluator"   # factory(**options) -> evaluator

Only registered names and installed entry points can be loaded by name;
nothing imports arbitrary modules from user input.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Union, runtime_checkable

TRAIN = "train"
TEST = "test"
SPLITS = (TRAIN, TEST)

ENTRY_POINT_GROUP = "sandfish.evaluators"


@dataclass(frozen=True)
class EvaluationResult:
    """One evaluation. ``n`` is the sample size behind the score, if any
    (e.g. number of observations); searches can require ``n >= min_n``.

    ``score=None`` means *not scorable*: the evaluator ran but the sample was
    too small to give a meaningful score (see :meth:`not_scorable`). Such a
    result is still a trial (it is recorded and counted in N) but it is never
    ranked: ``best()``/``top()`` and the researcher agents skip it. Use this
    instead of a fake ``0.0``, which would rank as a real score.
    """

    score: Optional[float]
    n: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def scorable(self) -> bool:
        return self.score is not None

    @classmethod
    def not_scorable(
        cls, n: Optional[int] = None, reason: Optional[str] = None, **metadata: Any
    ) -> "EvaluationResult":
        """A recorded-but-unranked result (e.g. too few trades to score)."""
        meta = dict(metadata)
        if reason is not None:
            meta["not_scorable_reason"] = reason
        return cls(score=None, n=n, metadata=meta)

    def to_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "n": self.n, "metadata": dict(self.metadata)}

    @classmethod
    def coerce(cls, value: Any) -> "EvaluationResult":
        """Accept an EvaluationResult, or a mapping with a numeric ``score``
        (an optional ``n`` and ``metadata``; any other keys go into metadata).

        A mapping may give ``"score": None`` or ``"scorable": False`` for a
        not-scorable result (see the class docstring)."""
        if isinstance(value, EvaluationResult):
            result = value
        elif isinstance(value, Mapping):
            unscorable = value.get("scorable") is False
            if "score" not in value and not unscorable:
                raise ValueError("evaluator result must include a 'score'")
            extra = {k: v for k, v in value.items() if k not in ("score", "n", "metadata", "scorable")}
            meta = dict(value.get("metadata") or {})
            meta.update(extra)
            result = cls(score=None if unscorable else value["score"], n=value.get("n"), metadata=meta)
        else:
            raise ValueError(
                f"evaluator must return EvaluationResult or a mapping, not {type(value).__name__}"
            )
        if result.score is not None:
            if isinstance(result.score, bool) or not isinstance(result.score, (int, float)):
                raise ValueError("score must be a number or None (not scorable)")
            if not math.isfinite(float(result.score)):
                raise ValueError("score must be finite")
        if result.n is not None and (isinstance(result.n, bool) or not isinstance(result.n, int)):
            raise ValueError("n must be an integer or None")
        # Metadata goes into the ledger verbatim: it must be JSON-serialisable
        # with string keys and finite numbers (unknown objects become strings).
        try:
            meta = json.loads(json.dumps(dict(result.metadata), default=str, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metadata is not JSON-serialisable: {exc}") from exc
        return cls(
            score=None if result.score is None else float(result.score), n=result.n, metadata=meta
        )


@runtime_checkable
class Evaluator(Protocol):
    """Anything with ``evaluate(variant, split)``. ``name`` is optional."""

    def evaluate(
        self, variant: Mapping[str, Any], split: str
    ) -> Union[EvaluationResult, Mapping[str, Any]]: ...


class FunctionEvaluator:
    """Adapts ``fn(variant, split) -> result`` to the Evaluator protocol."""

    def __init__(self, fn: Callable[[Mapping[str, Any], str], Any], name: Optional[str] = None):
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "function")

    def evaluate(self, variant: Mapping[str, Any], split: str) -> Any:
        return self._fn(variant, split)


def as_evaluator(obj: Any) -> Evaluator:
    """Return ``obj`` if it has ``evaluate``; wrap a plain callable."""
    if callable(getattr(obj, "evaluate", None)):
        return obj
    if callable(obj):
        return FunctionEvaluator(obj)
    raise TypeError("evaluator must have evaluate(variant, split) or be a callable(variant, split)")


def evaluator_name(evaluator: Any) -> str:
    return str(getattr(evaluator, "name", None) or type(evaluator).__name__)


def run_evaluation(evaluator: Evaluator, variant: Mapping[str, Any], split: str) -> EvaluationResult:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    return EvaluationResult.coerce(evaluator.evaluate(dict(variant), split))


# ----- registry -----

EvaluatorFactory = Callable[..., Any]
_REGISTRY: Dict[str, EvaluatorFactory] = {}


def register_evaluator(name: str, factory: EvaluatorFactory, *, replace: bool = False) -> None:
    """Register ``factory(**options) -> evaluator`` under ``name``."""
    if not name or not isinstance(name, str):
        raise ValueError("evaluator name must be a non-empty string")
    if name in _REGISTRY and not replace:
        raise ValueError(f"evaluator {name!r} is already registered")
    if not callable(factory):
        raise TypeError("factory must be callable")
    _REGISTRY[name] = factory


def unregister_evaluator(name: str) -> None:
    _REGISTRY.pop(name, None)


def _entry_points() -> Dict[str, Any]:
    from importlib.metadata import entry_points

    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # pragma: no cover - broken environment metadata
        return {}
    return {ep.name: ep for ep in eps}


def available_evaluators() -> List[str]:
    """Registered names plus installed ``sandfish.evaluators`` entry points."""
    return sorted(set(_REGISTRY) | set(_entry_points()))


def get_evaluator(name: str, **options: Any) -> Evaluator:
    """Build a registered (or entry-point) evaluator by name.

    ``options`` are passed to the factory as keyword arguments. Over MCP they
    come from the client, so treat them as untrusted input in your factory
    (e.g. don't accept arbitrary file paths or URLs from them).
    """
    factory = _REGISTRY.get(name)
    if factory is None:
        ep = _entry_points().get(name)
        if ep is None:
            raise KeyError(f"unknown evaluator {name!r}; available: {available_evaluators()}")
        factory = ep.load()
    evaluator = as_evaluator(factory(**options))
    if not getattr(evaluator, "name", None):
        try:
            evaluator.name = name  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - slotted/frozen evaluators
            pass
    return evaluator


# ----- toy evaluator (tests and docs) -----


class ToyQuadraticEvaluator:
    """Synthetic, data-free evaluator for tests and examples.

    ``score = -sum((variant[k] - target[k])**2) + noise``, where the noise is a
    deterministic function of (variant, split, seed). Train and test noise
    differ, so the best variant on train is not always best on test: a small,
    honest picture of why picked variants must be re-checked on held-out data.
    """

    name = "toy"
    DEFAULT_SPACE: Dict[str, List[Any]] = {
        "x": [-3, -2, -1, 0, 1, 2, 3],
        "y": [-3, -2, -1, 0, 1, 2, 3],
    }
    DEFAULT_TARGET: Dict[str, float] = {"x": 1.0, "y": -2.0}

    def __init__(
        self,
        target: Optional[Mapping[str, float]] = None,
        noise: float = 0.5,
        seed: int = 0,
        n: int = 100,
    ):
        self.target = {k: float(v) for k, v in (target or self.DEFAULT_TARGET).items()}
        if noise < 0:
            raise ValueError("noise must be >= 0")
        self.noise = float(noise)
        self.seed = int(seed)
        self.n = int(n)
        self.calls = 0

    def _noise(self, variant: Mapping[str, Any], split: str) -> float:
        if not self.noise:
            return 0.0
        key = json.dumps([self.seed, split, sorted(variant.items())], default=str)
        digest = hashlib.sha256(key.encode()).hexdigest()
        return random.Random(int(digest[:16], 16)).gauss(0.0, self.noise)

    def evaluate(self, variant: Mapping[str, Any], split: str) -> EvaluationResult:
        missing = sorted(set(self.target) - set(variant))
        if missing:
            raise ValueError(f"variant is missing {missing}")
        self.calls += 1
        dist = sum((float(variant[k]) - t) ** 2 for k, t in self.target.items())
        return EvaluationResult(
            score=round(-dist + self._noise(variant, split), 6),
            n=self.n,
            metadata={"distance2": dist, "split": split},
        )


register_evaluator("toy", ToyQuadraticEvaluator)

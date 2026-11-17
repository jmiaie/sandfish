"""Discrete search spaces: ``{"param": [value, ...], ...}``."""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Iterator, List, Mapping

MAX_PARAMS = 32
MAX_VALUES_PER_PARAM = 1000

_SCALARS = (str, int, float, bool, type(None))


def validate_space(space: Mapping[str, Any]) -> Dict[str, List[Any]]:
    """Return a normalised copy (lists, insertion order kept) or raise ValueError.

    Values must be JSON scalars (str, int, float, bool, None) so variants are
    hashable, loggable and reproducible.
    """
    if not isinstance(space, Mapping) or not space:
        raise ValueError("space must be a non-empty mapping of parameter -> list of values")
    if len(space) > MAX_PARAMS:
        raise ValueError(f"space has {len(space)} parameters; the limit is {MAX_PARAMS}")
    out: Dict[str, List[Any]] = {}
    for key, values in space.items():
        if not isinstance(key, str) or not key:
            raise ValueError("parameter names must be non-empty strings")
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
            raise ValueError(f"space[{key!r}] must be a list of values")
        if not values:
            raise ValueError(f"space[{key!r}] has no values")
        if len(values) > MAX_VALUES_PER_PARAM:
            raise ValueError(f"space[{key!r}] has more than {MAX_VALUES_PER_PARAM} values")
        for v in values:
            if not isinstance(v, _SCALARS):
                raise ValueError(f"space[{key!r}] values must be str/int/float/bool/None")
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"space[{key!r}] contains a non-finite number")
        out[key] = list(values)
    return out


def grid_size(space: Mapping[str, List[Any]]) -> int:
    """Number of variants in the full grid."""
    return math.prod(len(v) for v in space.values())


def iter_grid(space: Mapping[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    """Every variant, in a fixed order (sorted parameter names, listed value order)."""
    keys = sorted(space)
    for combo in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, combo))

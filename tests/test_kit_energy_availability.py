from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from sandfish.kits.energy.availability import (
    UNIFORM_MIN_LAG_HOURS,
    LeakageError,
    assert_no_leak,
    earliest_valid_lag,
    is_lag_valid,
    last_usable_hour,
    min_lag_hours,
    valid_variant,
)
from sandfish.search import iter_grid

LA = ZoneInfo("America/Los_Angeles")


def loc(y, m, d, h):
    return datetime(y, m, d, h, tzinfo=LA)


def test_last_usable_hour():
    assert last_usable_hour(date(2024, 6, 15)) == loc(2024, 6, 14, 9)


def test_lag_midnight():
    assert min_lag_hours(loc(2024, 6, 15, 0)) == 15


def test_lag_last_hour_ordinary():
    assert min_lag_hours(loc(2024, 6, 15, 23)) == 38


def test_lag_spring_forward():
    assert min_lag_hours(loc(2024, 3, 10, 23)) == 37


def test_lag_fall_back():
    assert min_lag_hours(loc(2024, 11, 3, 23)) == 39


def test_lag_accepts_utc_target():
    assert min_lag_hours(loc(2024, 6, 15, 23).astimezone(timezone.utc)) == 38


def test_naive_raises():
    with pytest.raises(ValueError):
        min_lag_hours(datetime(2024, 6, 15, 0))
    with pytest.raises(ValueError):
        assert_no_leak(loc(2024, 6, 15, 0), [datetime(2024, 6, 14, 9)])


def test_is_lag_valid():
    t = loc(2024, 6, 15, 23)
    assert is_lag_valid(t, 38) and not is_lag_valid(t, 37)
    with pytest.raises(TypeError):
        is_lag_valid(t, True)
    with pytest.raises(TypeError):
        is_lag_valid(t, 38.0)


def test_no_leak_boundary():
    t = loc(2024, 6, 15, 12)
    limit = loc(2024, 6, 14, 9)
    assert_no_leak(t, [limit, limit - timedelta(hours=5)])
    late = limit + timedelta(hours=1)
    with pytest.raises(LeakageError) as e:
        assert_no_leak(t, [limit, late])
    assert late.isoformat() in str(e.value)
    assert issubclass(LeakageError, ValueError)


def test_valid_variant_lags():
    assert valid_variant({"lags": [39, 168]})
    assert valid_variant({"lags": (39,)})
    assert not valid_variant({"lags": [38, 168]})
    assert not valid_variant({"lags": [True]})
    assert not valid_variant({"lags": [-48]})


def test_valid_variant_train_days():
    assert valid_variant({"lags": [168], "train_days": 7})
    assert not valid_variant({"lags": [168], "train_days": 6})
    assert not valid_variant({"lags": [48], "train_days": 7.0})
    assert not valid_variant({"lags": [48], "train_days": True})


def test_valid_variant_malformed():
    assert not valid_variant({})
    assert not valid_variant({"lags": []})
    assert not valid_variant({"lags": 48})
    assert not valid_variant({"lags": ["48"]})
    assert not valid_variant({"lags": [48.0]})
    assert not valid_variant(None)  # type: ignore[arg-type]


def test_earliest_valid_lag():
    assert UNIFORM_MIN_LAG_HOURS == 39
    assert earliest_valid_lag([date(2024, 6, 15)]) == 38
    assert earliest_valid_lag([date(2024, 3, 10)]) == 37
    assert earliest_valid_lag([date(2024, 11, 3)]) == 39
    assert earliest_valid_lag([date(2024, 6, 15), date(2024, 11, 3)]) == 39
    with pytest.raises(ValueError):
        earliest_valid_lag([])


def test_iter_grid_with_valid_variant():
    space = {"lags": [[24], [39], [39, 168]], "train_days": [1, 7]}
    got = list(iter_grid(space, valid_variant))
    assert {"lags": [24], "train_days": 7} not in got
    assert {"lags": [39], "train_days": 7} in got
    assert {"lags": [39, 168], "train_days": 1} not in got
    assert {"lags": [39], "train_days": 1} not in got  # 24 h of training < 39 h lag
    assert {"lags": [39, 168], "train_days": 7} in got
    assert len(got) == 2


def test_earliest_valid_lag_rollback_across_midnight():
    # St. John's 1988-10-30: clocks rolled back at 00:01, so local midnight happens twice.
    assert earliest_valid_lag([date(1988, 10, 30)], "America/St_Johns") == 40


def test_package_inits_have_empty_docstrings():
    import sandfish.kits
    import sandfish.kits.energy

    assert not sandfish.kits.__doc__
    assert not sandfish.kits.energy.__doc__

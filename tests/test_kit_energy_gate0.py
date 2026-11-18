import io
import json
from datetime import date

import pytest

from sandfish.kits.energy.gate0 import (
    check_ba,
    iter_series,
    local_day_hours,
    main,
    parse_eia_hour,
    summarize_series,
)

D_ID = "EBA.CISO-ALL.D.H"
DF_ID = "EBA.CISO-ALL.DF.H"


def _line(sid, data):
    return json.dumps({"series_id": sid, "name": "x", "units": "MWh", "f": "H", "data": data})


def _fixture():
    return [
        json.dumps({"category_id": "c"}),
        "garbage",
        _line(D_ID, [["20240101T02Z", 5], ["20240101T01Z", None], ["20240101T00Z", 4]]),
        _line(DF_ID, [["20240101T02Z", 6], ["20240101T01Z", 7]]),
    ]


def test_parse_hour_ok_and_bad():
    dt = parse_eia_hour("20240310T09Z")
    assert dt.isoformat() == "2024-03-10T09:00:00+00:00"
    for bad in ("2024-03-10T09Z", "20240310T24Z", "20240310T9Z", "", "20240310T09"):
        with pytest.raises(ValueError):
            parse_eia_hour(bad)


def test_summary_gap():
    s = summarize_series([["20240101T05Z", 1], ["20240101T00Z", 2], ["20240101T01Z", 3]])
    assert s["count"] == 3
    assert s["expected_hours"] == 6
    assert s["missing_hours"] == 3
    assert s["longest_gap_hours"] == 3
    assert s["first"] == "2024-01-01T00:00:00+00:00"
    assert s["last"] == "2024-01-01T05:00:00+00:00"


def test_summary_nulls_and_bad_periods():
    s = summarize_series([["20240101T00Z", None], ["20240101T01Z", 1], ["junk", 3]])
    assert s["null_values"] == 1
    assert s["bad_periods"] == 1
    assert s["count"] == 2
    assert s["longest_gap_hours"] == 0


def test_summary_duplicates_last_wins():
    s = summarize_series([["20240101T00Z", 1], ["20240101T00Z", None]])
    assert s["count"] == 1
    assert s["null_values"] == 1


def test_summary_empty():
    s = summarize_series([])
    assert s["count"] == 0 and s["first"] is None and s["missing_hours"] == 0


def test_iter_series_skips_and_stats():
    lines = [
        json.dumps({"category_id": "1", "name": "cat"}),
        "",
        "not json at all",
        _line(D_ID, []),
        _line("EBA.OTHER", []),
    ]
    stats: dict = {}
    out = list(iter_series(lines, {D_ID}, stats))
    assert [o["series_id"] for o in out] == [D_ID]
    assert stats == {"lines": 5, "series": 2, "skipped_malformed": 1}


@pytest.mark.parametrize(
    "day,hours",
    [(date(2024, 3, 10), 23), (date(2024, 11, 3), 25), (date(2024, 6, 15), 24)],
)
def test_local_day_hours(day, hours):
    assert local_day_hours(day) == hours


@pytest.mark.parametrize("day", [date(2024, 10, 6), date(2024, 4, 7)])
def test_local_day_hours_half_hour_dst_raises(day):
    with pytest.raises(ValueError) as exc:
        local_day_hours(day, "Australia/Lord_Howe")
    assert "Australia/Lord_Howe" in str(exc.value)
    assert day.isoformat() in str(exc.value)


def test_local_day_hours_half_hour_zone_ordinary_day():
    # Lord Howe has a 30-minute DST shift, but an ordinary day is still 24 h.
    assert local_day_hours(date(2024, 7, 1), "Australia/Lord_Howe") == 24


@pytest.mark.parametrize(
    "tz,day,hours",
    [
        ("America/Los_Angeles", date(2024, 3, 10), 23),
        ("America/Los_Angeles", date(2024, 11, 3), 25),
        ("America/Phoenix", date(2024, 3, 10), 24),
        ("America/Phoenix", date(2024, 11, 3), 24),
        ("Pacific/Honolulu", date(2024, 3, 10), 24),
        ("Pacific/Honolulu", date(2024, 11, 3), 24),
        ("America/St_Johns", date(2024, 3, 10), 23),
    ],
)
def test_local_day_hours_us_zones_unchanged(tz, day, hours):
    assert local_day_hours(day, tz) == hours


def test_check_ba_overlap():
    r = check_ba(_fixture())
    assert r["has_D"] and r["has_DF"]
    assert r["overlap_hours"] == 1
    assert r["D"]["null_values"] == 1
    assert r["parse_stats"]["skipped_malformed"] == 1


def test_check_ba_picks_larger_duplicate():
    short = _line(D_ID, [["20240101T00Z", 1]])
    longer = _line(D_ID, [["20240101T00Z", 1], ["20240101T01Z", 2]])
    assert check_ba([short, longer])["D"]["count"] == 2
    assert check_ba([longer, short])["D"]["count"] == 2


def test_cli_exit_codes(tmp_path, capsys):
    good = tmp_path / "good.txt"
    good.write_text("\n".join(_fixture()), encoding="utf-8")
    assert main([str(good)]) == 0
    assert json.loads(capsys.readouterr().out)["has_D"] is True
    bad = tmp_path / "bad.txt"
    bad.write_text(_line(D_ID, [["20240101T00Z", 1]]), encoding="utf-8")
    assert main([str(bad)]) == 1
    assert main([str(bad), "--ba", "NOPE"]) == 1
    capsys.readouterr()


def test_cli_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(_fixture())))
    assert main(["-"]) == 0
    capsys.readouterr()


def test_parse_rejects_non_ascii_digits():
    with pytest.raises(ValueError):
        parse_eia_hour("20240310T0９Z")
    with pytest.raises(ValueError):
        parse_eia_hour("20240310T09Z\n")


def test_invalid_series_id_does_not_abort_scan():
    lines = [
        '{"series_id": []}',
        '{"series_id": {"a": 1}}',
        '{"series_id": 5}',
        _line(D_ID, [["20240310T09Z", 1]]),
        _line(DF_ID, [["20240310T09Z", 2]]),
    ]
    assert len(list(iter_series(lines, {D_ID}))) == 1
    res = check_ba(lines)
    assert res["has_D"] and res["has_DF"]
    assert res["overlap_hours"] == 1

# File: test_recorder.py
# Location: tests/test_recorder.py
# Purpose: recorder's pre-flight quiescence wiring — sampled before t0, stamped into the header.
# Dependencies: flightrec.recorder, flightrec.quiesce

"""The quiescence stamp is GPU-free here: the NVML sampling is monkeypatched so
the wiring (skip-when-disabled, floor pass-through, header stamp) is tested on a
box with no GPU."""

from flightrec import quiesce
from flightrec.recorder import FlightRecorder


def test_quiescence_skipped_when_window_falsy():
    rec = FlightRecorder("/tmp/unused", quiesce_window=0)
    assert rec._sample_quiescence() is None
    rec_none = FlightRecorder("/tmp/unused", quiesce_window=None)
    assert rec_none._sample_quiescence() is None


def test_quiescence_sampled_with_floors(monkeypatch):
    captured = {}

    def fake_quiescence(seconds, **floors):
        captured["seconds"] = seconds
        captured["floors"] = floors
        return {"quiet": True, "verdict": "QUIET", "reasons": ["box at rest"]}

    monkeypatch.setattr(quiesce, "quiescence", fake_quiescence)
    rec = FlightRecorder("/tmp/unused", quiesce_window=0.5, quiesce_floors={"busy_floor": 1.0})
    result = rec._sample_quiescence()
    assert result["verdict"] == "QUIET"
    assert captured["seconds"] == 0.5
    assert captured["floors"] == {"busy_floor": 1.0}


def test_quiescence_swallows_sampler_failure(monkeypatch):
    def boom(seconds, **floors):
        raise RuntimeError("no NVML")

    monkeypatch.setattr(quiesce, "quiescence", boom)
    rec = FlightRecorder("/tmp/unused", quiesce_window=1.0)
    # a failed pre-flight sample must not crash the run — it just goes unstamped
    assert rec._sample_quiescence() is None

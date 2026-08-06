# File: test_live.py
# Location: tests/test_live.py
# Purpose: flightrec live — throttle-alarm logic, energy-rate/peak tracking, frame render.
# Dependencies: flightrec.live

"""The live arm: a standalone monitor whose throttle alarm reuses validate's wedge
rule (absolute PD-floor OR relative droop vs observed peak), gated on load. Alarm
logic + the rolling Dashboard + the frame renderer are tested without NVML; the loop
is exercised once with read_row monkeypatched."""

from flightrec import live as L


# ---------------------------------------------------------------------------
# alarm_state — same rule as validate, gated on load
# ---------------------------------------------------------------------------

def test_no_alarm_when_idle_even_if_clock_low():
    # Low clock but GPU not under load -> benign idle, no alarm.
    assert L.alarm_state(clock=611, power=13.0, busy=5.0, peak_clock=2300) is False


def test_alarm_on_absolute_pd_floor_under_load():
    # The 611 MHz / 13 W wedge signature under load -> alarm.
    assert L.alarm_state(clock=611, power=13.0, busy=90.0, peak_clock=0) is True


def test_alarm_on_relative_droop_vs_peak():
    # 1000 MHz is >50% below a 2300 MHz peak, under load -> alarm.
    assert L.alarm_state(clock=1000, power=45.0, busy=90.0, peak_clock=2300) is True


def test_no_alarm_at_healthy_clock_under_load():
    assert L.alarm_state(clock=2300, power=60.0, busy=90.0, peak_clock=2300) is False


def test_no_alarm_without_peak_when_clock_healthy():
    # No droop baseline yet and clock is above the absolute floor -> no alarm.
    assert L.alarm_state(clock=2200, power=55.0, busy=90.0, peak_clock=0) is False


# ---------------------------------------------------------------------------
# Dashboard — energy-rate diff + peak tracking + alarm
# ---------------------------------------------------------------------------

def test_energy_rate_none_on_first_tick_then_computed():
    board = L.Dashboard()
    f0 = board.update({"energy_mj": 0.0, "sm_clock_mhz": 2300, "gpu_busy_pct": 90.0}, t_s=0.0)
    assert f0["energy_rate_w"] is None            # no predecessor yet
    f1 = board.update({"energy_mj": 60000.0, "sm_clock_mhz": 2300, "gpu_busy_pct": 90.0}, t_s=1.0)
    assert f1["energy_rate_w"] == 60.0            # 60 J in 1 s


def test_peak_clock_tracks_loaded_max_and_drives_alarm():
    board = L.Dashboard()
    board.update({"sm_clock_mhz": 2300, "power_w": 60.0, "gpu_busy_pct": 90.0}, t_s=0.0)
    assert board.peak_clock == 2300
    # later droop to 900 MHz under load -> alarm fires against the recorded peak.
    frame = board.update({"sm_clock_mhz": 900, "power_w": 45.0, "gpu_busy_pct": 90.0}, t_s=1.0)
    assert frame["alarm"] is True


def test_idle_samples_do_not_set_peak():
    board = L.Dashboard()
    board.update({"sm_clock_mhz": 2300, "power_w": 8.0, "gpu_busy_pct": 2.0}, t_s=0.0)
    assert board.peak_clock == 0                  # idle sample ignored for the baseline


# ---------------------------------------------------------------------------
# format_frame — render + alarm banner
# ---------------------------------------------------------------------------

def test_format_frame_shows_lanes_and_na():
    frame = {"sm_clock_mhz": 2300, "power_w": 60.0, "gpu_busy_pct": 90.0,
             "peak_clock_mhz": 2300, "energy_rate_w": 59.0, "alarm": False}
    text = L.format_frame(frame, max_clock=2600)
    assert "SM clock   2300/2600 MHz" in text
    assert "n/a" in text                          # missing lanes degrade gracefully
    assert "THROTTLE ALARM" not in text


def test_format_frame_alarm_banner():
    frame = {"sm_clock_mhz": 900, "power_w": 13.0, "gpu_busy_pct": 90.0,
             "peak_clock_mhz": 2300, "energy_rate_w": 5.0, "alarm": True}
    assert "THROTTLE ALARM" in L.format_frame(frame)


# ---------------------------------------------------------------------------
# live() — one frame, read_row monkeypatched (no NVML)
# ---------------------------------------------------------------------------

def test_live_once_returns_frame(monkeypatch, capsys):
    monkeypatch.setattr(L, "open_gpu", lambda idx: None)
    monkeypatch.setattr(L, "_max_clock", lambda h: 2600)
    monkeypatch.setattr(L, "read_row", lambda h, prev: (
        {"sm_clock_mhz": 611, "power_w": 13.0, "gpu_busy_pct": 90.0, "energy_mj": 0.0}, {}))
    frame = L.live(once=True)
    assert frame["alarm"] is True                 # synthetic wedge -> alarm
    assert "THROTTLE ALARM" in capsys.readouterr().out

"""The card reading, the telemetry file and the stop limits."""
import os

import pytest

from app.core.sampler_compare import plan, telemetry
from app.core.sampler_compare.telemetry import Card, Sample

LIMITS = plan.Limits(temperature_c=94, vram_mib=88000, free_ram_gib=10, stall_minutes=15, run_minutes=120,
                     idle_vram_mib=20000, idle_wait_minutes=30, power_limit_w=500)
GIB_MIB = 1024


def _sample(card=Card(4801, 56.0, 300.0, 500.0), elapsed=60.0, ram_gib=50.0, step=10):
    return Sample(elapsed_s=elapsed, step=step, step_time_s=1.5, card=card,
                  available_ram_mib=ram_gib * GIB_MIB, rss_mib=2048, note="")


def _health(temperature=56, power=32.55, limit=500.0, readable=True):
    """A ``gpu_health.read_gpu_health`` reading, with the fields the card file needs."""
    return {"readable": readable, "temperature_c": temperature, "power_draw_w": power, "power_limit_w": limit}


def _reader(reading, vram=4801):
    asked = []

    def health(index):
        asked.append(("health", index))
        return reading

    def memory(index):
        asked.append(("vram", index))
        return vram

    return health, memory, asked


def test_a_reading_of_the_card_it_is_given_becomes_a_card():
    health, memory, asked = _reader(_health())
    assert telemetry.read_card(1, health=health, vram=memory) == Card(4801, 56.0, 32.55, 500.0)
    assert asked == [("health", 1), ("vram", 1)]


def test_a_field_the_card_does_not_offer_is_none():
    health, memory, _ = _reader(_health(power=None, limit=None))
    assert telemetry.read_card(0, health=health, vram=memory) == Card(4801, 56.0, None, None)


@pytest.mark.parametrize(
    ("reading", "vram"),
    [(None, 4801), ({"readable": False, "error": "GPU is lost"}, 4801), (_health(temperature=None), 4801),
     (_health(), None)],
    ids=["no-nvml", "card-lost", "no-temperature", "no-memory"],
)
def test_an_unreadable_card_is_no_reading(reading, vram):
    health, memory, _ = _reader(reading, vram)
    assert telemetry.read_card(0, health=health, vram=memory) is None


def test_without_nvml_the_memory_is_no_reading(monkeypatch):
    monkeypatch.setattr(telemetry.gpu_health, "nvml_ready", lambda: False)
    assert telemetry.vram_used_mib(0) is None


@pytest.mark.parametrize("count", [None, 0, 1])
def test_one_card_or_none_is_card_zero(count):
    assert telemetry.card_choice(count, {"CUDA_VISIBLE_DEVICES": "3"}) == (0, None)


@pytest.mark.parametrize(
    "env",
    [{}, {"CUDA_VISIBLE_DEVICES": "1"}, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
     {"CUDA_VISIBLE_DEVICES": "0,1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
     {"CUDA_VISIBLE_DEVICES": "2", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
     {"CUDA_VISIBLE_DEVICES": "GPU-1234", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}],
    ids=["nothing", "no-order", "no-card", "two-cards", "no-such-card", "uuid"],
)
def test_two_cards_without_one_named_in_bus_order_are_refused(env):
    index, problem = telemetry.card_choice(2, env)
    assert "2 cards" in problem and "CUDA_VISIBLE_DEVICES" in problem and "PCI_BUS_ID" in problem


def test_two_cards_with_one_named_in_bus_order_watch_that_one():
    assert telemetry.card_choice(2, {"CUDA_VISIBLE_DEVICES": " 1 ", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}) == (1, None)


def test_the_card_count_comes_from_nvml(monkeypatch):
    monkeypatch.setattr(telemetry.gpu_health, "nvml_ready", lambda: False)
    assert telemetry.nvml_device_count() is None

    class FakeNvml:
        NVMLError = RuntimeError

        @staticmethod
        def nvmlDeviceGetCount():
            return 2

    monkeypatch.setattr(telemetry.gpu_health, "nvml_ready", lambda: True)
    monkeypatch.setattr(telemetry.gpu_health, "pynvml", FakeNvml)
    assert telemetry.nvml_device_count() == 2


def test_the_file_has_a_header_and_a_row_once_started(tmp_path):
    file = telemetry.TelemetryFile(tmp_path / "run.telemetry.csv")
    file.start()
    file.row(_sample())
    lines = (tmp_path / "run.telemetry.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == telemetry.COLUMNS
    assert len(lines[1].split(",")) == 11


def test_every_row_is_on_disk_and_no_handle_stays_open(tmp_path):
    path = tmp_path / "run.telemetry.csv"
    file = telemetry.TelemetryFile(path)
    file.start()
    file.row(_sample())
    with os.scandir(tmp_path) as entries:
        size = next(e for e in entries if e.name == path.name).stat().st_size
    assert size > len(telemetry.COLUMNS) + 2
    os.rename(path, tmp_path / "moved.csv")  # Windows refuses to rename a file a writer holds
    os.rename(tmp_path / "moved.csv", path)
    file.row(_sample())
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_a_fresh_start_replaces_an_earlier_file(tmp_path):
    path = tmp_path / "run.telemetry.csv"
    path.write_text("old\nrows\nhere\n", encoding="utf-8")
    telemetry.TelemetryFile(path).start()
    assert path.read_text(encoding="utf-8").splitlines() == [telemetry.COLUMNS]


def test_an_unreadable_card_is_a_well_formed_row(tmp_path):
    path = tmp_path / "run.telemetry.csv"
    file = telemetry.TelemetryFile(path)
    file.start()
    file.row(_sample(card=None))
    cells = path.read_text(encoding="utf-8").splitlines()[1].split(",")
    header = telemetry.COLUMNS.split(",")
    assert len(cells) == 11
    row = dict(zip(header, cells, strict=True))
    assert [row[k] for k in ("vram_mib", "temperature_c", "power_w", "power_limit_w")] == ["", "", "", ""]
    assert row["note"] == "card unreadable"


def test_a_value_at_its_limit_does_not_stop_a_run():
    at_limits = _sample(card=Card(88000, 94.0, 499.0, 500.0), elapsed=7200.0, ram_gib=10.0)
    assert telemetry.breach(at_limits, LIMITS, quiet_s=900.0) is None


@pytest.mark.parametrize(
    ("sample", "quiet", "reason"),
    [
        (_sample(card=Card(100, 94.5, 1.0, 500.0)), 0.0, "the card reads 94.5 C, over 94 C"),
        (_sample(card=Card(88001, 50.0, 1.0, 500.0)), 0.0, "VRAM 88001 MiB, over 88000 MiB"),
        (_sample(ram_gib=9.9), 0.0, "host available RAM 9.9 GiB, under 10 GiB"),
        (_sample(), 901.0, "no log output for 15 min"),
        (_sample(elapsed=7201.0), 0.0, "over the 120 min limit"),
        (_sample(card=None), 0.0, "the card can't be read"),
    ],
)
def test_each_limit_names_its_breach(sample, quiet, reason):
    assert telemetry.breach(sample, LIMITS, quiet_s=quiet) == reason


def test_the_minimum_ram_is_the_lowest_sample_not_the_last():
    peaks = telemetry.Peaks()
    for gib in (54.0, 9.2, 60.0):
        peaks.add(_sample(ram_gib=gib))
    summary = peaks.summary()
    assert summary["min_available_ram_gib"] == 9.2
    assert summary["peak_vram_mib"] == 4801
    assert summary["peak_temperature_c"] == 56.0
    assert summary["peak_power_w"] == 300.0
    assert summary["peak_rss_gib"] == 2.0
    assert summary["mean_step_time_s"] == 1.5


def test_peaks_of_no_sample_are_none():
    assert set(telemetry.Peaks().summary().values()) == {None}

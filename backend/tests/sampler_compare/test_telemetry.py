"""The card reading, the telemetry file and the stop limits."""
import os
import subprocess

import pytest

from app.core.sampler_compare import plan, telemetry
from app.core.sampler_compare.telemetry import Card, Sample

LIMITS = plan.Limits(temperature_c=94, vram_mib=88000, free_ram_gib=10, stall_minutes=15, run_minutes=120,
                     idle_vram_mib=20000, idle_wait_minutes=30, power_limit_w=500)
GIB_MIB = 1024


def _sample(card=Card(4801, 56.0, 300.0, 500.0), elapsed=60.0, ram_gib=50.0, step=10):
    return Sample(elapsed_s=elapsed, step=step, step_time_s=1.5, card=card,
                  available_ram_mib=ram_gib * GIB_MIB, rss_mib=2048, note="")


def test_a_decimal_temperature_parses():
    assert telemetry.parse_card("4801, 56.22, 32.55, 500.00") == Card(4801, 56.22, 32.55, 500.0)


def test_a_field_the_card_does_not_offer_is_none():
    assert telemetry.parse_card("4801, 56, [N/A], [N/A]") == Card(4801, 56.0, None, None)


@pytest.mark.parametrize("line", ["", "NVIDIA-SMI has failed", "a, b, c, d", "4801, 56"])
def test_a_garbled_line_is_no_reading(line):
    assert telemetry.parse_card(line) is None


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("nvidia-smi", 30), FileNotFoundError("nvidia-smi")])
def test_a_reader_that_times_out_or_is_missing_is_no_reading(error):
    def run(*args, **kwargs):
        assert kwargs["timeout"] == 30
        raise error

    assert telemetry.read_card(run=run) is None


def test_a_reader_reads_the_first_line():
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="4801, 56.22, 32.55, 500.00\n100, 30, 20, 300\n")

    assert telemetry.read_card(run=run) == Card(4801, 56.22, 32.55, 500.0)


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

"""The GPU health reader, against a fake NVML table.

The fake holds the functions only. Its constants and error classes are the real
binding's, and errors are built as the binding builds them: ``_nvmlCheckReturn``
raises ``NVMLError(code)``, whose ``__new__`` picks a subclass generated at import
from the ``NVML_ERROR_*`` constants. No test here calls into a driver.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from types import SimpleNamespace

import pytest

from app.core import gpu_health

real = pytest.importorskip("pynvml")

HANDLE = object()
UNNAMED_ERROR_CODE = 12345

HEALTHY = {
    "handle": HANDLE,
    "temperature": 71,
    "margin": 12,
    "fan": 64,
    "clock_graphics": 2610,
    "clock_memory": 13801,
    "power_draw": 412500,
    "power_limit": 600000,
    "reasons": 0x4,
    "violation_thermal": 2_000_000,
    "violation_power": 10_543_881_184,
}

HEALTHY_READING = {
    "temperature_c": 71,
    "temperature_margin_c": 12,
    "fan_percent": 64,
    "clock_graphics_mhz": 2610,
    "clock_memory_mhz": 13801,
    "power_draw_w": 412.5,
    "power_limit_w": 600.0,
    "throttle_raw": "0x4",
    "throttle_reasons": ["sw_power_cap"],
    "thermal_throttle_ms": 2.0,
    "power_throttle_ms": 10543.881184,
    "readable": True,
}

# Each query and the reading's fields that depend on it alone.
QUERY_FIELDS = {
    "temperature": ("temperature_c",),
    "margin": ("temperature_margin_c",),
    "fan": ("fan_percent",),
    "clock_graphics": ("clock_graphics_mhz",),
    "clock_memory": ("clock_memory_mhz",),
    "power_draw": ("power_draw_w",),
    "power_limit": ("power_limit_w",),
    "reasons": ("throttle_raw", "throttle_reasons"),
    "violation_thermal": ("thermal_throttle_ms",),
    "violation_power": ("power_throttle_ms",),
}

FAKE_FUNCTIONS = (
    "nvmlDeviceGetHandleByIndex",
    "nvmlDeviceGetTemperature",
    "nvmlDeviceGetMarginTemperature",
    "nvmlDeviceGetFanSpeed",
    "nvmlDeviceGetClockInfo",
    "nvmlDeviceGetPowerUsage",
    "nvmlDeviceGetEnforcedPowerLimit",
    "nvmlDeviceGetCurrentClocksEventReasons",
    "nvmlDeviceGetViolationStatus",
)


def make_fake(**overrides):
    """A stand-in for the ``pynvml`` module, answering from a table."""
    table = {**HEALTHY, **overrides}
    asked_indexes: list[int] = []

    def answer(key):
        value = table[key]
        if isinstance(value, BaseException):
            raise value
        return value

    def device_query(key):
        def query(handle):
            assert handle is HANDLE
            return answer(key)

        return query

    def handle_by_index(index):
        asked_indexes.append(index)
        return answer("handle")

    def temperature(handle, sensor):
        assert handle is HANDLE
        assert sensor == real.NVML_TEMPERATURE_GPU
        return answer("temperature")

    def clock_info(handle, clock):
        assert handle is HANDLE
        return answer(
            {
                real.NVML_CLOCK_GRAPHICS: "clock_graphics",
                real.NVML_CLOCK_MEM: "clock_memory",
            }[clock]
        )

    def violation_status(handle, policy):
        assert handle is HANDLE
        key = {
            real.NVML_PERF_POLICY_THERMAL: "violation_thermal",
            real.NVML_PERF_POLICY_POWER: "violation_power",
        }[policy]
        return SimpleNamespace(violationTime=answer(key), referenceTime=999)

    constants = {name: getattr(real, name) for name in dir(real) if name.startswith("NVML")}
    return SimpleNamespace(
        **constants,
        asked_indexes=asked_indexes,
        nvmlDeviceGetHandleByIndex=handle_by_index,
        nvmlDeviceGetTemperature=temperature,
        nvmlDeviceGetMarginTemperature=device_query("margin"),
        nvmlDeviceGetFanSpeed=device_query("fan"),
        nvmlDeviceGetClockInfo=clock_info,
        nvmlDeviceGetPowerUsage=device_query("power_draw"),
        nvmlDeviceGetEnforcedPowerLimit=device_query("power_limit"),
        nvmlDeviceGetCurrentClocksEventReasons=device_query("reasons"),
        nvmlDeviceGetViolationStatus=violation_status,
    )


@pytest.fixture
def read(monkeypatch):
    """Read index 0 from a fake built with the given table overrides."""

    def _read(**overrides):
        monkeypatch.setattr(gpu_health, "_NVML_AVAILABLE", True)
        monkeypatch.setattr(gpu_health, "pynvml", make_fake(**overrides))
        return gpu_health.read_gpu_health(0)

    return _read


def load_fresh_copy():
    """Execute the module's source again, so its import-time degrade runs."""
    spec = importlib.util.spec_from_file_location("gpu_health_fresh", gpu_health.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- The fake stays honest about the binding ---------------------------------


def test_the_fake_only_offers_functions_the_real_binding_has():
    assert [name for name in FAKE_FUNCTIONS if not hasattr(real, name)] == []


@pytest.mark.parametrize(
    ("code", "error_class"),
    [
        (real.NVML_ERROR_NOT_SUPPORTED, real.NVMLError_NotSupported),
        (real.NVML_ERROR_NO_PERMISSION, real.NVMLError_NoPermission),
        (real.NVML_ERROR_FUNCTION_NOT_FOUND, real.NVMLError_FunctionNotFound),
        (real.NVML_ERROR_GPU_IS_LOST, real.NVMLError_GpuIsLost),
        (real.NVML_ERROR_UNKNOWN, real.NVMLError_Unknown),
    ],
)
def test_the_binding_turns_a_code_into_its_generated_subclass(code, error_class):
    error = real.NVMLError(code)
    assert type(error) is error_class
    assert isinstance(error, real.NVMLError)


def test_the_binding_keeps_the_base_class_for_a_code_it_does_not_name():
    assert type(real.NVMLError(UNNAMED_ERROR_CODE)) is real.NVMLError


@pytest.mark.parametrize(
    ("constant", "name"),
    [
        ("nvmlClocksEventReasonGpuIdle", "idle"),
        ("nvmlClocksEventReasonApplicationsClocksSetting", "applications_clocks"),
        ("nvmlClocksEventReasonSwPowerCap", "sw_power_cap"),
        ("nvmlClocksEventReasonHwSlowdown", "hw_slowdown"),
        ("nvmlClocksEventReasonSyncBoost", "sync_boost"),
        ("nvmlClocksEventReasonSwThermalSlowdown", "sw_thermal"),
        ("nvmlClocksEventReasonHwThermalSlowdown", "hw_thermal"),
        ("nvmlClocksEventReasonHwPowerBrakeSlowdown", "hw_power_brake"),
        ("nvmlClocksEventReasonDisplayClockSetting", "display_clock"),
    ],
)
def test_each_reason_the_binding_names_reads_as_its_name(read, constant, name):
    bit = getattr(real, constant)
    reading = read(reasons=bit)
    assert reading["throttle_reasons"] == [name]
    assert reading["throttle_raw"] == hex(bit)


# --- All fields --------------------------------------------------------------


def test_all_fields_are_read(read):
    assert read() == HEALTHY_READING


def test_the_asked_index_reaches_the_handle_lookup(monkeypatch):
    fake = make_fake()
    monkeypatch.setattr(gpu_health, "_NVML_AVAILABLE", True)
    monkeypatch.setattr(gpu_health, "pynvml", fake)
    gpu_health.read_gpu_health(3)
    assert fake.asked_indexes == [3]


def test_zero_values_stay_values(read):
    reading = read(
        margin=0,
        fan=0,
        clock_graphics=0,
        clock_memory=0,
        power_draw=0,
        violation_thermal=0,
        violation_power=0,
    )
    assert reading["temperature_margin_c"] == 0
    assert reading["fan_percent"] == 0
    assert reading["clock_graphics_mhz"] == 0
    assert reading["clock_memory_mhz"] == 0
    assert reading["power_draw_w"] == 0
    assert reading["thermal_throttle_ms"] == 0
    assert reading["power_throttle_ms"] == 0


# --- A field this card doesn't offer -----------------------------------------


@pytest.mark.parametrize("query", QUERY_FIELDS)
@pytest.mark.parametrize(
    "code",
    [
        real.NVML_ERROR_NOT_SUPPORTED,
        real.NVML_ERROR_NO_PERMISSION,
        real.NVML_ERROR_FUNCTION_NOT_FOUND,
    ],
)
def test_a_field_the_card_does_not_offer_is_none_and_the_others_stay(read, query, code):
    expected = dict(HEALTHY_READING)
    for field in QUERY_FIELDS[query]:
        expected[field] = None
    assert read(**{query: real.NVMLError(code)}) == expected


# --- A device that failed ----------------------------------------------------


@pytest.mark.parametrize("query", QUERY_FIELDS)
@pytest.mark.parametrize(
    "code",
    [
        real.NVML_ERROR_GPU_IS_LOST,
        real.NVML_ERROR_UNKNOWN,
        real.NVML_ERROR_DRIVER_NOT_LOADED,
        UNNAMED_ERROR_CODE,
    ],
)
def test_any_other_nvml_error_from_a_field_makes_the_reading_unreadable(read, query, code):
    reading = read(**{query: real.NVMLError(code)})
    assert set(reading) == {"readable", "error"}
    assert reading["readable"] is False
    assert isinstance(reading["error"], str) and reading["error"]


@pytest.mark.parametrize(
    "code",
    [
        real.NVML_ERROR_GPU_IS_LOST,
        real.NVML_ERROR_UNKNOWN,
        real.NVML_ERROR_NOT_SUPPORTED,
        real.NVML_ERROR_INVALID_ARGUMENT,
        UNNAMED_ERROR_CODE,
    ],
)
def test_a_handle_failure_makes_the_reading_unreadable(read, code):
    reading = read(handle=real.NVMLError(code))
    assert set(reading) == {"readable", "error"}
    assert reading["readable"] is False
    assert isinstance(reading["error"], str) and reading["error"]


def test_the_error_text_says_the_card_was_lost(read):
    reading = read(fan=real.NVMLError(real.NVML_ERROR_GPU_IS_LOST))
    assert reading == {"readable": False, "error": "GPU is lost"}


def test_a_lost_card_outweighs_a_field_that_was_only_absent(read):
    reading = read(
        margin=real.NVMLError(real.NVML_ERROR_NOT_SUPPORTED),
        violation_power=real.NVMLError(real.NVML_ERROR_GPU_IS_LOST),
    )
    assert reading == {"readable": False, "error": "GPU is lost"}


def test_an_error_outside_nvml_does_not_escape(read):
    reading = read(fan=OSError("the call blew up"))
    assert reading == {"readable": False, "error": "the call blew up"}


def test_an_error_whose_text_cannot_be_built_does_not_escape(read):
    class TextlessError(real.NVMLError):
        def __str__(self):
            raise RuntimeError("no text")

    reading = read(fan=TextlessError(real.NVML_ERROR_UNKNOWN))
    assert reading == {"readable": False, "error": "TextlessError"}


# --- No NVML -----------------------------------------------------------------


def test_no_nvml_gives_none(monkeypatch):
    monkeypatch.setattr(gpu_health, "_NVML_AVAILABLE", False)
    assert gpu_health.read_gpu_health(0) is None


def test_a_missing_binding_degrades_at_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    module = load_fresh_copy()
    assert module.nvml_ruled_out() is True
    assert module.read_gpu_health(0) is None


def test_importing_the_module_makes_no_nvml_call(monkeypatch):
    calls: list[str] = []
    fake = SimpleNamespace(nvmlInit=lambda: calls.append("init"))
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    module = load_fresh_copy()
    assert calls == []
    assert module.nvml_ruled_out() is False  # not known yet


def test_a_failing_init_degrades_at_the_first_read_and_is_not_tried_again(monkeypatch):
    calls: list[str] = []

    def no_library():
        calls.append("init")
        raise real.NVMLError(real.NVML_ERROR_LIBRARY_NOT_FOUND)

    monkeypatch.setitem(sys.modules, "pynvml", SimpleNamespace(nvmlInit=no_library))
    module = load_fresh_copy()
    assert module.read_gpu_health(0) is None
    assert module.read_gpu_health(0) is None
    assert calls == ["init"]
    assert module.nvml_ruled_out() is True


def test_a_working_init_runs_once_at_the_first_read(monkeypatch):
    calls: list[str] = []
    fake = make_fake()
    fake.nvmlInit = lambda: calls.append("init")
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    module = load_fresh_copy()
    assert module.read_gpu_health(0)["readable"] is True
    assert module.read_gpu_health(0)["readable"] is True
    assert calls == ["init"]
    assert module.nvml_ready() is True


def test_a_failing_init_logs_nothing_a_job_log_would_show(monkeypatch, caplog):
    def no_library():
        raise real.NVMLError(real.NVML_ERROR_LIBRARY_NOT_FOUND)

    monkeypatch.setitem(sys.modules, "pynvml", SimpleNamespace(nvmlInit=no_library))
    module = load_fresh_copy()
    # On the recorder's thread this could land after the job's stop.
    with caplog.at_level(logging.INFO):
        assert module.read_gpu_health(0) is None
    assert caplog.records == []


# --- Throttle reasons --------------------------------------------------------


@pytest.mark.parametrize(
    ("mask", "raw", "names"),
    [
        (0x0, "0x0", []),
        (0x4, "0x4", ["sw_power_cap"]),
        (0x400, "0x400", ["unknown_0x400"]),
        (0x44, "0x44", ["sw_power_cap", "hw_thermal"]),
        (0x404, "0x404", ["sw_power_cap", "unknown_0x400"]),
        (0xC00, "0xc00", ["unknown_0x400", "unknown_0x800"]),
    ],
)
def test_the_reasons_mask_reads_as_raw_hex_and_names(read, mask, raw, names):
    reading = read(reasons=mask)
    assert reading["throttle_raw"] == raw
    assert reading["throttle_reasons"] == names


# --- Margin ------------------------------------------------------------------


@pytest.mark.parametrize("value", ["56", 56.5, None, True, object()])
def test_a_margin_that_is_not_an_int_is_none(read, value):
    reading = read(margin=value)
    expected = {**HEALTHY_READING, "temperature_margin_c": None}
    assert reading == expected


def test_a_margin_past_the_limit_stays_negative(read):
    assert read(margin=-3)["temperature_margin_c"] == -3


# --- Violation counters ------------------------------------------------------


def test_a_violation_time_is_nanoseconds_read_as_milliseconds(read):
    reading = read(violation_thermal=2_000_000, violation_power=5_000_000)
    assert reading["thermal_throttle_ms"] == 2
    assert reading["power_throttle_ms"] == 5

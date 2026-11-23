
"""
TOKUMA 3-in-1 Integrated Model
Agricultural Sensor Data Quality Control

Purpose:
1. Detect missing sensor measurements.
2. Detect technically abnormal values.
3. Check timestamp validity.
4. Produce quality flags and status.
5. Preserve original sensor measurements.
"""

from datetime import datetime, timezone
import math
import json
from pathlib import Path


# --------------------------------------------------
# 1. Define technical measurement ranges
# --------------------------------------------------

VALID_RANGES = {
    "soil_moisture": (0, 100),
    "air_temperature": (-50, 70),
    "relative_humidity": (0, 100),
    "rainfall": (0, None)
}

REQUIRED_FIELDS = [
    "field_id",
    "sensor_id",
    "soil_moisture",
    "timestamp"
]

OPTIONAL_FIELDS = [
    "air_temperature",
    "relative_humidity",
    "rainfall"
]


# --------------------------------------------------
# 2. Evaluate one sensor reading
# --------------------------------------------------

def evaluate_sensor_reading(reading):

    flags = []

    # Check required fields
    for field in REQUIRED_FIELDS:

        if reading.get(field) is None:
            flags.append(f"MISSING_REQUIRED_{field.upper()}")

    # Check optional measurements
    for field in OPTIONAL_FIELDS:

        if reading.get(field) is None:
            flags.append(f"MISSING_{field.upper()}")

    # Check numerical measurements
    for field, limits in VALID_RANGES.items():

        value = reading.get(field)

        if value is None:
            continue

        if isinstance(value, bool):
            flags.append(f"INVALID_{field.upper()}")
            continue

        try:
            numeric_value = float(value)

        except (TypeError, ValueError):
            flags.append(f"INVALID_{field.upper()}")
            continue

        if not math.isfinite(numeric_value):
            flags.append(f"NON_FINITE_{field.upper()}")
            continue

        minimum, maximum = limits

        if minimum is not None and numeric_value < minimum:
            flags.append(f"OUT_OF_RANGE_{field.upper()}")

        if maximum is not None and numeric_value > maximum:
            flags.append(f"OUT_OF_RANGE_{field.upper()}")

    # Check timestamp
    timestamp = reading.get("timestamp")

    if timestamp is not None:

        try:
            datetime.fromisoformat(
                str(timestamp).replace("Z", "+00:00")
            )

        except (ValueError, TypeError):
            flags.append("INVALID_TIMESTAMP")

    # Determine quality status
    if any(
        flag.startswith("MISSING_REQUIRED")
        or flag.startswith("INVALID_")
        or flag.startswith("NON_FINITE_")
        or flag.startswith("OUT_OF_RANGE_")
        for flag in flags
    ):
        status = "REVIEW"

    elif flags:
        status = "INCOMPLETE"

    else:
        status = "ACCEPTABLE"

    return {
        "field_id": reading.get("field_id"),
        "sensor_id": reading.get("sensor_id"),
        "quality_status": status,
        "quality_flags": flags,
        "flag_count": len(flags)
    }


# --------------------------------------------------
# 3. Evaluate multiple sensor readings
# --------------------------------------------------

def evaluate_sensor_history(readings):

    results = []

    for reading in readings:

        results.append(
            evaluate_sensor_reading(reading)
        )

    return results


# --------------------------------------------------
# 4. Summarize quality results
# --------------------------------------------------

def summarize_quality_results(results):

    summary = {
        "total_readings": len(results),
        "acceptable": 0,
        "incomplete": 0,
        "review": 0,
        "total_flags": 0
    }

    for result in results:

        status = result["quality_status"]

        if status == "ACCEPTABLE":
            summary["acceptable"] += 1

        elif status == "INCOMPLETE":
            summary["incomplete"] += 1

        elif status == "REVIEW":
            summary["review"] += 1

        summary["total_flags"] += result["flag_count"]

    return summary

# --------------------------------------------------
# 5. Persistent sensor quality monitoring log
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

QUALITY_LOG_PATH = (
    PROJECT_DIR / "tokuma_sensor_quality_log.jsonl"
)


def log_quality_result(reading, result):

    log_entry = {

        "assessment_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "field_id": result.get("field_id"),

        "sensor_id": result.get("sensor_id"),

        "quality_status": result.get("quality_status"),

        "quality_flags": result.get("quality_flags"),

        "flag_count": result.get("flag_count")

    }

    with open(
        QUALITY_LOG_PATH,
        "a",
        encoding="utf-8"
    ) as log_file:

        log_file.write(
            json.dumps(log_entry) + "\n"
        )

    return log_entry


# --------------------------------------------------
# 6. Evaluate and record a quality assessment
# --------------------------------------------------

def evaluate_and_log_sensor_reading(reading):

    result = evaluate_sensor_reading(reading)

    log_quality_result(reading, result)

    return result

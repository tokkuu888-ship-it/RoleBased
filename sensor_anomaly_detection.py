
"""
TOKUMA 3-in-1 Integrated Model
Agricultural Sensor Anomaly Detection

Purpose:
1. Detect unusual changes in sensor measurements.
2. Identify persistent flatline observations.
3. Identify statistical outliers.
4. Detect delayed sensor observations.
5. Preserve original sensor measurements.

IMPORTANT:
Initial detection parameters are provisional software-testing
settings. They require validation using actual field observations.
"""

from datetime import datetime, timezone
import math
import statistics


# ==================================================
# 1. SENSOR MONITORING CONFIGURATION
# ==================================================

MONITORED_FIELDS = [
    "soil_moisture",
    "air_temperature",
    "relative_humidity",
    "rainfall"
]
# ==================================================
# 1.1 ANOMALY SEVERITY CLASSIFICATION
# ==================================================

VALID_SEVERITY_LEVELS = [
    "INFO",
    "WARNING",
    "CRITICAL"
]
# ==================================================
# 2. PROVISIONAL OBSERVATION INTERVAL
# ==================================================

EXPECTED_INTERVAL_SECONDS = 15

STALE_READING_MULTIPLIER = 3

STALE_READING_SECONDS = (
    EXPECTED_INTERVAL_SECONDS
    * STALE_READING_MULTIPLIER
)


# ==================================================
# 3. STATISTICAL DETECTION CONFIGURATION
# ==================================================

MINIMUM_HISTORY_FOR_STATISTICS = 5

ROLLING_WINDOW_SIZE = 10

MAD_ANOMALY_THRESHOLD = 3.5
# ==================================================
# 3.1 PROVISIONAL SUDDEN-CHANGE THRESHOLDS
# ==================================================

SUDDEN_CHANGE_THRESHOLDS = {
    "soil_moisture": 15.0,
    "air_temperature": 8.0,
    "relative_humidity": 20.0,
    "rainfall": 10.0
}

# ==================================================
# 4. FLATLINE DETECTION CONFIGURATION
# ==================================================

MINIMUM_FLATLINE_READINGS = 8
MINIMUM_FLATLINE_DURATION_SECONDS = (
    EXPECTED_INTERVAL_SECONDS
    * (MINIMUM_FLATLINE_READINGS - 1)
)
# ==================================================
# 5. DETECTION CONFIGURATION VALIDATION
# ==================================================

def validate_detection_configuration():

    if EXPECTED_INTERVAL_SECONDS <= 0:
        raise ValueError(
            "Expected observation interval must be positive."
        )

    if STALE_READING_MULTIPLIER <= 0:
        raise ValueError(
            "Stale reading multiplier must be positive."
        )

    if MINIMUM_HISTORY_FOR_STATISTICS < 3:
        raise ValueError(
            "Statistical history must contain at least 3 readings."
        )

    if ROLLING_WINDOW_SIZE < MINIMUM_HISTORY_FOR_STATISTICS:
        raise ValueError(
            "Rolling window must not be smaller than minimum history."
        )

    if MAD_ANOMALY_THRESHOLD <= 0:
        raise ValueError(
            "MAD anomaly threshold must be positive."
        )

    if MINIMUM_FLATLINE_READINGS < 2:
        raise ValueError(
            "Flatline detection requires at least 2 readings."
        )

    return True


# ==================================================
# 6. INITIALIZE CONFIGURATION
# ==================================================

validate_detection_configuration()

# ==================================================
# 7. TIMESTAMP NORMALIZATION
# ==================================================

def normalize_sensor_timestamp(timestamp):

    if timestamp is None:
        return None

    try:

        if isinstance(timestamp, datetime):
            parsed_timestamp = timestamp

        else:
            parsed_timestamp = datetime.fromisoformat(
                str(timestamp).replace("Z", "+00:00")
            )

        # Treat naive timestamps as UTC only when explicitly
        # required by the source convention.
        if parsed_timestamp.tzinfo is None:
            return None

        return parsed_timestamp.astimezone(timezone.utc)

    except (ValueError, TypeError, OverflowError):

        return None


# ==================================================
# 8. PREPARE SENSOR HISTORY
# ==================================================

def prepare_sensor_history(readings):

    prepared_readings = []

    if not isinstance(readings, (list, tuple)):
        return prepared_readings

    for reading in readings:

        if not isinstance(reading, dict):
            continue

        timestamp = normalize_sensor_timestamp(
            reading.get("timestamp")
        )

        if timestamp is None:
            continue

        prepared_reading = reading.copy()

        prepared_reading["_normalized_timestamp"] = timestamp

        prepared_readings.append(prepared_reading)

    # Sort chronologically.
    # Original sensor measurements remain unchanged.

    prepared_readings.sort(
        key=lambda item: item["_normalized_timestamp"]
    )

    return prepared_readings


# ==================================================
# 9. GROUP HISTORY BY INDIVIDUAL SENSOR
# ==================================================

def group_readings_by_sensor(readings):

    sensor_groups = {}

    prepared_readings = prepare_sensor_history(
        readings
    )

    for reading in prepared_readings:

        field_id = reading.get("field_id")
        sensor_id = reading.get("sensor_id")

        if not field_id or not sensor_id:
            continue

        sensor_key = (
            str(field_id),
            str(sensor_id)
        )

        sensor_groups.setdefault(
            sensor_key,
            []
        ).append(reading)

    return sensor_groups
# ==================================================
# 10. VALIDATE NUMERIC SENSOR MEASUREMENTS
# ==================================================

def is_valid_numeric_measurement(value):

    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    return math.isfinite(value)


# ==================================================
# 11. CREATE STANDARD ANOMALY RECORD
# ==================================================

def create_anomaly_record(
    reading,
    parameter,
    anomaly_type,
    severity,
    message,
    recommendation,
    reference_value=None
):

    if not isinstance(reading, dict):
        raise TypeError(
            "Reading must be a dictionary."
        )

    timestamp = reading.get("timestamp")

    return {

        "field_id": reading.get("field_id"),

        "sensor_id": reading.get("sensor_id"),

        "parameter": parameter,

        "observed_value": reading.get(parameter),

        "reference_value": reference_value,

        "timestamp": timestamp,

        "anomaly_type": anomaly_type,

        "severity": severity,

        "status": "ANOMALY",

        "message": message,

        "recommendation": recommendation,

        "detection_method": "TOKUMA Sensor Anomaly Detection"

    }
    # ==================================================
# 12. DETECT SUDDEN CHANGES IN SENSOR MEASUREMENTS
# ==================================================

def detect_sudden_changes(readings):

    anomalies = []

    prepared_readings = prepare_sensor_history(
        readings
    )

    for parameter, threshold in SUDDEN_CHANGE_THRESHOLDS.items():

        previous_reading = None

        for reading in prepared_readings:

            current_value = reading.get(parameter)

            # Ignore missing or invalid measurements.
            # Do not replace the previous valid measurement.

            if not is_valid_numeric_measurement(current_value):
                continue

            if previous_reading is not None:

                previous_value = previous_reading.get(parameter)

                change = abs(
                    current_value - previous_value
                )

                if change >= threshold:

                    anomalies.append(
                        create_anomaly_record(
                            reading=reading,
                            parameter=parameter,
                            anomaly_type="SUDDEN_CHANGE",
                            severity="WARNING",
                            message=(
                                f"{parameter} changed by "
                                f"{change:.2f} units, exceeding "
                                f"the provisional threshold of "
                                f"{threshold:.2f} units."
                            ),
                            recommendation=(
                                "Inspect the sensor, review the "
                                "previous observations, and verify "
                                "the measurement against a field "
                                "observation before taking action."
                            ),
                            reference_value=previous_value
                        )
                    )

            previous_reading = reading

    return anomalies
# ==================================================
# 13. CALCULATE MEDIAN ABSOLUTE DEVIATION (MAD)
# ==================================================

def calculate_median_absolute_deviation(values):

    valid_values = [
        value
        for value in values
        if is_valid_numeric_measurement(value)
    ]

    if not valid_values:
        return None

    median_value = statistics.median(
        valid_values
    )

    absolute_deviations = [
        abs(value - median_value)
        for value in valid_values
    ]

    mad_value = statistics.median(
        absolute_deviations
    )

    return mad_value
# ==================================================
# 14. DETECT STATISTICAL SENSOR OUTLIERS
# ==================================================

def detect_statistical_outliers(readings):

    anomalies = []

    sensor_groups = group_readings_by_sensor(
        readings
    )

    for sensor_key, sensor_readings in sensor_groups.items():

        for parameter in MONITORED_FIELDS:

            historical_values = []

            for reading in sensor_readings:

                current_value = reading.get(parameter)

                if not is_valid_numeric_measurement(current_value):
                    continue

                # Require sufficient historical observations.
                if len(historical_values) >= MINIMUM_HISTORY_FOR_STATISTICS:

                    reference_values = historical_values[
                        -ROLLING_WINDOW_SIZE:
                    ]

                    median_value = statistics.median(
                        reference_values
                    )

                    mad_value = calculate_median_absolute_deviation(
                        reference_values
                    )

                    # Modified MAD score.
                    if mad_value is not None and mad_value > 0:

                        modified_score = (
                            0.6745
                            * abs(current_value - median_value)
                            / mad_value
                        )

                        if modified_score > MAD_ANOMALY_THRESHOLD:

                            anomalies.append(
                                create_anomaly_record(
                                    reading=reading,
                                    parameter=parameter,
                                    anomaly_type="STATISTICAL_OUTLIER",
                                    severity="INFO",
                                    message=(
                                        f"{parameter} is statistically "
                                        f"unusual compared with previous "
                                        f"sensor observations. "
                                        f"Modified MAD score: "
                                        f"{modified_score:.2f}."
                                    ),
                                    recommendation=(
                                        "Inspect the sensor and compare "
                                        "the measurement with independent "
                                        "field observations before "
                                        "accepting it."
                                    ),
                                    reference_value=median_value
                                )
                            )

                # Add the current valid observation only after
                # evaluating it against its historical reference.
                historical_values.append(
                    current_value
                )

    return anomalies
# ==================================================
# 15. DETECT SENSOR FLATLINE CONDITIONS
# ==================================================

def detect_sensor_flatlines(readings):

    anomalies = []

    sensor_groups = group_readings_by_sensor(
        readings
    )

    for sensor_key, sensor_readings in sensor_groups.items():

        for parameter in MONITORED_FIELDS:

            consecutive_readings = []

            previous_value = None

            for reading in sensor_readings:

                current_value = reading.get(parameter)

                if not is_valid_numeric_measurement(current_value):
                    continue

                if previous_value is None or current_value != previous_value:

                    consecutive_readings = [reading]

                else:

                    consecutive_readings.append(reading)

                previous_value = current_value

                if len(consecutive_readings) >= MINIMUM_FLATLINE_READINGS:

                    first_timestamp = consecutive_readings[0][
                        "_normalized_timestamp"
                    ]

                    last_timestamp = consecutive_readings[-1][
                        "_normalized_timestamp"
                    ]

                    duration_seconds = (
                        last_timestamp - first_timestamp
                    ).total_seconds()

                    if duration_seconds >= MINIMUM_FLATLINE_DURATION_SECONDS:

                        anomalies.append(
                            create_anomaly_record(
                                reading=reading,
                                parameter=parameter,
                                anomaly_type="FLATLINE",
                                severity="WARNING",
                                message=(
                                    f"{parameter} remained unchanged "
                                    f"at {current_value} for "
                                    f"{len(consecutive_readings)} "
                                    f"consecutive observations over "
                                    f"{duration_seconds:.0f} seconds."
                                ),
                                recommendation=(
                                    "Check sensor power, connectivity, "
                                    "measurement updates, and physical "
                                    "sensor operation. Verify the "
                                    "measurement independently."
                                ),
                                reference_value=current_value
                            )
                        )

                        # Avoid generating repeated warnings for
                        # every subsequent identical observation.
                        break

    return anomalies
# ==================================================
# 16. DETECT DELAYED SENSOR OBSERVATIONS
# ==================================================

def detect_delayed_sensor_readings(
    readings,
    reference_time=None
):

    anomalies = []

    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    else:
        reference_time = normalize_sensor_timestamp(
            reference_time
        )

        if reference_time is None:
            raise ValueError(
                "Reference time must be a valid timezone-aware timestamp."
            )

    prepared_readings = prepare_sensor_history(
        readings
    )

    sensor_groups = group_readings_by_sensor(
        prepared_readings
    )

    for sensor_key, sensor_readings in sensor_groups.items():

        if not sensor_readings:
            continue

        latest_reading = sensor_readings[-1]

        latest_timestamp = latest_reading[
            "_normalized_timestamp"
        ]

        delay_seconds = (
            reference_time - latest_timestamp
        ).total_seconds()

        if delay_seconds > STALE_READING_SECONDS:

            anomalies.append(
                create_anomaly_record(
                    reading=latest_reading,
                    parameter="timestamp",
                    anomaly_type="DELAYED_READING",
                    severity="WARNING",
                    message=(
                        f"Latest sensor observation is "
                        f"{delay_seconds:.0f} seconds old, "
                        f"exceeding the provisional "
                        f"{STALE_READING_SECONDS}-second threshold."
                    ),
                    recommendation=(
                        "Check sensor power, network connectivity, "
                        "data transmission, and the sensor's "
                        "last successful observation."
                    ),
                    reference_value=(
                        STALE_READING_SECONDS
                    )
                )
            )

    return anomalies
# ==================================================
# 17. VALIDATE ANOMALY SEVERITY
# ==================================================

def classify_anomaly_severity(
    anomaly_type,
    confirmed_critical=False
):

    supported_anomalies = [
        "SUDDEN_CHANGE",
        "STATISTICAL_OUTLIER",
        "FLATLINE",
        "DELAYED_READING"
    ]

    if anomaly_type not in supported_anomalies:

        return {
            "severity": "INFO",
            "status": "NO_ANOMALY",
            "requires_action": False
        }

    if confirmed_critical is True:

        return {
            "severity": "CRITICAL",
            "status": "ANOMALY",
            "requires_action": True
        }

    return {
        "severity": "WARNING",
        "status": "ANOMALY",
        "requires_action": True
    }

# ==================================================
# 18. SUMMARIZE ANOMALY RESULTS
# ==================================================

def summarize_anomaly_results(anomalies):

    if not isinstance(anomalies, (list, tuple)):
        raise TypeError(
            "Anomalies must be provided as a list or tuple."
        )

    severity_counts = {
        "INFO": 0,
        "WARNING": 0,
        "CRITICAL": 0
    }

    anomaly_type_counts = {}

    affected_sensors = set()

    for anomaly in anomalies:

        if not isinstance(anomaly, dict):
            continue

        severity = anomaly.get("severity", "INFO")

        if severity not in VALID_SEVERITY_LEVELS:
            severity = "INFO"

        severity_counts[severity] += 1

        anomaly_type = anomaly.get(
            "anomaly_type",
            "UNKNOWN"
        )

        anomaly_type_counts[anomaly_type] = (
            anomaly_type_counts.get(anomaly_type, 0) + 1
        )

        sensor_id = anomaly.get("sensor_id")

        if sensor_id:
            affected_sensors.add(sensor_id)

    return {
        "total_anomalies": sum(severity_counts.values()),
        "severity_counts": severity_counts,
        "anomaly_type_counts": anomaly_type_counts,
        "affected_sensor_count": len(affected_sensors),
        "affected_sensors": sorted(affected_sensors)
    }

# ==================================================
# 19. RUN INTEGRATED SENSOR ANOMALY DETECTION
# ==================================================

def run_sensor_anomaly_detection(
    readings,
    reference_time=None
):

    if not isinstance(readings, (list, tuple)):
        raise TypeError(
            "Sensor readings must be provided as a list or tuple."
        )

    # Run individual anomaly detection methods
    sudden_change_anomalies = detect_sudden_changes(
        readings
    )

    statistical_anomalies = detect_statistical_outliers(
        readings
    )

    flatline_anomalies = detect_sensor_flatlines(
        readings
    )

    delayed_anomalies = detect_delayed_sensor_readings(
        readings,
        reference_time=reference_time
    )

    # Combine all detected anomalies
    all_anomalies = (
        sudden_change_anomalies
        + statistical_anomalies
        + flatline_anomalies
        + delayed_anomalies
    )

    # Generate anomaly summary
    summary = summarize_anomaly_results(
        all_anomalies
    )

    return {
        "success": True,
        "detection_methods": [
            "SUDDEN_CHANGE",
            "STATISTICAL_OUTLIER",
            "FLATLINE",
            "DELAYED_READING"
        ],
        "anomalies": all_anomalies,
        "summary": summary
    }

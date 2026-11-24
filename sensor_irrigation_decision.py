"""
TOKUMA 3-in-1 Integrated Model
Sensor-Based Irrigation Decision Support

Step: 5.3.11.3

Purpose:
1. Validate the latest sensor observation.
2. Check observation freshness.
3. Reuse existing sensor quality-control functions.
4. Reuse existing sensor anomaly detection.
5. Prevent uncalibrated sensor percentages from being
   interpreted as soil-water depth.
6. Return a transparent irrigation decision status.

Important:
This module provides decision support only.
It does not automatically operate irrigation equipment.
"""

from datetime import datetime, timezone

from sensor_quality_control import evaluate_sensor_reading

from sensor_anomaly_detection import (
    normalize_sensor_timestamp,
    run_sensor_anomaly_detection,
    STALE_READING_SECONDS,
)


# --------------------------------------------------
# 1. DECISION STATUS CONSTANTS
# --------------------------------------------------

INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

SENSOR_REVIEW_REQUIRED = "SENSOR_REVIEW_REQUIRED"

SENSOR_DATA_ACCEPTABLE = "SENSOR_DATA_ACCEPTABLE"

IRRIGATION_ASSESSMENT_READY = "IRRIGATION_ASSESSMENT_READY"


# --------------------------------------------------
# 2. STANDARD DECISION RESULT
# --------------------------------------------------

def create_decision_result(
    reading,
    observation_age_seconds=None,
    quality_status="NOT_ASSESSED",
    anomaly_status="NOT_RUN",
    decision_status=INSUFFICIENT_DATA,
    decision_reasons=None,
    relevant_anomalies=None,
):
    """
    Create a consistent decision-support output.

    Quantitative irrigation recommendations remain unavailable
    until calibrated soil-water and agronomic parameters have
    been implemented and validated.
    """

    reading = reading if isinstance(reading, dict) else {}

    return {
        "field_id": reading.get("field_id"),
        "sensor_id": reading.get("sensor_id"),
        "observation_timestamp": reading.get("timestamp"),
        "observation_age_seconds": observation_age_seconds,
        "quality_status": quality_status,
        "anomaly_status": anomaly_status,
        "decision_status": decision_status,
        "decision_reasons": decision_reasons or [],
        "relevant_anomalies": relevant_anomalies or [],
        "irrigation_amount_mm": None,
        "irrigation_duration_minutes": None,
        "automatic_irrigation_enabled": False,
    }


# --------------------------------------------------
# 3. IDENTIFY ANOMALIES ASSOCIATED WITH LATEST READING
# --------------------------------------------------

def identify_latest_observation_anomalies(
    anomalies,
    latest_reading,
):
    """
    Match anomaly records to the exact latest observation.

    Matching criteria:
    - Same field ID.
    - Same sensor ID.
    - Same normalized observation timestamp.

    Historical anomalies from other observations are excluded.
    """

    latest_timestamp = normalize_sensor_timestamp(
        latest_reading.get("timestamp")
    )

    if latest_timestamp is None:
        return []

    relevant_anomalies = []

    for anomaly in anomalies:

        if not isinstance(anomaly, dict):
            continue

        if anomaly.get("field_id") != latest_reading.get("field_id"):
            continue

        if anomaly.get("sensor_id") != latest_reading.get("sensor_id"):
            continue

        anomaly_timestamp = normalize_sensor_timestamp(
            anomaly.get("timestamp")
        )

        if anomaly_timestamp != latest_timestamp:
            continue

        severity = str(
            anomaly.get("severity", "")
        ).upper()

        if severity in ("WARNING", "CRITICAL"):
            relevant_anomalies.append(anomaly)

    return relevant_anomalies


# --------------------------------------------------
# 4. EVALUATE SENSOR-BASED IRRIGATION DECISION
# --------------------------------------------------

def evaluate_sensor_irrigation_decision(
    latest_reading,
    readings_history=None,
    reference_time=None,
):
    """
    Evaluate whether the latest sensor observation can support
    irrigation decision-making.

    Parameters
    ----------
    latest_reading : dict
        Latest sensor observation.

    readings_history : list or tuple, optional
        Historical observations for the same sensor.

    reference_time : datetime or ISO timestamp, optional
        Time used to calculate observation age.

    Returns
    -------
    dict
        Structured sensor-based irrigation decision result.
    """

    # --------------------------------------------------
    # A. Validate input structure
    # --------------------------------------------------

    if not isinstance(latest_reading, dict):

        return create_decision_result(
            reading={},
            decision_reasons=[
                "Latest sensor observation must be a dictionary."
            ],
        )

    reading = latest_reading.copy()

    # --------------------------------------------------
    # B. Validate required observation information
    # --------------------------------------------------

    required_fields = (
        "field_id",
        "sensor_id",
        "soil_moisture",
        "timestamp",
    )

    missing_fields = [
        field
        for field in required_fields
        if reading.get(field) is None
        or (
            field in ("field_id", "sensor_id")
            and not str(reading.get(field)).strip()
        )
    ]

    if missing_fields:

        return create_decision_result(
            reading=reading,
            decision_reasons=[
                "Required observation information is missing: "
                + ", ".join(missing_fields)
            ],
        )

    # --------------------------------------------------
    # C. Validate timestamp and timezone
    # --------------------------------------------------

    observation_time = normalize_sensor_timestamp(
        reading.get("timestamp")
    )

    if observation_time is None:

        return create_decision_result(
            reading=reading,
            decision_reasons=[
                "Observation timestamp is invalid or timezone-naive."
            ],
        )

    # --------------------------------------------------
    # D. Establish reference time
    # --------------------------------------------------

    if reference_time is None:

        current_time = datetime.now(timezone.utc)

    else:

        current_time = normalize_sensor_timestamp(
            reference_time
        )

        if current_time is None:

            raise ValueError(
                "Reference time must be a valid "
                "timezone-aware timestamp."
            )

    # --------------------------------------------------
    # E. Check future-dated observations
    # --------------------------------------------------

    observation_age_seconds = (
        current_time - observation_time
    ).total_seconds()

    if observation_age_seconds < 0:

        return create_decision_result(
            reading=reading,
            observation_age_seconds=observation_age_seconds,
            decision_reasons=[
                "Observation timestamp is in the future."
            ],
        )

    # --------------------------------------------------
    # F. Check observation freshness
    # --------------------------------------------------

    if observation_age_seconds > STALE_READING_SECONDS:

        return create_decision_result(
            reading=reading,
            observation_age_seconds=observation_age_seconds,
            decision_reasons=[
                "Observation is stale. Its age exceeds the "
                f"provisional {STALE_READING_SECONDS}-second threshold."
            ],
        )

    # --------------------------------------------------
    # G. Evaluate current sensor data quality
    # --------------------------------------------------

    try:

        quality_result = evaluate_sensor_reading(reading)

        if not isinstance(quality_result, dict):
            raise ValueError(
                "Sensor quality-control result is not a dictionary."
            )

        quality_status = quality_result.get(
            "quality_status",
            "REVIEW",
        )

        quality_flags = quality_result.get(
            "quality_flags",
            [],
        )

        if not isinstance(quality_flags, list):
            quality_flags = [
                "Quality-control flags have an invalid format."
            ]

    except Exception as error:

        return create_decision_result(
            reading=reading,
            observation_age_seconds=observation_age_seconds,
            quality_status="REVIEW",
            anomaly_status="NOT_RUN",
            decision_status=SENSOR_REVIEW_REQUIRED,
            decision_reasons=[
                "Sensor quality assessment could not be completed safely.",
                str(error),
            ],
        )

    # --------------------------------------------------
    # H. Reject technically invalid or unexpected quality statuses
    # --------------------------------------------------

    if quality_status not in ("ACCEPTABLE", "INCOMPLETE"):

        return create_decision_result(
            reading=reading,
            observation_age_seconds=observation_age_seconds,
            quality_status=str(quality_status),
            anomaly_status="NOT_RUN",
            decision_status=SENSOR_REVIEW_REQUIRED,
            decision_reasons=[
                "Current observation failed sensor quality checks "
                "or returned an unsupported quality status."
            ] + quality_flags,
        )
    # I. Prepare relevant sensor history
    # --------------------------------------------------

    # Reuse the timestamp already validated in Section C.
    latest_timestamp = observation_time

    sensor_history = []

    if readings_history is None:

        sensor_history = []

    elif isinstance(readings_history, (list, tuple)):

        for item in readings_history:

            if not isinstance(item, dict):
                continue

            if item.get("field_id") != reading.get("field_id"):
                continue

            if item.get("sensor_id") != reading.get("sensor_id"):
                continue

            history_timestamp = normalize_sensor_timestamp(
                item.get("timestamp")
            )

            # Exclude invalid or timezone-naive history timestamps.
            if history_timestamp is None:
                continue

            # Exclude observations later than the current observation.
            if history_timestamp > latest_timestamp:
                continue

            # The supplied latest observation takes precedence
            # over any historical record with the same timestamp.
            if history_timestamp == latest_timestamp:
                continue

            sensor_history.append(item.copy())

    else:

        return create_decision_result(
            reading=reading,
            observation_age_seconds=observation_age_seconds,
            quality_status=quality_status,
            decision_reasons=[
                "Sensor history must be provided as a list or tuple."
            ],
        )

    # Always include the actual latest observation.
    sensor_history.append(reading.copy())
    # J. Run existing anomaly detection
    # --------------------------------------------------

    try:

        anomaly_result = run_sensor_anomaly_detection(
            sensor_history,
            reference_time=current_time,
        )

        if not anomaly_result.get("success"):

            raise RuntimeError(
                "Sensor anomaly detection did not complete successfully."
            )

        anomalies = anomaly_result.get(
            "anomalies",
            [],
        )

        relevant_anomalies = identify_latest_observation_anomalies(
            anomalies,
            reading,
        )

    except Exception as error:

        return create_decision_result(
            reading=reading,
            observation_age_seconds=observation_age_seconds,
            quality_status=quality_status,
            anomaly_status="ERROR",
            decision_status=SENSOR_REVIEW_REQUIRED,
            decision_reasons=[
                "Anomaly detection could not be completed safely.",
                str(error),
            ],
        )

    # --------------------------------------------------
    # K. Apply anomaly safeguards
    # --------------------------------------------------

    if relevant_anomalies:

        anomaly_reasons = [
            (
                f"{item.get('anomaly_type', 'UNKNOWN')} "
                f"({item.get('severity', 'UNKNOWN')}): "
                f"{item.get('message', 'Anomaly detected.')}"
            )
            for item in relevant_anomalies
        ]

        return create_decision_result(
            reading=reading,
            observation_age_seconds=observation_age_seconds,
            quality_status=quality_status,
            anomaly_status="REVIEW_REQUIRED",
            decision_status=SENSOR_REVIEW_REQUIRED,
            decision_reasons=[
                "Relevant warning or critical anomaly detected "
                "for the latest observation."
            ] + anomaly_reasons,
            relevant_anomalies=relevant_anomalies,
        )

    # --------------------------------------------------
    # L. Sensor checks passed; agronomic assessment pending
    # --------------------------------------------------

    decision_reasons = [
        "Latest observation passed the implemented technical "
        "quality and freshness checks.",
        "No matching WARNING or CRITICAL anomaly was detected "
        "for the latest observation.",
        "Soil moisture is currently an uncalibrated percentage.",
        "Calibrated volumetric soil moisture and validated "
        "soil-water retention parameters are not yet available.",
        "A quantitative irrigation requirement cannot yet "
        "be calculated scientifically.",
    ]

    if quality_status == "INCOMPLETE":

        decision_reasons.append(
            "Some optional sensor measurements are missing."
        )

    return create_decision_result(
        reading=reading,
        observation_age_seconds=observation_age_seconds,
        quality_status=quality_status,
        anomaly_status="NO_RELEVANT_ANOMALIES",
        decision_status=SENSOR_DATA_ACCEPTABLE,
        decision_reasons=decision_reasons,
    )


# --------------------------------------------------
# 5. OPERATIONAL SAFETY
# --------------------------------------------------

def is_automatic_irrigation_enabled():

    """
    Automatic irrigation is disabled in this implementation.
    """

    return False

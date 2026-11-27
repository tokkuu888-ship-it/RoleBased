"""
TOKUMA 3-in-1 Integrated Model
PhD Field Observation Database

Step 5.3.12.4.1

Purpose:
- Store actual experimental field observations.
- Link measurements to registered sites and subplots.
- Preserve measurement provenance and verification status.
- Separate actual irrigation events from generated schedules.

Supports SQLite and PostgreSQL.

Transaction ownership remains with the calling application.
This module does not modify the separate sensor database.
"""

from typing import Any
from datetime import datetime, timezone
import math


# ==========================================================
# 1. FIELD OBSERVATION DATABASE SCHEMA
# ==========================================================

FIELD_OBSERVATION_SCHEMA = [

    """
    CREATE TABLE IF NOT EXISTS field_observations (
        observation_id TEXT PRIMARY KEY,

        site_id TEXT,
        subplot_id TEXT,

        observation_type TEXT NOT NULL
            CHECK (
                observation_type IN (
                    'SOIL_MOISTURE',
                    'IRRIGATION',
                    'CROP',
                    'WEATHER',
                    'YIELD',
                    'GENERAL'
                )
            ),

        observed_at TEXT NOT NULL,
        observed_at_original TEXT NOT NULL,

        source_type TEXT NOT NULL
            CHECK (
                source_type IN (
                    'MANUAL',
                    'PHYSICAL_SENSOR',
                    'WEATHER_STATION',
                    'API',
                    'IMPORTED'
                )
            ),

        observer TEXT,
        measurement_method TEXT,
        instrument_id TEXT,
        calibration_reference TEXT,

        verification_status TEXT NOT NULL
            DEFAULT 'RECORDED'
            CHECK (
                verification_status IN (
                    'RECORDED',
                    'REVIEWED',
                    'VERIFIED',
                    'REJECTED'
                )
            ),

        notes TEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        CHECK (
            site_id IS NOT NULL
            OR subplot_id IS NOT NULL
        ),

        FOREIGN KEY (site_id)
            REFERENCES experimental_sites(site_id),

        FOREIGN KEY (subplot_id)
            REFERENCES experimental_subplots(subplot_id)
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS soil_moisture_observations (
        observation_id TEXT PRIMARY KEY,

        moisture_content_percent REAL NOT NULL
            CHECK (moisture_content_percent >= 0),

        moisture_basis TEXT NOT NULL
            CHECK (
                moisture_basis IN (
                    'VOLUMETRIC',
                    'GRAVIMETRIC'
                )
            ),

        measurement_depth_cm REAL NOT NULL
            CHECK (measurement_depth_cm > 0),

        CHECK (
            moisture_basis = 'GRAVIMETRIC'
            OR moisture_content_percent <= 100
        ),

        FOREIGN KEY (observation_id)
            REFERENCES field_observations(observation_id)
            ON DELETE RESTRICT
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS actual_irrigation_events (
        event_id TEXT PRIMARY KEY,

        observation_id TEXT NOT NULL UNIQUE,

        irrigation_start TEXT NOT NULL,
        irrigation_start_original TEXT NOT NULL,
        irrigation_end TEXT,
        irrigation_end_original TEXT,

        applied_depth_mm REAL NOT NULL
            CHECK (applied_depth_mm > 0),

        duration_minutes REAL
            CHECK (
                duration_minutes IS NULL
                OR duration_minutes > 0
            ),

        flow_rate_lps REAL
            CHECK (
                flow_rate_lps IS NULL
                OR flow_rate_lps > 0
            ),

        measured_volume_liters REAL
            CHECK (
                measured_volume_liters IS NULL
                OR measured_volume_liters > 0
            ),

        application_method TEXT,

        verification_method TEXT,

        FOREIGN KEY (observation_id)
            REFERENCES field_observations(observation_id)
            ON DELETE RESTRICT
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS crop_observations (
        observation_id TEXT PRIMARY KEY,

        growth_stage TEXT,

        plant_height_cm REAL
            CHECK (
                plant_height_cm IS NULL
                OR plant_height_cm >= 0
            ),

        canopy_cover_percent REAL
            CHECK (
                canopy_cover_percent IS NULL
                OR (
                    canopy_cover_percent >= 0
                    AND canopy_cover_percent <= 100
                )
            ),

        FOREIGN KEY (observation_id)
            REFERENCES field_observations(observation_id)
            ON DELETE RESTRICT
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS weather_observations (
        observation_id TEXT PRIMARY KEY,

        rainfall_mm REAL
            CHECK (
                rainfall_mm IS NULL
                OR rainfall_mm >= 0
            ),

        temperature_min_c REAL,
        temperature_max_c REAL,

        relative_humidity_percent REAL
            CHECK (
                relative_humidity_percent IS NULL
                OR (
                    relative_humidity_percent >= 0
                    AND relative_humidity_percent <= 100
                )
            ),

        wind_speed_m_s REAL
            CHECK (
                wind_speed_m_s IS NULL
                OR wind_speed_m_s >= 0
            ),

        CHECK (
            temperature_min_c IS NULL
            OR temperature_max_c IS NULL
            OR temperature_min_c <= temperature_max_c
        ),

        FOREIGN KEY (observation_id)
            REFERENCES field_observations(observation_id)
            ON DELETE RESTRICT
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS yield_observations (
        observation_id TEXT PRIMARY KEY,

        harvested_area_m2 REAL NOT NULL
            CHECK (harvested_area_m2 > 0),

        grain_mass_kg REAL NOT NULL
            CHECK (grain_mass_kg >= 0),

        grain_moisture_percent REAL
            CHECK (
                grain_moisture_percent IS NULL
                OR (
                    grain_moisture_percent >= 0
                    AND grain_moisture_percent <= 100
                )
            ),

        yield_basis TEXT NOT NULL
            CHECK (
                yield_basis IN (
                    'AS_HARVESTED',
                    'STANDARDIZED',
                    'NOT_SPECIFIED'
                )
            ),

        FOREIGN KEY (observation_id)
            REFERENCES field_observations(observation_id)
            ON DELETE RESTRICT
    )
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_field_observations_site_time
    ON field_observations(site_id, observed_at)
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_field_observations_subplot_time
    ON field_observations(subplot_id, observed_at)
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_field_observations_type
    ON field_observations(observation_type)
    """,

]


# ==========================================================
# 2. DATABASE INITIALIZATION
# ==========================================================

def initialize_field_observation_schema(
    connection: Any,
    db_type: str
) -> None:
    """
    Initialize field-observation tables using an existing
    SQLite or PostgreSQL connection.

    Transaction ownership remains with the caller.

    This function does not:
    - Commit or roll back.
    - Insert field observations.
    - Modify experimental-design records.
    - Modify the separate sensor database.
    """

    if not isinstance(db_type, str):
        raise TypeError("db_type must be a string.")

    normalized_db_type = db_type.strip().lower()

    if normalized_db_type not in ("sqlite", "postgres"):
        raise ValueError(
            "Unsupported database type. "
            "Expected 'sqlite' or 'postgres'."
        )

    cursor = connection.cursor()

    try:
        if normalized_db_type == "sqlite":

            cursor.execute("PRAGMA foreign_keys")
            result = cursor.fetchone()

            if result is None or result[0] != 1:
                raise RuntimeError(
                    "SQLite foreign keys are not enabled. "
                    "Enable them before initializing the schema."
                )

        for statement in FIELD_OBSERVATION_SCHEMA:
            cursor.execute(statement)

    finally:
        cursor.close()


# ==========================================================
# 3. FIELD OBSERVATION RECORD COUNTS
# ==========================================================

def get_field_observation_counts(
    connection: Any
) -> dict:
    """
    Return record counts for all field-observation tables.

    This function is read-only.
    """

    tables = (
        "field_observations",
        "soil_moisture_observations",
        "actual_irrigation_events",
        "crop_observations",
        "weather_observations",
        "yield_observations",
    )

    cursor = connection.cursor()

    try:
        counts = {}

        for table_name in tables:
            cursor.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            )

            result = cursor.fetchone()
            counts[table_name] = result[0]

        return counts

    finally:
        cursor.close()


# ==========================================================
# 4. DATABASE STRUCTURE VALIDATION
# ==========================================================

def validate_field_observation_schema(
    connection: Any,
    db_type: str
) -> dict:
    """
    Verify that all required field-observation tables exist.

    This function is read-only.
    """

    normalized_db_type = db_type.strip().lower()

    if normalized_db_type not in ("sqlite", "postgres"):
        raise ValueError(
            "db_type must be 'sqlite' or 'postgres'."
        )

    required_tables = {
        "field_observations",
        "soil_moisture_observations",
        "actual_irrigation_events",
        "crop_observations",
        "weather_observations",
        "yield_observations",
    }

    cursor = connection.cursor()

    try:

        if normalized_db_type == "sqlite":

            cursor.execute("""
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
            """)

        else:

            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = current_schema()
                AND table_type = 'BASE TABLE'
            """)

        existing_tables = {
            row[0] for row in cursor.fetchall()
        }

        missing_tables = sorted(
            required_tables - existing_tables
        )

        errors = []

        if missing_tables:
            errors.append(
                "Missing tables: " + ", ".join(missing_tables)
            )

        return {
            "valid": len(errors) == 0,
            "required_tables": sorted(required_tables),
            "existing_tables": sorted(
                required_tables & existing_tables
            ),
            "missing_tables": missing_tables,
            "errors": errors,
        }

    finally:
        cursor.close()
# ==========================================================
# 5. RECORDING AND VALIDATION HELPERS
# ==========================================================





def _normalize_timestamp_utc(value: str, field_name: str):
    """
    Validate timezone-aware ISO 8601 timestamps.

    Return canonical UTC text, original input text,
    and the timezone-aware UTC datetime.
    """
    original_value = _require_text(value, field_name)

    try:
        parsed = datetime.fromisoformat(
            original_value.replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError(
            f"{field_name} must be a valid ISO 8601 timestamp."
        ) from error

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            f"{field_name} must include a timezone offset, "
            "such as +03:00 or Z."
        )

    utc_datetime = parsed.astimezone(timezone.utc)

    return (
        utc_datetime.isoformat(),
        original_value,
        utc_datetime
    )


def _normalize_observation_db_type(db_type: str) -> str:
    if not isinstance(db_type, str):
        raise TypeError("db_type must be a string.")

    normalized = db_type.strip().lower()

    if normalized not in ("sqlite", "postgres"):
        raise ValueError("db_type must be 'sqlite' or 'postgres'.")

    return normalized


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")

    return value.strip()


def _validate_number(
    value: Any,
    field_name: str,
    minimum: float = None,
    maximum: float = None,
    allow_none: bool = False
):
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} is required.")

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a numeric value.")

    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite.")

    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")

    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must not exceed {maximum}.")

    return value


def _execute_field_observation_statement(
    connection: Any,
    db_type: str,
    sql: str,
    parameters: tuple = ()
) -> None:
    normalized = _normalize_observation_db_type(db_type)

    placeholder = "?" if normalized == "sqlite" else "%s"
    database_sql = sql.replace("?", placeholder)

    cursor = connection.cursor()

    try:
        cursor.execute(database_sql, parameters)
    finally:
        cursor.close()


def _validate_measurement_parent(
    connection: Any,
    db_type: str,
    observation_id: str,
    expected_type: str
) -> None:
    """
    Ensure a measurement is attached to an observation
    of the correct observation_type.
    """

    normalized = _normalize_observation_db_type(db_type)
    placeholder = "?" if normalized == "sqlite" else "%s"

    cursor = connection.cursor()

    try:
        cursor.execute(
            f"""
            SELECT observation_type
            FROM field_observations
            WHERE observation_id = {placeholder}
            """,
            (observation_id,)
        )

        result = cursor.fetchone()

        if result is None:
            raise ValueError(
                f"Observation {observation_id} does not exist."
            )

        actual_type = result[0]

        if actual_type != expected_type:
            raise ValueError(
                f"Observation type mismatch: expected "
                f"{expected_type}, found {actual_type}."
            )

    finally:
        cursor.close()


# ==========================================================
# 6. REGISTER COMMON FIELD OBSERVATION
# ==========================================================

def record_field_observation(
    connection: Any,
    db_type: str,
    observation_id: str,
    observation_type: str,
    observed_at: str,
    source_type: str,
    site_id: str = None,
    subplot_id: str = None,
    observer: str = None,
    measurement_method: str = None,
    instrument_id: str = None,
    calibration_reference: str = None,
    notes: str = None
) -> str:
    """
    Register the common metadata for one field observation.

    If subplot_id is supplied, its registered site is checked.
    The site_id is inferred from the subplot when omitted.

    Does not commit or roll back.
    """

    _normalize_observation_db_type(db_type)

    observation_id = _require_text(
        observation_id, "observation_id"
    )

    observation_type = _require_text(
        observation_type, "observation_type"
    ).upper()

    observed_at, observed_at_original, _ = _normalize_timestamp_utc(
        observed_at, "observed_at"
    )

    source_type = _require_text(
        source_type, "source_type"
    ).upper()


    allowed_types = {
        "SOIL_MOISTURE",
        "IRRIGATION",
        "CROP",
        "WEATHER",
        "YIELD",
        "GENERAL",
    }

    allowed_sources = {
        "MANUAL",
        "PHYSICAL_SENSOR",
        "WEATHER_STATION",
        "API",
        "IMPORTED",
    }


    if observation_type not in allowed_types:
        raise ValueError("Unsupported observation_type.")

    if source_type not in allowed_sources:
        raise ValueError("Unsupported source_type.")


    if site_id is not None:
        site_id = _require_text(site_id, "site_id")

    if subplot_id is not None:
        subplot_id = _require_text(subplot_id, "subplot_id")

    if site_id is None and subplot_id is None:
        raise ValueError(
            "At least one of site_id or subplot_id is required."
        )

    normalized = _normalize_observation_db_type(db_type)
    placeholder = "?" if normalized == "sqlite" else "%s"

    cursor = connection.cursor()

    try:
        if subplot_id is not None:

            cursor.execute(
                f"""
                SELECT s.site_id
                FROM experimental_subplots sp
                JOIN experimental_main_plots mp
                    ON sp.main_plot_id = mp.main_plot_id
                JOIN experimental_blocks b
                    ON mp.block_id = b.block_id
                JOIN experimental_sites s
                    ON b.site_id = s.site_id
                WHERE sp.subplot_id = {placeholder}
                """,
                (subplot_id,)
            )

            result = cursor.fetchone()

            if result is None:
                raise ValueError(
                    f"Unknown experimental subplot: {subplot_id}"
                )

            registered_site_id = result[0]

            if site_id is None:
                site_id = registered_site_id

            elif site_id != registered_site_id:
                raise ValueError(
                    "The supplied site_id does not match "
                    "the registered site's subplot."
                )

        cursor.execute(
            f"""
            INSERT INTO field_observations (
                observation_id,
                site_id,
                subplot_id,
                observation_type,
                observed_at,
                observed_at_original,
                source_type,
                observer,
                measurement_method,
                instrument_id,
                calibration_reference,
                verification_status,
                notes
            )
            VALUES (
                {placeholder}, {placeholder}, {placeholder},
                {placeholder}, {placeholder}, {placeholder},
                {placeholder}, {placeholder}, {placeholder},
                {placeholder}, {placeholder}, {placeholder},
                {placeholder}
            )
            """,
            (
                observation_id,
                site_id,
                subplot_id,
                observation_type,
                observed_at,
                observed_at_original,
                source_type,
                observer,
                measurement_method,
                instrument_id,
                calibration_reference,
                "RECORDED",
                notes,
            )
        )

        return observation_id

    finally:
        cursor.close()


# ==========================================================
# 7. SOIL MOISTURE RECORDING
# ==========================================================

def record_soil_moisture_observation(
    connection: Any,
    db_type: str,
    observation_id: str,
    moisture_content_percent: float,
    moisture_basis: str,
    measurement_depth_cm: float
) -> str:

    observation_id = _require_text(
        observation_id, "observation_id"
    )

    moisture_basis = _require_text(
        moisture_basis, "moisture_basis"
    ).upper()

    if moisture_basis not in {"VOLUMETRIC", "GRAVIMETRIC"}:
        raise ValueError(
            "moisture_basis must be VOLUMETRIC or GRAVIMETRIC."
        )

    moisture_content_percent = _validate_number(
        moisture_content_percent,
        "moisture_content_percent",
        minimum=0,
        maximum=(
            100 if moisture_basis == "VOLUMETRIC" else None
        )
    )

    measurement_depth_cm = _validate_number(
        measurement_depth_cm,
        "measurement_depth_cm",
        minimum=0.000001
    )

    _validate_measurement_parent(
        connection,
        db_type,
        observation_id,
        "SOIL_MOISTURE"
    )

    _execute_field_observation_statement(
        connection,
        db_type,
        """
        INSERT INTO soil_moisture_observations (
            observation_id,
            moisture_content_percent,
            moisture_basis,
            measurement_depth_cm
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            observation_id,
            moisture_content_percent,
            moisture_basis,
            measurement_depth_cm,
        )
    )

    return observation_id


# ==========================================================
# 8. ACTUAL IRRIGATION EVENT RECORDING
# ==========================================================

def record_actual_irrigation_event(
    connection: Any,
    db_type: str,
    event_id: str,
    observation_id: str,
    irrigation_start: str,
    applied_depth_mm: float,
    irrigation_end: str = None,
    duration_minutes: float = None,
    flow_rate_lps: float = None,
    measured_volume_liters: float = None,
    application_method: str = None,
    verification_method: str = None
) -> str:
    """
    Record an actual irrigation event.

    Input timestamps must include timezone information.
    Stored timestamps are normalized to UTC. Original
    timestamp strings are retained for provenance.

    Transaction ownership remains with the caller.
    """

    event_id = _require_text(event_id, "event_id")
    observation_id = _require_text(
        observation_id, "observation_id"
    )

    (
        irrigation_start,
        irrigation_start_original,
        start_datetime
    ) = _normalize_timestamp_utc(
        irrigation_start, "irrigation_start"
    )

    irrigation_end_original = None
    end_datetime = None

    if irrigation_end is not None:
        (
            irrigation_end,
            irrigation_end_original,
            end_datetime
        ) = _normalize_timestamp_utc(
            irrigation_end, "irrigation_end"
        )

        if end_datetime < start_datetime:
            raise ValueError(
                "irrigation_end cannot precede irrigation_start."
            )

    applied_depth_mm = _validate_number(
        applied_depth_mm,
        "applied_depth_mm",
        minimum=0.000001
    )

    duration_minutes = _validate_number(
        duration_minutes,
        "duration_minutes",
        minimum=0.000001,
        allow_none=True
    )

    flow_rate_lps = _validate_number(
        flow_rate_lps,
        "flow_rate_lps",
        minimum=0.000001,
        allow_none=True
    )

    measured_volume_liters = _validate_number(
        measured_volume_liters,
        "measured_volume_liters",
        minimum=0.000001,
        allow_none=True
    )

    _validate_measurement_parent(
        connection,
        db_type,
        observation_id,
        "IRRIGATION"
    )

    _execute_field_observation_statement(
        connection,
        db_type,
        """
        INSERT INTO actual_irrigation_events (
            event_id,
            observation_id,
            irrigation_start,
            irrigation_start_original,
            irrigation_end,
            irrigation_end_original,
            applied_depth_mm,
            duration_minutes,
            flow_rate_lps,
            measured_volume_liters,
            application_method,
            verification_method
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            observation_id,
            irrigation_start,
            irrigation_start_original,
            irrigation_end,
            irrigation_end_original,
            applied_depth_mm,
            duration_minutes,
            flow_rate_lps,
            measured_volume_liters,
            application_method,
            verification_method,
        )
    )

    return event_id


# ==========================================================
# 9. CROP OBSERVATION RECORDING
# ==========================================================

def record_crop_observation(
    connection: Any,
    db_type: str,
    observation_id: str,
    growth_stage: str = None,
    plant_height_cm: float = None,
    canopy_cover_percent: float = None
) -> str:

    observation_id = _require_text(
        observation_id, "observation_id"
    )

    plant_height_cm = _validate_number(
        plant_height_cm,
        "plant_height_cm",
        minimum=0,
        allow_none=True
    )

    canopy_cover_percent = _validate_number(
        canopy_cover_percent,
        "canopy_cover_percent",
        minimum=0,
        maximum=100,
        allow_none=True
    )

    if growth_stage is not None:
        growth_stage = _require_text(
            growth_stage, "growth_stage"
        )

    _validate_measurement_parent(
        connection,
        db_type,
        observation_id,
        "CROP"
    )

    _execute_field_observation_statement(
        connection,
        db_type,
        """
        INSERT INTO crop_observations (
            observation_id,
            growth_stage,
            plant_height_cm,
            canopy_cover_percent
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            observation_id,
            growth_stage,
            plant_height_cm,
            canopy_cover_percent,
        )
    )

    return observation_id


# ==========================================================
# 10. WEATHER OBSERVATION RECORDING
# ==========================================================

def record_weather_observation(
    connection: Any,
    db_type: str,
    observation_id: str,
    rainfall_mm: float = None,
    temperature_min_c: float = None,
    temperature_max_c: float = None,
    relative_humidity_percent: float = None,
    wind_speed_m_s: float = None
) -> str:

    observation_id = _require_text(
        observation_id, "observation_id"
    )

    rainfall_mm = _validate_number(
        rainfall_mm, "rainfall_mm",
        minimum=0, allow_none=True
    )

    temperature_min_c = _validate_number(
        temperature_min_c, "temperature_min_c",
        allow_none=True
    )

    temperature_max_c = _validate_number(
        temperature_max_c, "temperature_max_c",
        allow_none=True
    )

    relative_humidity_percent = _validate_number(
        relative_humidity_percent,
        "relative_humidity_percent",
        minimum=0,
        maximum=100,
        allow_none=True
    )

    wind_speed_m_s = _validate_number(
        wind_speed_m_s, "wind_speed_m_s",
        minimum=0, allow_none=True
    )

    if (
        temperature_min_c is not None
        and temperature_max_c is not None
        and temperature_min_c > temperature_max_c
    ):
        raise ValueError(
            "Minimum temperature cannot exceed maximum temperature."
        )

    _validate_measurement_parent(
        connection,
        db_type,
        observation_id,
        "WEATHER"
    )

    _execute_field_observation_statement(
        connection,
        db_type,
        """
        INSERT INTO weather_observations (
            observation_id,
            rainfall_mm,
            temperature_min_c,
            temperature_max_c,
            relative_humidity_percent,
            wind_speed_m_s
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            observation_id,
            rainfall_mm,
            temperature_min_c,
            temperature_max_c,
            relative_humidity_percent,
            wind_speed_m_s,
        )
    )

    return observation_id


# ==========================================================
# 11. YIELD OBSERVATION RECORDING
# ==========================================================

def record_yield_observation(
    connection: Any,
    db_type: str,
    observation_id: str,
    harvested_area_m2: float,
    grain_mass_kg: float,
    yield_basis: str,
    grain_moisture_percent: float = None
) -> str:

    observation_id = _require_text(
        observation_id, "observation_id"
    )

    harvested_area_m2 = _validate_number(
        harvested_area_m2,
        "harvested_area_m2",
        minimum=0.000001
    )

    grain_mass_kg = _validate_number(
        grain_mass_kg,
        "grain_mass_kg",
        minimum=0
    )

    grain_moisture_percent = _validate_number(
        grain_moisture_percent,
        "grain_moisture_percent",
        minimum=0,
        maximum=100,
        allow_none=True
    )

    yield_basis = _require_text(
        yield_basis, "yield_basis"
    ).upper()

    if yield_basis not in {
        "AS_HARVESTED",
        "STANDARDIZED",
        "NOT_SPECIFIED",
    }:
        raise ValueError("Unsupported yield_basis.")

    _validate_measurement_parent(
        connection,
        db_type,
        observation_id,
        "YIELD"
    )

    _execute_field_observation_statement(
        connection,
        db_type,
        """
        INSERT INTO yield_observations (
            observation_id,
            harvested_area_m2,
            grain_mass_kg,
            grain_moisture_percent,
            yield_basis
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            observation_id,
            harvested_area_m2,
            grain_mass_kg,
            grain_moisture_percent,
            yield_basis,
        )
    )

    return observation_id


# ==========================================================
# END OF STEP 5.3.12.4.3 RECORDING FUNCTIONS
# ==========================================================


"""
TOKUMA 3-in-1 Integrated Model
Edge Computing - Persistent Sensor API

Features:
1. Receive agricultural sensor measurements.
2. Validate incoming measurements.
3. Save measurements permanently in SQLite.
4. Retrieve the latest measurement.
5. Retrieve complete sensor history.
"""

from datetime import datetime, timezone
from typing import Optional
import math

from fastapi import FastAPI, HTTPException
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator
)

from sensor_database import (
    save_sensor_reading,
    get_latest_reading,
    get_all_readings
)
from sensor_quality_control import (
    evaluate_sensor_reading,
    log_quality_result
)

# ==================================================
# 1. INITIALIZE FASTAPI
# ==================================================

app = FastAPI(
    title="TOKUMA Agricultural Sensor API",
    description="Edge Computing interface for receiving, validating and storing agricultural sensor measurements.",
    version="1.1.0"
)


# ==================================================
# 2. SENSOR DATA MODEL
# ==================================================


class SensorReading(BaseModel):

    model_config = ConfigDict(
        str_strip_whitespace=True,
        allow_inf_nan=False
    )

    field_id: str = Field(
        min_length=1,
        max_length=50
    )

    sensor_id: str = Field(
        min_length=1,
        max_length=50
    )

    soil_moisture: float = Field(
        ge=0,
        le=100
    )

    air_temperature: Optional[float] = Field(
        default=None,
        ge=-50,
        le=70
    )

    relative_humidity: Optional[float] = Field(
        default=None,
        ge=0,
        le=100
    )

    rainfall: Optional[float] = Field(
        default=None,
        ge=0
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("field_id", "sensor_id")
    @classmethod
    def validate_identifiers(cls, value):

        if not value.strip():
            raise ValueError(
                "Field ID and Sensor ID cannot be empty."
            )

        return value

    @field_validator(
        "soil_moisture",
        "air_temperature",
        "relative_humidity",
        "rainfall",
        mode="before"
    )
    @classmethod
    def validate_numeric_values(cls, value):

        if value is None:
            return value

        if isinstance(value, bool):
            raise ValueError(
                "Boolean values are not valid sensor measurements."
            )

        if isinstance(value, (int, float)):

            if not math.isfinite(value):
                raise ValueError(
                    "Sensor measurements must be finite numbers."
                )

        return value



# ==================================================
# 3. HOME ENDPOINT
# ==================================================

@app.get("/")
def home():

    return {
        "system": "TOKUMA 3-in-1 Integrated Model",
        "module": "Edge Computing Sensor Listener",
        "status": "running",
        "storage": "SQLite persistent database"
    }


# ==================================================
# 4. HEALTH CHECK
# ==================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "TOKUMA Agricultural Sensor API",
        "database": "SQLite"
    }


# ==================================================
# 5. RECEIVE AND SAVE SENSOR DATA
# ==================================================

@app.post("/sensor-data")
def receive_sensor_data(reading: SensorReading):

    try:
        # 1. Convert validated sensor data into a dictionary
        reading_data = reading.model_dump(mode="json")

        # 2. Assess sensor data quality
        quality_result = evaluate_sensor_reading(reading_data)

        # 3. Save the measurement permanently in SQLite
        record_id = save_sensor_reading(reading_data)

        # 4. Record quality assessment after successful saving
        quality_log_status = "success"
        quality_log_message = None

        try:
            log_quality_result(reading_data, quality_result)

        except Exception as log_error:
            quality_log_status = "failed"
            quality_log_message = str(log_error)

        # 5. Return sensor data and quality results
        return {
            "status": "success",
            "message": "Sensor reading saved permanently",
            "record_id": record_id,
            "field_id": reading.field_id,
            "sensor_id": reading.sensor_id,
            "soil_moisture": reading.soil_moisture,
            "quality_status": quality_result["quality_status"],
            "quality_flags": quality_result["quality_flags"],
            "quality_flag_count": quality_result["flag_count"],
            "quality_log_status": quality_log_status,
            "quality_log_message": quality_log_message,
            "received_at": datetime.now(
                timezone.utc
            ).isoformat()
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Could not save sensor reading: {str(error)}"
        )

# ==================================================
# 6. RETRIEVE LATEST SENSOR READING
# ==================================================

@app.get("/latest/{field_id}")
def get_latest_sensor_reading(field_id: str):

    reading = get_latest_reading(field_id)

    if reading is None:

        raise HTTPException(
            status_code=404,
            detail="No sensor reading found for this field"
        )

    return reading


# ==================================================
# 7. RETRIEVE COMPLETE SENSOR HISTORY
# ==================================================

@app.get("/history/{field_id}")
def get_sensor_history(field_id: str):

    readings = get_all_readings(field_id)

    if not readings:

        raise HTTPException(
            status_code=404,
            detail="No sensor history found for this field"
        )

    return {
        "field_id": field_id,
        "total_readings": len(readings),
        "readings": readings
    }


# ==================================================
# 8. STARTUP MESSAGE
# ==================================================

@app.on_event("startup")
def startup_event():

    print("---------------------------------------")
    print("TOKUMA AGRICULTURAL SENSOR API")
    print("API STATUS: RUNNING")
    print("DATABASE: SQLITE")
    print("SENSOR HISTORY: ENABLED")
    print("---------------------------------------")

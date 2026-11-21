
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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sensor_database import (
    save_sensor_reading,
    get_latest_reading,
    get_all_readings
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

    field_id: str = Field(min_length=1, max_length=50)

    sensor_id: str = Field(min_length=1, max_length=50)

    soil_moisture: float = Field(ge=0, le=100)

    air_temperature: Optional[float] = Field(
        default=None, ge=-50, le=70
    )

    relative_humidity: Optional[float] = Field(
        default=None, ge=0, le=100
    )

    rainfall: Optional[float] = Field(
        default=None, ge=0
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


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

        reading_data = reading.model_dump(mode="json")

        record_id = save_sensor_reading(reading_data)

        return {
            "status": "success",
            "message": "Sensor reading saved permanently",
            "record_id": record_id,
            "field_id": reading.field_id,
            "sensor_id": reading.sensor_id,
            "soil_moisture": reading.soil_moisture,
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

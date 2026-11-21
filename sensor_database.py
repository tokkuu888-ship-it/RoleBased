
"""
TOKUMA 3-in-1 Integrated Model
Persistent Agricultural Sensor Database

Purpose:
1. Create a separate SQLite database.
2. Store incoming agricultural sensor measurements.
3. Retrieve the latest measurement for each field.
4. Retrieve the complete sensor history for each field.
"""

import sqlite3
from pathlib import Path


# --------------------------------------------------
# 1. Database configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DB_PATH = PROJECT_DIR / "tokuma_sensor_data.db"


# --------------------------------------------------
# 2. Database connection
# --------------------------------------------------

def get_connection():

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    return conn


# --------------------------------------------------
# 3. Initialize database and create sensor table
# --------------------------------------------------

def initialize_database():

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                field_id TEXT NOT NULL,

                sensor_id TEXT NOT NULL,

                soil_moisture REAL NOT NULL,

                air_temperature REAL,

                relative_humidity REAL,

                rainfall REAL,

                timestamp TEXT NOT NULL

            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_field_timestamp
            ON sensor_readings (field_id, timestamp)
        """)

    print("TOKUMA sensor database initialized successfully.")


# --------------------------------------------------
# 4. Save sensor measurement
# --------------------------------------------------

def save_sensor_reading(reading):

    with get_connection() as conn:

        cursor = conn.execute("""

            INSERT INTO sensor_readings (

                field_id,
                sensor_id,
                soil_moisture,
                air_temperature,
                relative_humidity,
                rainfall,
                timestamp

            )

            VALUES (?, ?, ?, ?, ?, ?, ?)

        """, (

            reading["field_id"],

            reading["sensor_id"],

            reading["soil_moisture"],

            reading.get("air_temperature"),

            reading.get("relative_humidity"),

            reading.get("rainfall"),

            reading["timestamp"]

        ))

        record_id = cursor.lastrowid

        return record_id


# --------------------------------------------------
# 5. Retrieve latest sensor measurement
# --------------------------------------------------

def get_latest_reading(field_id):

    with get_connection() as conn:

        row = conn.execute("""

            SELECT *

            FROM sensor_readings

            WHERE field_id = ?

            ORDER BY timestamp DESC, id DESC

            LIMIT 1

        """, (field_id,)).fetchone()

        if row is None:

            return None

        return dict(row)


# --------------------------------------------------
# 6. Retrieve complete sensor history
# --------------------------------------------------

def get_all_readings(field_id):

    with get_connection() as conn:

        rows = conn.execute("""

            SELECT *

            FROM sensor_readings

            WHERE field_id = ?

            ORDER BY timestamp DESC, id DESC

        """, (field_id,)).fetchall()

        return [dict(row) for row in rows]


# --------------------------------------------------
# 7. Count sensor records
# --------------------------------------------------

def count_sensor_readings(field_id=None):

    with get_connection() as conn:

        if field_id is None:

            row = conn.execute("""

                SELECT COUNT(*) AS total

                FROM sensor_readings

            """).fetchone()

        else:

            row = conn.execute("""

                SELECT COUNT(*) AS total

                FROM sensor_readings

                WHERE field_id = ?

            """, (field_id,)).fetchone()

        return row["total"]


# --------------------------------------------------
# 8. Initialize database when module is imported
# --------------------------------------------------

initialize_database()

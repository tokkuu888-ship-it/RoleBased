
import json
import time
import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:8000/sensor-data"

# Simulated measurements for software testing only.
FIELD_DATA = {
    "MEL-001": {
        "sensor_id": "SENSOR-MEL-001",
        "readings": [
            (24.5, 28.4, 52.0, 0.0),
            (24.2, 28.7, 51.0, 0.0),
            (23.9, 29.0, 50.0, 0.2),
            (23.6, 29.2, 49.0, 0.0),
        ]
    },
    "HOL-001": {
        "sensor_id": "SENSOR-HOL-001",
        "readings": [
            (27.5, 22.4, 60.0, 0.0),
            (27.2, 22.7, 59.0, 0.0),
            (26.9, 23.0, 58.0, 0.5),
            (26.6, 23.2, 57.0, 0.0),
        ]
    }
}

def send_sensor_reading(field_id, sensor_info, index):
    values = sensor_info["readings"]
    soil, temperature, humidity, rainfall = values[
        index % len(values)
    ]

    payload = {
        "field_id": field_id,
        "sensor_id": sensor_info["sensor_id"],
        "soil_moisture": soil,
        "air_temperature": temperature,
        "relative_humidity": humidity,
        "rainfall": rainfall
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))

            print(
                f"SUCCESS | {field_id} | "
                f"Soil moisture: {soil}% | "
                f"Temperature: {temperature}°C | "
                f"HTTP: {response.status}"
            )

    except urllib.error.URLError as error:
        print(f"CONNECTION ERROR | {field_id} | {error}")


def main():
    print("=" * 55)
    print("TOKUMA AGRICULTURAL SENSOR SIMULATOR")
    print("SIMULATED DATA - NOT REAL FIELD OBSERVATIONS")
    print("=" * 55)

    index = 0

    try:
        while True:
            print(f"\nTransmission cycle {index + 1}")

            for field_id, sensor_info in FIELD_DATA.items():
                send_sensor_reading(
                    field_id,
                    sensor_info,
                    index
                )

            index += 1

            print("\nNext transmission in 15 seconds...")
            time.sleep(15)

    except KeyboardInterrupt:
        print("\nSensor simulation stopped by user.")
        print("Previously submitted readings remain in SQLite.")


if __name__ == "__main__":
    main()

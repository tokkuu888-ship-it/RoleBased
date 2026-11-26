"""
TOKUMA 3-in-1 Integrated Model
PhD Experimental Design Database

Purpose:
1. Define the experimental sites.
2. Define experimental blocks (replications).
3. Define irrigation-treatment main plots.
4. Define crop subplots.
5. Register verified sensor deployments.

Supports SQLite and PostgreSQL.

Important:
- Does not insert experimental observations.
- Does not generate physical plot randomization.
- Does not modify the existing sensor database.
"""

from typing import Any


# ==================================================
# 1. EXPERIMENTAL DATABASE SCHEMA
# ==================================================

EXPERIMENTAL_SCHEMA = [

    """
    CREATE TABLE IF NOT EXISTS experimental_sites (
        site_id TEXT PRIMARY KEY,
        site_code TEXT NOT NULL UNIQUE,
        site_name TEXT NOT NULL,
        region TEXT,
        latitude REAL,
        longitude REAL,
        altitude_m REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS experimental_blocks (
        block_id TEXT PRIMARY KEY,
        site_id TEXT NOT NULL,
        replication_number INTEGER NOT NULL
            CHECK (replication_number BETWEEN 1 AND 3),

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (site_id)
            REFERENCES experimental_sites(site_id),

        UNIQUE (site_id, replication_number)
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS experimental_main_plots (
        main_plot_id TEXT PRIMARY KEY,
        block_id TEXT NOT NULL,

        irrigation_treatment TEXT NOT NULL
            CHECK (
                irrigation_treatment IN (
                    'FIXED',
                    'SOIL_MOISTURE',
                    'ET_BASED'
                )
            ),

        main_plot_number INTEGER NOT NULL
            CHECK (main_plot_number BETWEEN 1 AND 3),

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (block_id)
            REFERENCES experimental_blocks(block_id),

        UNIQUE (block_id, irrigation_treatment),
        UNIQUE (block_id, main_plot_number)
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS experimental_subplots (
        subplot_id TEXT PRIMARY KEY,
        main_plot_id TEXT NOT NULL,

        crop_code TEXT NOT NULL
            CHECK (
                crop_code IN (
                    'MAIZE',
                    'WHEAT',
                    'TEFF'
                )
            ),

        subplot_number INTEGER NOT NULL
            CHECK (subplot_number BETWEEN 1 AND 3),

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (main_plot_id)
            REFERENCES experimental_main_plots(main_plot_id),

        UNIQUE (main_plot_id, crop_code),
        UNIQUE (main_plot_id, subplot_number)
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS sensor_deployments (
        deployment_id TEXT PRIMARY KEY,

        sensor_id TEXT NOT NULL,
        field_id TEXT NOT NULL,
        subplot_id TEXT NOT NULL,

        sensor_type TEXT,
        measurement_variable TEXT,

        measurement_depth_cm REAL
            CHECK (
                measurement_depth_cm IS NULL
                OR measurement_depth_cm > 0
            ),

        planned_installation_date TEXT,
        installation_timestamp TEXT,
        removal_timestamp TEXT,

        deployment_status TEXT NOT NULL
            CHECK (
                deployment_status IN (
                    'PLANNED',
                    'ACTIVE',
                    'COMPLETED',
                    'CANCELLED',
                    'UNVERIFIED'
                )
            ),

        calibration_reference TEXT,
        notes TEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        CHECK (
            (
                deployment_status = 'PLANNED'
                AND installation_timestamp IS NULL
                AND removal_timestamp IS NULL
            )
            OR
            (
                deployment_status = 'ACTIVE'
                AND installation_timestamp IS NOT NULL
                AND removal_timestamp IS NULL
            )
            OR
            (
                deployment_status = 'COMPLETED'
                AND installation_timestamp IS NOT NULL
                AND removal_timestamp IS NOT NULL
            )
            OR
            (
                deployment_status IN ('CANCELLED', 'UNVERIFIED')
                AND installation_timestamp IS NULL
                AND removal_timestamp IS NULL
            )
        ),

        FOREIGN KEY (subplot_id)
            REFERENCES experimental_subplots(subplot_id)
    )
    """,
]


# ==================================================
# 2. DATABASE INITIALIZATION
# ==================================================

def initialize_experimental_design_schema(
    connection: Any,
    db_type: str
) -> None:
    """
    Create experimental-design tables using an existing connection.

    Transaction ownership remains with the calling application.
    This function does not commit or roll back the connection.

    SQLite foreign keys must already be enabled before a transaction
    begins. The function verifies this requirement.
    """

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
                    "Enable them before starting a transaction."
                )

        for statement in EXPERIMENTAL_SCHEMA:
            cursor.execute(statement)

    finally:
        cursor.close()


# ==================================================
# 3. EXPERIMENTAL DESIGN RECORD COUNTS
# ==================================================

def get_experimental_design_counts(
    connection: Any
) -> dict:
    """
    Return the number of records in each experimental
    registry table.

    An empty registry should return zero for all tables.
    """

    tables = (
        "experimental_sites",
        "experimental_blocks",
        "experimental_main_plots",
        "experimental_subplots",
        "sensor_deployments",
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

# ==================================================


# ==================================================
# 4. COMPLETE EXPERIMENTAL DESIGN VALIDATION
# ==================================================

def validate_experimental_design(
    connection: Any,
    db_type: str
) -> dict:
    """
    Validate the complete two-site split-plot design.

    Expected:
    - 2 research sites: MARC and HARC
    - 3 blocks per site
    - 3 irrigation main plots per block
    - 3 crop subplots per main plot

    Supports SQLite and PostgreSQL.
    Does not modify experimental records.
    """

    normalized_db_type = db_type.strip().lower()

    if normalized_db_type not in ("sqlite", "postgres"):
        raise ValueError(
            "db_type must be 'sqlite' or 'postgres'."
        )

    placeholder = "?" if normalized_db_type == "sqlite" else "%s"

    expected_sites = {"MARC", "HARC"}

    expected_treatments = {
        "FIXED",
        "SOIL_MOISTURE",
        "ET_BASED",
    }

    expected_crops = {
        "MAIZE",
        "WHEAT",
        "TEFF",
    }

    errors = []
    cursor = connection.cursor()

    try:
        cursor.execute("""
            SELECT site_id, site_code
            FROM experimental_sites
        """)

        sites = cursor.fetchall()

        actual_site_codes = {
            row[1] for row in sites
        }

        if actual_site_codes != expected_sites:
            errors.append(
                "Expected exactly MARC and HARC as research sites."
            )

        if len(sites) != 2:
            errors.append(
                f"Expected 2 sites; found {len(sites)}."
            )

        for site_id, site_code in sites:

            cursor.execute(
                f"""
                SELECT block_id, replication_number
                FROM experimental_blocks
                WHERE site_id = {placeholder}
                """,
                (site_id,)
            )

            blocks = cursor.fetchall()

            replication_numbers = {
                row[1] for row in blocks
            }

            if len(blocks) != 3:
                errors.append(
                    f"{site_code}: expected 3 blocks; "
                    f"found {len(blocks)}."
                )

            if replication_numbers != {1, 2, 3}:
                errors.append(
                    f"{site_code}: replication numbers "
                    "must be 1, 2, and 3."
                )

            for block_id, replication_number in blocks:

                cursor.execute(
                    f"""
                    SELECT main_plot_id,
                           irrigation_treatment,
                           main_plot_number
                    FROM experimental_main_plots
                    WHERE block_id = {placeholder}
                    """,
                    (block_id,)
                )

                main_plots = cursor.fetchall()

                treatments = {
                    row[1] for row in main_plots
                }

                main_plot_numbers = {
                    row[2] for row in main_plots
                }

                if len(main_plots) != 3:
                    errors.append(
                        f"{block_id}: expected 3 main plots; "
                        f"found {len(main_plots)}."
                    )

                if treatments != expected_treatments:
                    errors.append(
                        f"{block_id}: irrigation treatments "
                        "are incomplete or incorrect."
                    )

                if main_plot_numbers != {1, 2, 3}:
                    errors.append(
                        f"{block_id}: main plot numbers "
                        "must be 1, 2, and 3."
                    )

                for main_plot_id, treatment, main_plot_number in main_plots:

                    cursor.execute(
                        f"""
                        SELECT crop_code, subplot_number
                        FROM experimental_subplots
                        WHERE main_plot_id = {placeholder}
                        """,
                        (main_plot_id,)
                    )

                    subplot_rows = cursor.fetchall()

                    crops = {
                        row[0] for row in subplot_rows
                    }

                    subplot_numbers = {
                        row[1] for row in subplot_rows
                    }

                    if len(subplot_rows) != 3:
                        errors.append(
                            f"{main_plot_id}: expected 3 crop "
                            f"subplots; found {len(subplot_rows)}."
                        )

                    if crops != expected_crops:
                        errors.append(
                            f"{main_plot_id}: crop assignments "
                            "are incomplete or incorrect."
                        )

                    if subplot_numbers != {1, 2, 3}:
                        errors.append(
                            f"{main_plot_id}: subplot numbers "
                            "must be 1, 2, and 3."
                        )

        counts = get_experimental_design_counts(connection)

        expected_counts = {
            "experimental_blocks": 6,
            "experimental_main_plots": 18,
            "experimental_subplots": 54,
        }

        for table_name, expected_count in expected_counts.items():

            actual_count = counts[table_name]

            if actual_count != expected_count:
                errors.append(
                    f"{table_name}: expected {expected_count}; "
                    f"found {actual_count}."
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "counts": counts,
        }

    finally:
        cursor.close()

# ==================================================


# ==================================================
# 4A. READ-ONLY EXPERIMENTAL DESIGN INVENTORY
# ==================================================

def get_experimental_design_inventory(
    connection: Any,
    db_type: str,
) -> dict:
    """
    Return a complete, read-only inventory of the
    registered experimental design.

    Includes:
    - Research sites
    - Experimental blocks
    - Irrigation-treatment main plots
    - Crop subplots
    - Registry record counts
    - Structural validation results

    Supports SQLite and PostgreSQL.

    This function executes SELECT statements only.
    It does not insert, update, delete, or commit records.
    """

    normalized_db_type = _normalize_experimental_db_type(db_type)

    cursor = connection.cursor()

    try:
        inventory_queries = {
            "sites": """
                SELECT
                    site_id,
                    site_code,
                    site_name,
                    region,
                    latitude,
                    longitude,
                    altitude_m,
                    created_at
                FROM experimental_sites
                ORDER BY site_code
            """,

            "blocks": """
                SELECT
                    s.site_id,
                    s.site_code,
                    s.site_name,
                    b.block_id,
                    b.replication_number,
                    b.created_at
                FROM experimental_blocks AS b
                INNER JOIN experimental_sites AS s
                    ON s.site_id = b.site_id
                ORDER BY s.site_code, b.replication_number
            """,

            "main_plots": """
                SELECT
                    s.site_code,
                    b.block_id,
                    b.replication_number,
                    m.main_plot_id,
                    m.irrigation_treatment,
                    m.main_plot_number,
                    m.created_at
                FROM experimental_main_plots AS m
                INNER JOIN experimental_blocks AS b
                    ON b.block_id = m.block_id
                INNER JOIN experimental_sites AS s
                    ON s.site_id = b.site_id
                ORDER BY
                    s.site_code,
                    b.replication_number,
                    m.main_plot_number
            """,

            "subplots": """
                SELECT
                    s.site_code,
                    b.block_id,
                    b.replication_number,
                    m.main_plot_id,
                    m.irrigation_treatment,
                    p.subplot_id,
                    p.crop_code,
                    p.subplot_number,
                    p.created_at
                FROM experimental_subplots AS p
                INNER JOIN experimental_main_plots AS m
                    ON m.main_plot_id = p.main_plot_id
                INNER JOIN experimental_blocks AS b
                    ON b.block_id = m.block_id
                INNER JOIN experimental_sites AS s
                    ON s.site_id = b.site_id
                ORDER BY
                    s.site_code,
                    b.replication_number,
                    m.main_plot_number,
                    p.subplot_number
            """,
        }

        records = {}

        for section, query in inventory_queries.items():
            cursor.execute(query)

            if cursor.description is None:
                raise RuntimeError(
                    f"Inventory query returned no columns: {section}"
                )

            columns = [
                description[0]
                for description in cursor.description
            ]

            records[section] = [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]

    finally:
        cursor.close()

    counts = get_experimental_design_counts(connection)

    validation = validate_experimental_design(
        connection,
        normalized_db_type,
    )

    return {
        "sites": records["sites"],
        "blocks": records["blocks"],
        "main_plots": records["main_plots"],
        "subplots": records["subplots"],
        "counts": counts,
        "validation": validation,
    }


# 5. EXPERIMENTAL DESIGN REGISTRATION FUNCTIONS
# ==================================================

def _normalize_experimental_db_type(db_type: str) -> str:
    """Validate and normalize the supported database type."""

    if not isinstance(db_type, str):
        raise TypeError("db_type must be a string.")

    normalized = db_type.strip().lower()

    if normalized not in ("sqlite", "postgres"):
        raise ValueError(
            "Unsupported database type. "
            "Expected 'sqlite' or 'postgres'."
        )

    return normalized


def _require_nonempty_text(value: str, field_name: str) -> str:
    """Validate a required text field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string."
        )

    return value.strip()


def _execute_experimental_insert(
    connection: Any,
    db_type: str,
    sql: str,
    parameters: tuple,
) -> None:
    """
    Execute one parameterized experimental-design INSERT.

    Transaction ownership remains with the caller.
    This helper does not commit or roll back.
    """

    normalized_db_type = _normalize_experimental_db_type(
        db_type
    )

    placeholder = (
        "?" if normalized_db_type == "sqlite" else "%s"
    )

    database_sql = sql.replace("?", placeholder)

    cursor = connection.cursor()

    try:
        cursor.execute(database_sql, parameters)
    finally:
        cursor.close()


def register_experimental_site(
    connection: Any,
    db_type: str,
    site_id: str,
    site_code: str,
    site_name: str,
    region: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    altitude_m: float | None = None,
) -> str:
    """
    Register one research site.

    Permitted site codes: MARC and HARC.

    Returns the registered site_id.
    Does not commit or roll back.
    """

    site_id = _require_nonempty_text(site_id, "site_id")
    site_code = _require_nonempty_text(
        site_code, "site_code"
    ).upper()
    site_name = _require_nonempty_text(
        site_name, "site_name"
    )

    if site_code not in {"MARC", "HARC"}:
        raise ValueError(
            "site_code must be either 'MARC' or 'HARC'."
        )

    if region is not None:
        region = _require_nonempty_text(region, "region")

    _execute_experimental_insert(
        connection,
        db_type,
        """
        INSERT INTO experimental_sites (
            site_id,
            site_code,
            site_name,
            region,
            latitude,
            longitude,
            altitude_m
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            site_id,
            site_code,
            site_name,
            region,
            latitude,
            longitude,
            altitude_m,
        ),
    )

    return site_id


def register_experimental_block(
    connection: Any,
    db_type: str,
    block_id: str,
    site_id: str,
    replication_number: int,
) -> str:
    """
    Register one replication block under an existing site.

    Valid replication numbers: 1, 2, and 3.

    Returns the registered block_id.
    Does not commit or roll back.
    """

    block_id = _require_nonempty_text(
        block_id, "block_id"
    )
    site_id = _require_nonempty_text(
        site_id, "site_id"
    )

    if (
        isinstance(replication_number, bool)
        or not isinstance(replication_number, int)
        or replication_number not in (1, 2, 3)
    ):
        raise ValueError(
            "replication_number must be an integer "
            "of 1, 2, or 3."
        )

    _execute_experimental_insert(
        connection,
        db_type,
        """
        INSERT INTO experimental_blocks (
            block_id,
            site_id,
            replication_number
        )
        VALUES (?, ?, ?)
        """,
        (
            block_id,
            site_id,
            replication_number,
        ),
    )

    return block_id


def register_experimental_main_plot(
    connection: Any,
    db_type: str,
    main_plot_id: str,
    block_id: str,
    irrigation_treatment: str,
    main_plot_number: int,
) -> str:
    """
    Register one irrigation-treatment main plot.

    Permitted treatments:
    FIXED, SOIL_MOISTURE, ET_BASED.

    Returns the registered main_plot_id.
    Does not commit or roll back.
    """

    main_plot_id = _require_nonempty_text(
        main_plot_id, "main_plot_id"
    )
    block_id = _require_nonempty_text(
        block_id, "block_id"
    )
    irrigation_treatment = _require_nonempty_text(
        irrigation_treatment,
        "irrigation_treatment",
    ).upper()

    allowed_treatments = {
        "FIXED",
        "SOIL_MOISTURE",
        "ET_BASED",
    }

    if irrigation_treatment not in allowed_treatments:
        raise ValueError(
            "Invalid irrigation treatment. "
            f"Expected one of {sorted(allowed_treatments)}."
        )

    if (
        isinstance(main_plot_number, bool)
        or not isinstance(main_plot_number, int)
        or main_plot_number not in (1, 2, 3)
    ):
        raise ValueError(
            "main_plot_number must be an integer "
            "of 1, 2, or 3."
        )

    _execute_experimental_insert(
        connection,
        db_type,
        """
        INSERT INTO experimental_main_plots (
            main_plot_id,
            block_id,
            irrigation_treatment,
            main_plot_number
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            main_plot_id,
            block_id,
            irrigation_treatment,
            main_plot_number,
        ),
    )

    return main_plot_id


def register_experimental_subplot(
    connection: Any,
    db_type: str,
    subplot_id: str,
    main_plot_id: str,
    crop_code: str,
    subplot_number: int,
) -> str:
    """
    Register one crop subplot under an existing main plot.

    Permitted crops: MAIZE, WHEAT, TEFF.

    Returns the registered subplot_id.
    Does not commit or roll back.
    """

    subplot_id = _require_nonempty_text(
        subplot_id, "subplot_id"
    )
    main_plot_id = _require_nonempty_text(
        main_plot_id, "main_plot_id"
    )
    crop_code = _require_nonempty_text(
        crop_code, "crop_code"
    ).upper()

    allowed_crops = {
        "MAIZE",
        "WHEAT",
        "TEFF",
    }

    if crop_code not in allowed_crops:
        raise ValueError(
            "Invalid crop code. "
            f"Expected one of {sorted(allowed_crops)}."
        )

    if (
        isinstance(subplot_number, bool)
        or not isinstance(subplot_number, int)
        or subplot_number not in (1, 2, 3)
    ):
        raise ValueError(
            "subplot_number must be an integer "
            "of 1, 2, or 3."
        )

    _execute_experimental_insert(
        connection,
        db_type,
        """
        INSERT INTO experimental_subplots (
            subplot_id,
            main_plot_id,
            crop_code,
            subplot_number
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            subplot_id,
            main_plot_id,
            crop_code,
            subplot_number,
        ),
    )

    return subplot_id

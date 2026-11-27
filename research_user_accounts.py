
"""
TOKUMA 3-in-1 Integrated Model
Individual Research User Account Management

Step 5.3.12.4.37.12

Initial scope:
- Define individual research account schema.
- Support SQLite and PostgreSQL.
- Provide Argon2id password hashing and verification.
- Validate account schema.
- Provide read-only account counts.

Security principles:
- Never store plaintext passwords.
- Never create default administrator credentials.
- Store normalized usernames.
- Use stable UUID account identifiers.
- Preserve caller ownership of database transactions.
- Do not commit or rollback inside this module.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from importlib.metadata import version
from uuid import uuid4
from typing import Any
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)


SUPPORTED_DB_TYPES = {"sqlite", "postgres"}

ACCOUNT_ROLES = {
    "FIELD_RESEARCHER",
    "INDEPENDENT_REVIEWER",
    "AUTHORIZED_VERIFIER",
    "SYSTEM_ADMIN",
}

ACCOUNT_STATUSES = {
    "ACTIVE",
    "DEACTIVATED",
}

MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 1024

_PASSWORD_HASHER = PasswordHasher()
_DUMMY_PASSWORD_HASH = _PASSWORD_HASHER.hash(
    secrets.token_urlsafe(32)
)

RESEARCH_USER_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS research_users (
        user_id TEXT PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        display_name TEXT NOT NULL,
        institution TEXT NOT NULL DEFAULT '',
        password_hash TEXT NOT NULL,

        role TEXT NOT NULL CHECK (
            role IN (
                'FIELD_RESEARCHER',
                'INDEPENDENT_REVIEWER',
                'AUTHORIZED_VERIFIER',
                'SYSTEM_ADMIN'
            )
        ),

        account_status TEXT NOT NULL DEFAULT 'ACTIVE'
            CHECK (account_status IN ('ACTIVE', 'DEACTIVATED')),

        created_at TEXT NOT NULL,
        created_by_user_id TEXT,

        last_login_at TEXT,
        failed_login_attempts INTEGER NOT NULL DEFAULT 0
            CHECK (failed_login_attempts >= 0),
        locked_until TEXT,

        deactivated_at TEXT,

        FOREIGN KEY (created_by_user_id)
            REFERENCES research_users(user_id)
            ON DELETE RESTRICT,

        CHECK (
            (account_status = 'ACTIVE' AND deactivated_at IS NULL)
            OR
            (account_status = 'DEACTIVATED' AND deactivated_at IS NOT NULL)
        )
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS
        idx_research_users_role_status
    ON research_users(role, account_status)
    """,
]


REQUIRED_USER_COLUMNS = {
    "user_id",
    "username",
    "display_name",
    "institution",
    "password_hash",
    "role",
    "account_status",
    "created_at",
    "created_by_user_id",
    "last_login_at",
    "failed_login_attempts",
    "locked_until",
    "deactivated_at",
}


def _normalize_db_type(db_type: str) -> str:
    if not isinstance(db_type, str):
        raise TypeError("db_type must be a string.")

    normalized = db_type.strip().lower()

    if normalized not in SUPPORTED_DB_TYPES:
        raise ValueError(
            f"Unsupported database type: {db_type!r}"
        )

    return normalized


def normalize_username(username: str) -> str:
    if not isinstance(username, str):
        raise TypeError("Username must be a string.")

    normalized = username.strip().lower()

    if not normalized:
        raise ValueError("Username cannot be empty.")

    if len(normalized) > 100:
        raise ValueError("Username cannot exceed 100 characters.")

    if any(character.isspace() for character in normalized):
        raise ValueError("Username cannot contain whitespace.")

    return normalized


def _validate_password(password: str) -> None:
    if not isinstance(password, str):
        raise TypeError("Password must be a string.")

    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"Password must contain at least "
            f"{MIN_PASSWORD_LENGTH} characters."
        )

    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(
            f"Password cannot exceed {MAX_PASSWORD_LENGTH} characters."
        )


def hash_password(password: str) -> str:
    """
    Generate an Argon2id password hash.

    The original password is never returned or stored.
    """
    _validate_password(password)
    return _PASSWORD_HASHER.hash(password)


def verify_password(password: str, encoded_hash: str) -> bool:
    """
    Verify a password against its Argon2 hash.

    Returns False for incorrect passwords or invalid hashes.
    """
    if not isinstance(password, str):
        return False

    if not isinstance(encoded_hash, str) or not encoded_hash:
        return False

    try:
        return bool(
            _PASSWORD_HASHER.verify(encoded_hash, password)
        )
    except (
        VerifyMismatchError,
        VerificationError,
        InvalidHashError,
        TypeError,
        ValueError,
    ):
        return False


def password_hash_needs_rehash(encoded_hash: str) -> bool:
    """
    Check whether an existing Argon2 hash should be upgraded
    to the current PasswordHasher configuration.
    """
    if not isinstance(encoded_hash, str) or not encoded_hash:
        return False

    try:
        return _PASSWORD_HASHER.check_needs_rehash(encoded_hash)
    except (VerificationError, InvalidHashError, TypeError, ValueError):
        return False


def get_password_security_info() -> dict[str, Any]:
    """Return non-secret password-security configuration details."""
    return {
        "algorithm": "Argon2id",
        "argon2_cffi_version": version("argon2-cffi"),
        "minimum_password_length": MIN_PASSWORD_LENGTH,
        "maximum_password_length": MAX_PASSWORD_LENGTH,
    }


def _execute_statement(connection: Any, db_type: str, statement: str) -> None:
    cursor = connection.cursor()

    try:
        cursor.execute(statement)
    finally:
        cursor.close()


def initialize_user_account_schema(
    connection: Any,
    db_type: str,
) -> None:
    """
    Initialize the account schema.

    Transaction ownership remains with the caller.
    This function does not commit or rollback.
    """
    normalized_db_type = _normalize_db_type(db_type)

    if connection is None:
        raise ValueError("A database connection is required.")

    if normalized_db_type == "sqlite":
        cursor = connection.cursor()

        try:
            result = cursor.execute(
                "PRAGMA foreign_keys"
            ).fetchone()

            if not result or result[0] != 1:
                raise RuntimeError(
                    "SQLite foreign-key enforcement must be enabled "
                    "before initializing the account schema."
                )
        finally:
            cursor.close()

    for statement in RESEARCH_USER_SCHEMA:
        _execute_statement(
            connection,
            normalized_db_type,
            statement,
        )


def upgrade_user_account_schema(
    connection: Any,
    db_type: str,
) -> list[str]:
    """
    Explicitly add missing account security columns.

    Transaction ownership remains with the caller.
    Existing account records are preserved.
    """
    normalized_db_type = _normalize_db_type(db_type)

    if connection is None:
        raise ValueError("A database connection is required.")

    cursor = connection.cursor()

    try:
        if normalized_db_type == "sqlite":
            cursor.execute("PRAGMA table_info(research_users)")
            rows = cursor.fetchall()
            actual_columns = {row[1] for row in rows}
        else:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = %s
                """,
                ("research_users",),
            )
            rows = cursor.fetchall()
            actual_columns = {row[0] for row in rows}
    finally:
        cursor.close()

    if not actual_columns:
        raise RuntimeError(
            "research_users does not exist. Initialize the account schema first."
        )

    additions = [
        (
            "last_login_at",
            "ALTER TABLE research_users ADD COLUMN last_login_at TEXT",
        ),
        (
            "failed_login_attempts",
            """
            ALTER TABLE research_users
            ADD COLUMN failed_login_attempts INTEGER NOT NULL DEFAULT 0
            CHECK (failed_login_attempts >= 0)
            """,
        ),
        (
            "locked_until",
            "ALTER TABLE research_users ADD COLUMN locked_until TEXT",
        ),
    ]

    added_columns = []

    for column_name, statement in additions:
        if column_name not in actual_columns:
            _execute_statement(
                connection,
                normalized_db_type,
                statement,
            )
            added_columns.append(column_name)

    return added_columns


def validate_user_account_schema(
    connection: Any,
    db_type: str,
) -> dict[str, Any]:
    """
    Read-only structural validation of the research_users table.
    """
    normalized_db_type = _normalize_db_type(db_type)

    if connection is None:
        raise ValueError("A database connection is required.")

    cursor = connection.cursor()

    try:
        if normalized_db_type == "sqlite":
            cursor.execute("PRAGMA table_info(research_users)")
            rows = cursor.fetchall()
            actual_columns = {row[1] for row in rows}
        else:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = %s
                """,
                ("research_users",),
            )
            rows = cursor.fetchall()
            actual_columns = {row[0] for row in rows}

    finally:
        cursor.close()

    missing_columns = sorted(
        REQUIRED_USER_COLUMNS - actual_columns
    )

    return {
        "valid": not missing_columns,
        "table": "research_users",
        "required_column_count": len(REQUIRED_USER_COLUMNS),
        "actual_column_count": len(actual_columns),
        "missing_columns": missing_columns,
    }


def get_user_account_counts(
    connection: Any,
    db_type: str,
) -> dict[str, int]:
    """Read-only account counts by status and role."""
    normalized_db_type = _normalize_db_type(db_type)

    if connection is None:
        raise ValueError("A database connection is required.")

    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            SELECT account_status, role, COUNT(*)
            FROM research_users
            GROUP BY account_status, role
            """
        )

        rows = cursor.fetchall()

    finally:
        cursor.close()

    counts = {
        "total": 0,
        "ACTIVE": 0,
        "DEACTIVATED": 0,
    }

    for status, role, count in rows:
        counts["total"] += int(count)
        counts[status] += int(count)
        counts[f"{status}_{role}"] = int(count)

    return counts




def create_research_user(
    connection: Any,
    db_type: str,
    creator_user_id: str,
    username: str,
    password: str,
    display_name: str,
    role: str,
    institution: str = "",
) -> str:
    """
    Create an individual research account.

    Only an active SYSTEM_ADMIN may create ordinary accounts.
    SYSTEM_ADMIN accounts require the separate bootstrap procedure.

    Passwords are stored only as Argon2id hashes.
    The database transaction remains owned by the caller.
    """

    normalized_db_type = _normalize_db_type(db_type)

    if connection is None:
        raise ValueError("A database connection is required.")

    if not isinstance(creator_user_id, str) or not creator_user_id.strip():
        raise ValueError("A valid creator user ID is required.")

    normalized_username = normalize_username(username)
    _validate_password(password)

    if not isinstance(display_name, str):
        raise TypeError("Display name must be a string.")

    normalized_display_name = display_name.strip()

    if not normalized_display_name:
        raise ValueError("Display name cannot be empty.")

    if len(normalized_display_name) > 200:
        raise ValueError("Display name cannot exceed 200 characters.")

    if not isinstance(institution, str):
        raise TypeError("Institution must be a string.")

    normalized_institution = institution.strip()

    if len(normalized_institution) > 300:
        raise ValueError("Institution cannot exceed 300 characters.")

    if not isinstance(role, str):
        raise TypeError("Role must be a string.")

    normalized_role = role.strip().upper()

    if normalized_role not in ACCOUNT_ROLES:
        raise ValueError("Unsupported account role.")

    if normalized_role == "SYSTEM_ADMIN":
        raise PermissionError(
            "SYSTEM_ADMIN accounts require explicit administrator "
            "bootstrap authorization."
        )

    placeholder = "?" if normalized_db_type == "sqlite" else "%s"
    cursor = connection.cursor()

    try:
        cursor.execute(
            f"""
            SELECT account_status, role
            FROM research_users
            WHERE user_id = {placeholder}
            """,
            (creator_user_id.strip(),),
        )

        creator = cursor.fetchone()

        if creator is None:
            raise PermissionError("Creator account does not exist.")

        creator_status, creator_role = creator

        if creator_status != "ACTIVE" or creator_role != "SYSTEM_ADMIN":
            raise PermissionError(
                "Only an active SYSTEM_ADMIN may create research accounts."
            )

        cursor.execute(
            f"""
            SELECT user_id
            FROM research_users
            WHERE username = {placeholder}
            """,
            (normalized_username,),
        )

        if cursor.fetchone() is not None:
            raise ValueError("Username already exists.")

        encoded_password = hash_password(password)
        new_user_id = generate_user_id()
        created_at = current_utc_timestamp()

        cursor.execute(
            f"""
            INSERT INTO research_users (
                user_id,
                username,
                display_name,
                institution,
                password_hash,
                role,
                account_status,
                created_at,
                created_by_user_id,
                failed_login_attempts
            )
            VALUES (
                {placeholder}, {placeholder}, {placeholder},
                {placeholder}, {placeholder}, {placeholder},
                {placeholder}, {placeholder}, {placeholder}, 0
            )
            """,
            (
                new_user_id,
                normalized_username,
                normalized_display_name,
                normalized_institution,
                encoded_password,
                normalized_role,
                "ACTIVE",
                created_at,
                creator_user_id.strip(),
            ),
        )

        return new_user_id

    finally:
        cursor.close()



def authenticate_research_user(
    connection: Any,
    db_type: str,
    username: str,
    password: str,
) -> dict[str, Any]:
    """
    Authenticate an individual research account.

    Security controls:
    - Argon2id password verification.
    - Generic failure response.
    - Five consecutive failures trigger a 15-minute lock.
    - Deactivated accounts cannot authenticate.
    - Successful login resets failed attempts and lockout.
    - Successful login updates last_login_at.
    - Outdated Argon2 hashes are upgraded when current password
      policy permits.

    Concurrency:
    - SQLite requires no active transaction and starts BEGIN IMMEDIATE.
    - PostgreSQL uses SELECT ... FOR UPDATE and requires autocommit off.

    The function never commits or rolls back. The caller owns the
    transaction and must commit authentication-attempt updates.
    """

    normalized_db_type = _normalize_db_type(db_type)

    failure = {
        "authenticated": False,
        "message": "Invalid username or password, or account unavailable.",
    }

    if connection is None:
        raise ValueError("A database connection is required.")

    try:
        normalized_username = normalize_username(username)
    except (TypeError, ValueError):
        return failure

    if not isinstance(password, str):
        return failure

    if normalized_db_type == "sqlite":
        if connection.in_transaction:
            raise RuntimeError(
                "SQLite authentication requires no active transaction. "
                "Commit or roll back the existing transaction first."
            )

        connection.execute("BEGIN IMMEDIATE")

    else:
        if getattr(connection, "autocommit", False):
            raise RuntimeError(
                "PostgreSQL authentication requires autocommit to be disabled."
            )

    placeholder = "?" if normalized_db_type == "sqlite" else "%s"
    cursor = connection.cursor()

    try:
        query = f"""
            SELECT
                user_id,
                username,
                display_name,
                institution,
                password_hash,
                role,
                account_status,
                failed_login_attempts,
                locked_until
            FROM research_users
            WHERE username = {placeholder}
        """

        if normalized_db_type == "postgres":
            query += " FOR UPDATE"

        cursor.execute(query, (normalized_username,))
        account = cursor.fetchone()

        if account is None:
            verify_password(password, _DUMMY_PASSWORD_HASH)
            return failure

        (
            user_id,
            stored_username,
            display_name,
            institution,
            encoded_password,
            role,
            account_status,
            failed_attempts,
            locked_until,
        ) = account

        if account_status != "ACTIVE":
            verify_password(password, _DUMMY_PASSWORD_HASH)
            return failure

        now = datetime.now(timezone.utc)

        if locked_until:
            try:
                lock_time = datetime.fromisoformat(
                    locked_until.replace("Z", "+00:00")
                )
            except (TypeError, ValueError):
                raise RuntimeError(
                    "Stored account lock timestamp is invalid."
                )

            if lock_time.tzinfo is None:
                raise RuntimeError(
                    "Stored account lock timestamp is timezone-naive."
                )

            if now < lock_time.astimezone(timezone.utc):
                verify_password(password, _DUMMY_PASSWORD_HASH)
                return failure

            # Lockout has expired. Start a new consecutive-failure period.
            failed_attempts = 0
            locked_until = None

        password_matches = verify_password(
            password,
            encoded_password,
        )

        if not password_matches:
            new_failed_attempts = int(failed_attempts) + 1
            new_locked_until = None

            if new_failed_attempts >= 5:
                new_locked_until = (
                    now + timedelta(minutes=15)
                ).isoformat()

            cursor.execute(
                f"""
                UPDATE research_users
                SET failed_login_attempts = {placeholder},
                    locked_until = {placeholder}
                WHERE user_id = {placeholder}
                """,
                (
                    new_failed_attempts,
                    new_locked_until,
                    user_id,
                ),
            )

            return failure

        updated_hash = encoded_password

        if password_hash_needs_rehash(encoded_password):
            try:
                _validate_password(password)
                updated_hash = hash_password(password)
            except (TypeError, ValueError):
                # Existing valid credentials remain usable. A password
                # outside current policy must be changed separately.
                updated_hash = encoded_password

        login_timestamp = now.isoformat()

        cursor.execute(
            f"""
            UPDATE research_users
            SET password_hash = {placeholder},
                last_login_at = {placeholder},
                failed_login_attempts = 0,
                locked_until = NULL
            WHERE user_id = {placeholder}
            """,
            (
                updated_hash,
                login_timestamp,
                user_id,
            ),
        )

        return {
            "authenticated": True,
            "message": "Authentication successful.",
            "user": {
                "user_id": user_id,
                "username": stored_username,
                "display_name": display_name,
                "institution": institution,
                "role": role,
            },
        }

    finally:
        cursor.close()


def generate_user_id() -> str:
    """Generate a stable UUID account identifier."""
    return str(uuid4())


def current_utc_timestamp() -> str:
    """Return a timezone-aware UTC ISO 8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()

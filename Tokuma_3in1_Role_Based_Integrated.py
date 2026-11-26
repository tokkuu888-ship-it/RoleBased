import sqlite3
from datetime import datetime
from importlib import import_module
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
# ==================================================
# STEP 5.3.10 - SENSOR ANOMALY DETECTION
# ==================================================

from sensor_anomaly_detection import (
    run_sensor_anomaly_detection
)

# STEP 5.3.11 - SENSOR-BASED IRRIGATION DECISION SUPPORT
from sensor_irrigation_decision import (
    evaluate_sensor_irrigation_decision
)

# STEP 5.3.12.1 - EXPERIMENTAL DESIGN DATABASE
from experimental_design_database import (
    initialize_experimental_design_schema,
    get_experimental_design_inventory,
)
def _load_optional_import(module_name, symbol_name=None, feature_name=None) -> Any | None:
    try:
        module = import_module(module_name)
        return getattr(module, symbol_name) if symbol_name else module
    except ImportError as error:
        if feature_name:
            print(f"{feature_name} not available: {error}")
        return None


# PostgreSQL support for cloud deployment
psycopg2 = _load_optional_import("psycopg2")
PSYCOPG2_AVAILABLE = psycopg2 is not None

from nexus_core import (
    _pack_context,
    monte_carlo_multiparam,
    pymoo_with_file_coupling,
    run_file_coupled_simulation,
    run_integrated_engine,
    run_morris_analysis,
    run_pymoo_ga_single_objective,
    run_pymoo_nsga2,
    run_sobol_analysis,
    salib_problem_spec,
)

# Enhancement modules are optional and may not be included in every deployment.
get_gis_integration = _load_optional_import("gis_integration", "get_gis_integration", "GIS module")
GIS_AVAILABLE = get_gis_integration is not None

get_ml_surrogate_manager = _load_optional_import("ml_surrogates", "get_ml_surrogate_manager", "ML surrogates")
ML_AVAILABLE = get_ml_surrogate_manager is not None

get_irrigation_physics = _load_optional_import(
    "irrigation_physics", "get_irrigation_physics", "Irrigation physics"
)
IRRIGATION_AVAILABLE = get_irrigation_physics is not None

get_climate_downscaling = _load_optional_import(
    "climate_downscaling", "get_climate_downscaling", "Climate downscaling"
)
CLIMATE_AVAILABLE = get_climate_downscaling is not None

get_economic_optimizer = _load_optional_import(
    "economic_optimization", "get_economic_optimizer", "Economic optimization"
)
ECONOMIC_AVAILABLE = get_economic_optimizer is not None

get_report_generator = _load_optional_import("report_generator", "get_report_generator", "Report generator")
REPORT_AVAILABLE = get_report_generator is not None

get_api_manager = _load_optional_import("api_connectors", "get_api_manager", "API connectors")
API_AVAILABLE = get_api_manager is not None


# ------------------------------
# NEW: Communication & Hardware Modules
# ------------------------------
class FieldCommunicationManager:
    """Manages server-field communication for offline/online sync"""
    
    def __init__(self):
        self.pending_requests = []
        self.irrigation_instructions = []
        self.field_reports = []
    
    def add_field_request(self, farmer_id, location, crop_stage, current_soil_moisture):
        """Add irrigation request from field user"""
        request = {
            'timestamp': datetime.now().isoformat(),
            'farmer_id': farmer_id,
            'location': location,
            'crop_stage': crop_stage,
            'soil_moisture': current_soil_moisture,
            'status': 'pending'
        }
        self.pending_requests.append(request)
        return request
    
    def generate_irrigation_instruction(self, request, optimized_schedule):
        """Generate irrigation instruction based on optimized schedule"""
        instruction = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request['timestamp'],
            'farmer_id': request['farmer_id'],
            'irrigation_amount': optimized_schedule.get('net_irrigation_mm', optimized_schedule.get('amount_mm', 45)),
            'irrigation_timing': optimized_schedule.get('timing', 'immediate'),
            'duration_minutes': optimized_schedule.get('duration_minutes', optimized_schedule.get('duration', 60)),
            'flow_rate': optimized_schedule.get('flow_rate', 2.5),
            'status': 'sent'
        }
        self.irrigation_instructions.append(instruction)
        return instruction
    
    def record_field_application(self, instruction_id, actual_amount, actual_duration, applied_time):
        """Record actual irrigation application from field"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'instruction_id': instruction_id,
            'actual_amount_mm': actual_amount,
            'actual_duration_minutes': actual_duration,
            'applied_time': applied_time,
            'status': 'completed'
        }
        self.field_reports.append(report)
        return report


class SMSCommunicationModule:
    """SMS communication module for field users with limited internet"""
    
    def __init__(self):
        self.sms_queue = []
        self.sms_history = []
    
    def format_irrigation_instruction_sms(self, instruction):
        """Format irrigation instruction as SMS message"""
        message = f"TOKUMA IRR: Apply {instruction['irrigation_amount']}mm water. "
        duration = instruction["duration_minutes"]
        message += f"Duration: {duration:.1f}min. " if duration is not None else "Duration: not calculated. "
        message += f"Flow: {instruction['flow_rate']}L/s. "
        message += f"Time: {instruction['irrigation_timing']}. "
        message += f"ID: {instruction['request_id'][:8]}"
        return message
    
    def queue_sms(self, phone_number, message):
        """Queue SMS for sending"""
        sms = {
            'timestamp': datetime.now().isoformat(),
            'phone_number': phone_number,
            'message': message,
            'status': 'queued'
        }
        self.sms_queue.append(sms)
        return sms
    
    def send_sms(self, sms_item):
        """Simulate SMS sending (in production, integrate with SMS gateway)"""
        sms_item['status'] = 'sent'
        sms_item['sent_timestamp'] = datetime.now().isoformat()
        self.sms_history.append(sms_item)
        return sms_item


class HardwareControlInterface:
    """Interface for furrow irrigation hardware control"""
    
    def __init__(self):
        self.hardware_status = {
            'valve_open': False,
            'flow_rate': 0.0,
            'pressure': 0.0,
            'battery_level': 100,
            'connection_status': 'disconnected'
        }
        self.control_log = []
    
    def connect_hardware(self, device_id):
        """Connect to irrigation control hardware"""
        self.hardware_status['connection_status'] = 'connected'
        self.hardware_status['device_id'] = device_id
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'connect',
            'device_id': device_id,
            'status': 'success'
        }
        self.control_log.append(log_entry)
        return log_entry
    
    def open_valve(self, flow_rate):
        """Open irrigation valve with specified flow rate"""
        self.hardware_status['valve_open'] = True
        self.hardware_status['flow_rate'] = flow_rate
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'open_valve',
            'flow_rate': flow_rate,
            'status': 'success'
        }
        self.control_log.append(log_entry)
        return log_entry
    
    def close_valve(self):
        """Close irrigation valve"""
        self.hardware_status['valve_open'] = False
        self.hardware_status['flow_rate'] = 0.0
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'close_valve',
            'status': 'success'
        }
        self.control_log.append(log_entry)
        return log_entry
    
    def automate_irrigation(self, amount_mm, duration_minutes, flow_rate):
        """Automated irrigation based on schedule"""
        self.open_valve(flow_rate)
        # In production, this would interface with actual hardware timers
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'automate_irrigation',
            'amount_mm': amount_mm,
            'duration_minutes': duration_minutes,
            'flow_rate': flow_rate,
            'status': 'scheduled'
        }
        self.control_log.append(log_entry)
        return log_entry


class IrrigationScheduler:
    """Estimate irrigation need from field-scale soil-water inputs."""
    
    def __init__(self):
        self.schedule_history = []
    
    def calculate_irrigation_schedule(
        self,
        soil_water_capacity,
        soil_moisture,
        rainfall_expected,
        water_availability,
        field_area_m2,
        flow_rate_lps,
    ):
        """Estimate the soil-water deficit and cap it by available irrigation supply."""
        inputs = (
            soil_water_capacity,
            soil_moisture,
            rainfall_expected,
            water_availability,
            field_area_m2,
            flow_rate_lps,
        )
        if not all(np.isfinite(value) for value in inputs):
            raise ValueError("Irrigation inputs must be finite numbers.")
        if any(value < 0 for value in inputs[:-1]):
            raise ValueError("Soil water, rainfall, supply, and field area cannot be negative.")
        if flow_rate_lps <= 0:
            raise ValueError("Flow rate must be greater than zero.")

        soil_water_deficit = max(
            0.0, soil_water_capacity - soil_moisture - rainfall_expected
        )
        net_irrigation = min(soil_water_deficit, water_availability)
        deficit_fraction = (
            min(1.0, soil_water_deficit / soil_water_capacity)
            if soil_water_capacity > 0
            else 0.0
        )

        if net_irrigation == 0:
            timing = "no irrigation needed"
        elif deficit_fraction > 0.7:
            timing = "immediate"
        elif deficit_fraction > 0.4:
            timing = "within_24h"
        else:
            timing = "within_48h"

        # A millimeter over one square meter is one liter.
        duration_minutes = (
            net_irrigation * field_area_m2 / (flow_rate_lps * 60)
            if net_irrigation > 0 and field_area_m2 > 0
            else 0.0 if net_irrigation == 0 else None
        )
        
        schedule = {
            'timestamp': datetime.now().isoformat(),
            'net_irrigation_mm': net_irrigation,
            'timing': timing,
            'duration_minutes': duration_minutes,
            'flow_rate': flow_rate_lps,
            'stress_level': deficit_fraction,
            'soil_moisture_current': soil_moisture,
            'rainfall_expected_mm': rainfall_expected,
            'water_availability_mm': water_availability,
            'soil_water_capacity_mm': soil_water_capacity,
            'field_area_m2': field_area_m2,
        }
        
        self.schedule_history.append(schedule)
        return schedule


# ------------------------------
# DATABASE SETUP
# ------------------------------
def init_db():
    """Initialize database: PostgreSQL for cloud, SQLite for local dev"""
    # Check for PostgreSQL credentials (Streamlit Cloud secrets)
    use_postgres = (
        "database" in st.secrets and 
        "host" in st.secrets and 
        PSYCOPG2_AVAILABLE
    )
    
    if use_postgres:
        # Cloud: PostgreSQL
        try:
            assert psycopg2 is not None
            conn = psycopg2.connect(
                host=st.secrets["host"],
                database=st.secrets["database"],
                user=st.secrets["user"],
                password=st.secrets["password"],
                port=st.secrets.get("port", 5432)
            )
            c = conn.cursor()
            st.session_state.db_type = "postgres"
        except Exception as e:
            st.warning(f"PostgreSQL not available: {e}. Using local SQLite.")
            conn = sqlite3.connect("tokuma_phd_research.db", check_same_thread=False, timeout=30)
            c = conn.cursor()
            st.session_state.db_type = "sqlite"
    else:
        # Local: SQLite
        conn = sqlite3.connect("tokuma_phd_research.db", check_same_thread=False, timeout=30)
        c = conn.cursor()
        st.session_state.db_type = "sqlite"
    
    # Enable referential integrity for local SQLite connections.
    if st.session_state.db_type == "sqlite":
        c.execute("PRAGMA foreign_keys = ON")
        fk_result = c.execute("PRAGMA foreign_keys").fetchone()

        if fk_result is None or fk_result[0] != 1:
            raise RuntimeError(
                "SQLite foreign-key enforcement could not be enabled."
            )

    c.execute(
        """CREATE TABLE IF NOT EXISTS research_logs(
        timestamp TEXT, researcher TEXT, email TEXT, country TEXT,
        crop TEXT, year INTEGER, scenario TEXT, wp_index REAL,
        total_demand REAL, ag_demand REAL, supply REAL,
        schedule_method TEXT, sim_engine TEXT)"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS field_data(
        timestamp TEXT, location TEXT, crop TEXT, soil_type TEXT,
        temp_max REAL, humidity REAL, rainfall REAL, irrigation REAL,
        furrow_length REAL, furrow_slope REAL, notes TEXT)"""
    )
    # NEW: Tables for communication and scheduling
    c.execute(
        """CREATE TABLE IF NOT EXISTS irrigation_schedules(
        timestamp TEXT, location TEXT, crop TEXT, 
        irrigation_amount REAL, timing TEXT, duration REAL,
        flow_rate REAL, stress_level REAL, status TEXT)"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS field_requests(
        timestamp TEXT, farmer_id TEXT, location TEXT, 
        crop_stage TEXT, soil_moisture REAL, status TEXT)"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS hardware_control_log(
        timestamp TEXT, device_id TEXT, action TEXT, 
        parameters TEXT, status TEXT)"""
    )
    # STEP 5.3.12.1 - Initialize experimental-design schema.
    # The caller (init_db) retains transaction ownership.
    initialize_experimental_design_schema(
        conn,
        st.session_state.db_type
    )

    conn.commit()
    return conn


db_conn = init_db()


def _execute_db(query, params=()):
    """Execute a parameterized statement on either supported database."""
    if st.session_state.db_type == "postgres":
        query = query.replace("?", "%s")
    cursor = db_conn.cursor()
    cursor.execute(query, params)
    return cursor


def _read_db_query(query):
    """Read a query into a DataFrame for SQLite or PostgreSQL connections."""
    if st.session_state.db_type != "postgres":
        return pd.read_sql_query(query, db_conn)

    cursor = _execute_db(query)
    if cursor.description is None:
        raise RuntimeError("Database query did not return a result set.")
    columns = [description[0] for description in cursor.description]
    return pd.DataFrame.from_records(cursor.fetchall(), columns=columns)


def metrics_tuple_from_dict(m):
    return (
        m["pop"],
        m["supply"],
        m["demand"],
        m["ag_demand"],
        m["ag_cons"],
        m["yield"],
        m["wp"],
        m["temp_rise"],
        m["eff"],
    )


# ------------------------------
# REACTIVE PARAMETER UPDATES
# ------------------------------
def update_reactive_parameters():
    """Update reactive parameters when sidebar changes"""
    if 'last_params' not in st.session_state:
        st.session_state.last_params = {}
    
    current_params = {
        'country': st.session_state.get('country', 'Ethiopia'),
        'target_year': st.session_state.get('target_year', 2050),
        'crop_type': st.session_state.get('crop_type', 'Wheat'),
        'climate_scenario': st.session_state.get('climate_scenario', 'SSP2-4.5'),
        'schedule': st.session_state.get('schedule_method', 'ET-based'),
        'f_len': st.session_state.get('f_len', 150.0),
        'f_slp': st.session_state.get('f_slp', 0.1),
        'climate_sensitivity': st.session_state.get('climate_sensitivity', 1.0),
    }
    
    params_changed = current_params != st.session_state.last_params
    
    if params_changed and st.session_state.get('simulation_run', False):
        st.session_state.params_changed = True
        st.info("⚠️ Parameters changed. Click 'Run Simulation-Optimization' to update results.")
    
    st.session_state.last_params = current_params.copy()
    return params_changed


# ------------------------------
# UI LAYOUT
# ------------------------------
st.set_page_config(page_title="TOKUMA 3-in-1", layout="wide", page_icon="🧬")

# ------------------------------
# ROLE-BASED AUTHENTICATION
# ------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "auth_user" not in st.session_state:
    st.session_state.auth_user = ""
if "auth_institution" not in st.session_state:
    st.session_state.auth_institution = ""

def _login():
    """Validate credentials configured in Streamlit secrets."""
    username_value = st.session_state.get("login_username", "").strip()
    password_value = st.session_state.get("login_password", "")
    admin_password = st.secrets.get("ADMIN_PASSWORD", "")
    field_password = st.secrets.get("FIELD_PASSWORD", "")

    if admin_password and username_value == "admin" and password_value == admin_password:
        st.session_state.authenticated = True
        st.session_state.user_role = "Admin / Researcher"
        st.session_state.auth_user = st.session_state.get("login_name", "Administrator").strip() or "Administrator"
        st.session_state.auth_institution = st.session_state.get("login_institution", "Addis Ababa University").strip()
        st.session_state.pop("login_error", None)
    elif field_password and username_value == "field" and password_value == field_password:
        st.session_state.authenticated = True
        st.session_state.user_role = "Field Extension Worker / Farmer"
        st.session_state.auth_user = st.session_state.get("login_name", "Field User").strip() or "Field User"
        st.session_state.auth_institution = st.session_state.get("login_institution", "").strip()
        st.session_state.pop("login_error", None)
    else:
        st.session_state.login_error = True

if not st.session_state.authenticated:
    st.title("🔐 Tokuma 3-in-1 Secure Login")
    st.caption("Sign in with the account assigned to your role.")
    with st.form("role_based_login"):
        st.text_input("Full name", key="login_name")
        st.text_input("Institution / Organization", key="login_institution")
        st.text_input("Username", key="login_username")
        st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Login")
        if submitted:
            _login()
            if st.session_state.authenticated:
                st.rerun()
            st.error("User ID or password incorrect.")
    st.info(
        "Set ADMIN_PASSWORD and FIELD_PASSWORD in Streamlit secrets before signing in. "
        "Admin/Researcher and Field Worker accounts have different phase permissions."
    )
    st.stop()

# Authenticated session: identity and logout
st.session_state.auth = True  # compatibility with the original application's auth flag
st.session_state.user = st.session_state.auth_user
st.session_state.inst = st.session_state.auth_institution
user_role = st.session_state.user_role
is_admin = user_role == "Admin / Researcher"
is_field_user = user_role == "Field Extension Worker / Farmer"

with st.sidebar:
    st.success(f"Logged in as: {user_role}")
    st.caption(st.session_state.user)
    if st.button("Log out", key="role_logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.divider()
    st.header("🌿 TOKUMA Control Panel")
    if is_admin:
        st.caption("Configure the nexus study, then run the integrated analysis.")
    else:
        st.caption("Field tools are available in the workspace tabs.")

# Main application banner
if is_admin:
    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, #002b5c, #0056b3); padding:30px; border-radius:15px 15px 0 0; color:white; text-align:center; box-shadow: 0px 6px 15px rgba(0,0,0,0.3); margin-bottom: -15px;">
            <h1 style="margin-bottom:5px; font-size:2.5rem;">Tokuma 3-in-1 Integrated Model</h1>
            <p style="font-size:1.4rem; font-weight:400; margin-bottom:15px; opacity:0.95;">An Intelligent DSS for Water-Food-Climate Nexus Analysis </p>
            <div style="border-top: 1px solid rgba(255,255,255,0.3); width:60%; margin: 10px auto; padding-top:10px;">
                <p style="font-size:1.1rem; font-weight:300;">Researcher: <b>{st.session_state.user}</b> | Institution: <b>{st.session_state.inst}</b></p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("🌾 Tokuma 3-in-1: Field Operations Portal")
    st.success("Welcome. You can access irrigation scheduling and field operations only.")

st.markdown(
    """
    <style>
    div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
        flex-wrap: nowrap;
        overflow-x: auto;
        scrollbar-width: thin;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        flex: 1 0 180px;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Role-specific sidebar controls. Field accounts do not receive research/admin controls.
country = "Ethiopia"
target_year = 2050
crop_type = "Wheat"
s_mod = "SWAT"
p_mod = "Cohort-Component"
d_meth = "System Dynamics-WEAP"
sim_eng = "AquaCrop"
use_ml_surrogate = False
use_lstm = False
use_breeding = False
opt_eng = "Genetic Algorithm (GA)"
use_file_coupling = False
climate_scenario = "SSP2-4.5"
climate_sensitivity = 1.0
sobol_n = 128
mc_n = 1500
run_morris_ui = False
run_trigger = False

if is_admin:
    with st.sidebar:
        with st.expander("👤 Research Profile", expanded=False):
            username = st.text_input("Name", value=st.session_state.user)

        with st.expander("🌍 Environment & Crop", expanded=True):
            country = st.selectbox("Select Country", ["Ethiopia", "Kenya", "Sudan", "Egypt"], key="country")
            target_year = st.slider("Target Year", 2024, 2100, 2050, key="target_year")
            crop_type = st.selectbox("Crop Type", ["Wheat", "Maize", "Teff"], key="crop_type")
            climate_scenario = st.select_slider(
                "Emission Scenario (SSP)",
                options=["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"],
                key="climate_scenario",
            )
            climate_sensitivity = st.slider(
                "Regional Sensitivity Index", 0.5, 3.0, 1.0, key="climate_sensitivity"
            )

        with st.expander("🚜 Management & Nexus Models", expanded=False):
            s_mod = st.selectbox("Supply Model", ["SWAT", "MODFLOW", "SWAT + MODFLOW"])
            p_mod = st.selectbox("Population Model", ["Cohort-Component", "Bayesian"])
            d_meth = st.selectbox("Demand Projection Method", ["System Dynamics-WEAP", "Agent-Based"])
            f_len = st.number_input("Furrow Length (m)", value=150.0, key="f_len")
            f_slp = st.number_input("Furrow Slope (%)", value=0.1, key="f_slp")
            schedule = st.selectbox(
                "Scheduling", ["Traditional", "ET-based", "Soil moisture based"], key="schedule_method"
            )

        with st.expander("🔬 Simulation & Analysis", expanded=False):
            sim_eng = st.selectbox("Simulation Engine", ["AquaCrop", "EPIC", "WaPOR"])
            opt_eng = st.selectbox("Optimization Engine", ["Genetic Algorithm (GA)", "NSGA-II"])
            use_ml_surrogate = st.checkbox("Enable PINNs/GNNs Surrogate")
            use_lstm = st.checkbox("Enable LSTM Surrogate")
            use_breeding = st.checkbox("Enable In-Silico Breeding Gain")
            st.caption("Integrated proxy vs file-coupled subprocess (template → runner → CSV).")
            use_file_coupling = st.checkbox(
                "File-coupled simulation (subprocess pipeline)",
                value=False,
                help="Writes model_templates inputs, runs coupled_runner.py, reads sim_output.csv — same core, closed loop.",
            )
            sobol_n = st.slider("Sobol base sample size (N)", 64, 384, 128, 32)
            mc_n = st.slider("Monte Carlo draws", 500, 5000, 1500, 100)
            run_morris_ui = st.checkbox("Compute Morris screening (μ*, σ)", value=False)

        with st.expander("🚜 Field Operations Settings", expanded=False):
            st.write("**Communication Settings**")
            enable_sms = st.checkbox("Enable SMS Communication", value=True)
            sms_gateway = st.selectbox("SMS Gateway", ["Twilio", "AfricasTalking", "Local"])
            st.write("**Hardware Control**")
            enable_hardware = st.checkbox("Enable Hardware Control", value=False)
            hardware_device_id = st.text_input("Device ID", value="HW001")
            st.write("**Irrigation Scheduling**")
            auto_schedule = st.checkbox("Auto-generate Schedules", value=True)
            schedule_frequency = st.selectbox("Schedule Frequency", ["Daily", "Weekly", "Bi-weekly"])
        st.divider()
        run_trigger = st.button(
            "🚀 Run Simulation-Optimization", type="primary", use_container_width=True
        )
else:
    # Field profile: local operational context only; research/model controls remain hidden.
    with st.sidebar:
        st.subheader("Field Profile")
        username = st.session_state.user
        country = st.selectbox("Field Region", ["Ethiopia", "Kenya", "Sudan", "Egypt"], key="field_country")
        crop_type = st.selectbox("Crop", ["Wheat", "Maize", "Teff"], key="field_crop")
    target_year = datetime.now().year
    f_len, f_slp = 150.0, 0.1
    schedule = "ET-based"
    enable_sms, sms_gateway = True, "Local"
    enable_hardware, hardware_device_id = False, ""
    auto_schedule, schedule_frequency = True, "Daily"

# Research simulation and optimization are restricted to Admin/Researcher.
if is_admin:
    update_reactive_parameters()
else:
    run_trigger = False

# Preserve the original five phase implementations, but only render authorized phases.
phase1 = phase2 = phase5 = None
if is_admin:
    phase_labels = [
        "🌍 Drivers",
        "🔬 Simulation",
        "💧 Irrigation",
        "🚜 Field Operations",
        "📊 Reports",
    ]
    phase_handles = st.tabs(phase_labels)
    phase1, phase2, phase3, phase4, phase5 = phase_handles
else:
    field_handles = st.tabs(["💧 Phase 3: Irrigation Scheduling", "🚜 Phase 4: Field Operations"])
    phase3, phase4 = field_handles

if is_admin and run_trigger:
    ctx = _pack_context(
        target_year,
        country,
        p_mod,
        s_mod,
        d_meth,
        schedule,
        sim_eng,
        crop_type,
        use_ml_surrogate,
        use_lstm,
        climate_scenario,
        climate_sensitivity,
        use_breeding,
    )

    with st.spinner("Evaluating integrated nexus (proxy or file-coupled)…"):
        if use_file_coupling:
            m0 = run_file_coupled_simulation(ctx, f_len, f_slp)
            metrics = metrics_tuple_from_dict(m0)
        else:
            metrics = run_integrated_engine(
                target_year,
                country,
                p_mod,
                s_mod,
                d_meth,
                schedule,
                sim_eng,
                f_len,
                f_slp,
                crop_type,
                use_ml_surrogate,
                use_lstm,
                climate_scenario,
                climate_sensitivity,
                use_breeding,
            )
        pop, supply, demand, ag_demand, ag_cons, yld, wp, temp, eff = metrics

    with st.spinner("Multi-objective optimization (pymoo ↔ engine)…"):
        if opt_eng == "NSGA-II":
            if use_file_coupling:
                X_pareto, F_pareto = pymoo_with_file_coupling(ctx, pop_size=28, n_gen=35)
            else:
                X_pareto, F_pareto = run_pymoo_nsga2(ctx, pop_size=40, n_gen=50)
            x_ga, f_ga, fitness_hist = None, None, None
        else:
            X_pareto, F_pareto = None, None
            x_ga, f_ga, fitness_hist = run_pymoo_ga_single_objective(ctx, pop_size=40, n_gen=50)

    with st.spinner("SALib + multi-parameter Monte Carlo…"):
        Si_wp, Y_wp = run_sobol_analysis(
            ctx, n_samples=sobol_n, calc_second_order=False, output="wp", seed=42
        )
        Si_ag, _ = run_sobol_analysis(
            ctx, n_samples=sobol_n, calc_second_order=False, output="ag_cons", seed=43
        )
        mc_df = monte_carlo_multiparam(ctx, n=mc_n, seed=42)
        morris_wp = run_morris_analysis(ctx, n_trajectories=30, output="wp", seed=44) if run_morris_ui else None

    # Store results in session state
    st.session_state.simulation_results = {
        'pop': pop, 'supply': supply, 'demand': demand,
        'ag_demand': ag_demand, 'ag_cons': ag_cons, 'yld': yld,
        'wp': wp, 'temp': temp, 'eff': eff,
        'X_pareto': X_pareto, 'F_pareto': F_pareto,
        'x_ga': x_ga, 'f_ga': f_ga, 'fitness_hist': fitness_hist,
        'Si_wp': Si_wp, 'Y_wp': Y_wp, 'Si_ag': Si_ag,
        'mc_df': mc_df, 'morris_wp': morris_wp,
        'use_file_coupling': use_file_coupling,
        'country': country, 'target_year': target_year,
        'crop_type': crop_type
    }
    st.session_state.simulation_run = True
    st.session_state.params_changed = False

if is_admin:
    assert phase1 is not None
    # ==================== PHASE 1: DRIVERS & PROJECTIONS ====================
    with phase1:
        st.subheader("🌍 Phase 1: Drivers & Projections")
        st.markdown("Climate scenarios, population projections, and water availability analysis")

        # Use sub-tabs within Phase 1 for organization
        p1_tab1, p1_tab2, p1_tab3 = st.tabs(["📊 Dashboard", "🌡️ Climate Downscaling", "👥 Population Analysis"])

        with p1_tab1:
            if st.session_state.get('simulation_run', False):
                results = st.session_state.simulation_results
                pop = results['pop']
                supply = results['supply']
                demand = results['demand']
                ag_demand = results['ag_demand']
                ag_cons = results['ag_cons']
                yld = results['yld']
                wp = results['wp']
                temp = results['temp']
                eff = results['eff']
                country = results['country']
                target_year = results['target_year']
                use_file_coupling = results['use_file_coupling']

                st.subheader(f"📈 Nexus Projections for {country} ({target_year})")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Projected Population", f"{pop:.2f} M")
                c2.metric("Total Water Supply", f"{supply:.2f} BCM")
                c3.metric("Total Water Demand", f"{demand:.2f} BCM")
                c4.metric("Temperature Rise", f"+{temp:.2f} °C")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Agri Water Demand", f"{ag_demand:.2f} BCM")
                c6.metric("Agri Water Consumed", f"{ag_cons:.2f} BCM")
                c7.metric("Projected Yield", f"{yld:.2f} t/ha")
                c8.metric("Water Productivity", f"{wp:.3f} kg/m³")

                if use_file_coupling:
                    st.success(
                        "Simulation path: **file-coupled** (template → `coupled_runner.py` → `sim_output.csv`). "
                        "Replace runner internals with SWAT/AquaCrop executables when available."
                    )
                else:
                    st.info(
                        "Simulation path: **in-memory proxy** (fast, for research testing). "
                        "Switch to **file-coupled** for external model integration."
                    )
            else:
                st.info("Please adjust parameters in the sidebar and click 'Run Simulation-Optimization' above to view results.")
                if st.session_state.get('params_changed', False):
                    st.warning("Sidebar parameters changed since last run — re-run simulation to refresh all phase results.")

        with p1_tab2:
            if CLIMATE_AVAILABLE:
                assert get_climate_downscaling is not None
                climate_results = get_climate_downscaling().render_climate_downscaling_interface()
                if climate_results:
                    st.session_state.climate_results = climate_results
            else:
                st.error("Climate Downscaling module not available. Please install required packages: netcdf4, xarray")

        with p1_tab3:
            st.subheader("👥 Population Analysis")
            st.info("Population projections and demographic analysis based on selected model")
            if st.session_state.get('simulation_run', False):
                results = st.session_state.simulation_results
                pop = results['pop']
                st.metric("Projected Population", f"{pop:.2f} Million")
                st.write(f"Population Model: {p_mod}")
                st.write(f"Target Year: {target_year}")

if is_admin:
    assert phase2 is not None
    # ==================== PHASE 2: SIMULATION & ANALYSIS ====================
    with phase2:
        st.subheader("🔬 Phase 2: Simulation & Analysis")
        st.markdown("Crop models, optimization, uncertainty analysis, and advanced analytics")

        p2_tab1, p2_tab2, p2_tab3, p2_tab4, p2_tab5 = st.tabs(
            ["🎯 Optimization", "🎲 Uncertainty", "🗺️ GIS", "🤖 ML & Economic", "⚙️ Irrigation Physics"]
        )

        with p2_tab1:
            if st.session_state.get('simulation_run', False):
                results = st.session_state.simulation_results
                X_pareto = results['X_pareto']
                F_pareto = results['F_pareto']
                x_ga = results['x_ga']
                f_ga = results['f_ga']
                fitness_hist = results['fitness_hist']
                ag_cons = results['ag_cons']
                eff = results['eff']

                st.subheader(f"🎯 Optimization Engine: {opt_eng}")
                st.caption(
                    "Single objective function wraps the nexus engine; pymoo evolves furrow length & slope. "
                    "NSGA-II minimizes water consumed and maximizes WP (Pareto front). **No random scatter.**"
                )
                opt_depth = ag_cons * 0.84
                opt_f_len = f_len * 1.1
                opt_eff_val = eff * 1.06

                if opt_eng == "Genetic Algorithm (GA)" and fitness_hist is not None:
                    gens = np.arange(1, len(fitness_hist) + 1)
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=gens,
                            y=fitness_hist,
                            mode="lines+markers",
                            name="Best ag_cons / WP",
                        )
                    )
                    fig.update_layout(
                        title="GA: convergence (minimize ag water per unit productivity)",
                        xaxis_title="Generation",
                        yaxis_title="Objective",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    if f_ga is not None and x_ga is not None and len(np.asarray(f_ga).ravel()):
                        xv = np.asarray(x_ga).ravel()
                        st.caption(f"Best candidate (f_len, f_slope): ({float(xv[0]):.1f} m, {float(xv[1]):.3f})")
                elif opt_eng == "NSGA-II" and X_pareto is not None and F_pareto is not None:
                    ag_c = F_pareto[:, 0]
                    wp_vals = -F_pareto[:, 1]
                    fl = X_pareto[:, 0]
                    fs = X_pareto[:, 1]
                    color_choice = st.radio("Color Pareto points by", ["Furrow length (m)", "Furrow slope (%)"], horizontal=True)
                    color = fl if color_choice.startswith("Furrow length") else fs
                    fig_pareto = px.scatter(
                        x=ag_c,
                        y=wp_vals,
                        color=color,
                        labels={"x": "Ag water consumed (BCM)", "y": "Water productivity index", "color": color_choice},
                        title="Pareto front (pymoo NSGA-II on analytical / file-coupled objective)",
                    )
                    st.plotly_chart(fig_pareto, use_container_width=True)
                    st.dataframe(
                        pd.DataFrame({"f_len_m": fl, "f_slope_pct": fs, "ag_cons": ag_c, "wp": wp_vals}),
                        use_container_width=True,
                    )

                st.markdown("### 💎 Reference parameter set (sidebar)")
                o1, o2, o3 = st.columns(3)
                o1.metric("Reference irrigation depth proxy", f"{opt_depth:.2f} mm")
                o2.metric("Reference furrow length", f"{opt_f_len:.1f} m")
                o3.metric("System efficiency", f"{opt_eff_val:.1%}")
            else:
                st.info("Please run simulation to view optimization results.")

        with p2_tab2:
            if st.session_state.get('simulation_run', False):
                results = st.session_state.simulation_results
                Si_wp = results['Si_wp']
                Si_ag = results['Si_ag']
                mc_df = results['mc_df']
                morris_wp = results['morris_wp']

                st.subheader("🎲 Uncertainty (MC) & global sensitivity (SALib)")
                st.write(
                    f"**Monte Carlo:** {mc_n} draws over the same parameter bounds as Sobol "
                    f"({', '.join(salib_problem_spec()['names'])}). **Sobol:** N={sobol_n} (first-order S1, total-order ST)."
                )

                fig_comb = make_subplots(
                    rows=2,
                    cols=1,
                    row_heights=[0.38, 0.62],
                    subplot_titles=("Sobol indices (WP): S1 and ST", "WP distribution (MC) with kernel density"),
                    vertical_spacing=0.12,
                )
                names = salib_problem_spec()["names"]
                fig_comb.add_bar(
                    x=names,
                    y=Si_wp["S1"],
                    name="S1 (WP)",
                    marker_color="#0056b3",
                    row=1,
                    col=1,
                )
                fig_comb.add_bar(
                    x=names,
                    y=Si_wp["ST"],
                    name="ST (WP)",
                    marker_color="#88c",
                    row=1,
                    col=1,
                )
                fig_comb.add_trace(
                    go.Histogram(x=mc_df["wp"], nbinsx=45, name="WP (MC)", marker_color="#002b5c", opacity=0.75),
                    row=2,
                    col=1,
                )
                fig_comb.update_layout(
                    height=720,
                    title_text="Overlay: Sobol sensitivity bars + WP uncertainty",
                    showlegend=True,
                    barmode="group",
                )
                fig_comb.update_yaxes(title_text="Index", row=1, col=1)
                fig_comb.update_yaxes(title_text="Count", row=2, col=1)
                st.plotly_chart(fig_comb, use_container_width=True)

                cleft, cright = st.columns(2)
                with cleft:
                    st.markdown("**Ag water consumed — Sobol (stress metric)**")
                    fig_ag = go.Figure()
                    fig_ag.add_bar(x=names, y=Si_ag["S1"], name="S1 (ag_cons)", marker_color="#c0392b")
                    fig_ag.add_bar(x=names, y=Si_ag["ST"], name="ST (ag_cons)", marker_color="#e74c3c")
                    fig_ag.update_layout(barmode="group", title="S1 and ST for agricultural water consumed")
                    st.plotly_chart(fig_ag, use_container_width=True)
                with cright:
                    st.markdown("**Correlation (MC sample)**")
                    num_cols = ["climate_sensitivity", "f_len", "f_slp", "temp_scale", "supply_perturb", "wp", "ag_cons", "yield"]
                    corr = mc_df[num_cols].corr()
                    fig_hm = px.imshow(
                        corr,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title="Parameter & output correlation heatmap",
                    )
                    fig_hm.update_traces(texttemplate="%{z:.2f}")
                    st.plotly_chart(fig_hm, use_container_width=True)

                if morris_wp is not None:
                    st.markdown("**Morris screening (ranking μ* vs σ)**")
                    mu_star = np.array(morris_wp["mu_star"])
                    sigma = np.array(morris_wp["sigma"])
                    fig_m = go.Figure()
                    fig_m.add_trace(
                        go.Scatter(
                            x=mu_star,
                            y=sigma,
                            mode="markers+text",
                            text=names,
                            textposition="top center",
                            marker=dict(size=12, color="#0056b3"),
                        )
                    )
                    fig_m.update_layout(
                        title="Morris: μ* vs σ (screening)",
                        xaxis_title="μ*",
                        yaxis_title="σ",
                    )
                    st.plotly_chart(fig_m, use_container_width=True)
            else:
                st.info("Please run simulation to view uncertainty analysis results.")

        with p2_tab3:
            if GIS_AVAILABLE:
                assert get_gis_integration is not None
                gis_results = get_gis_integration().render_gis_interface()
                if gis_results and len(gis_results) == 2:
                    site_data, map_obj = gis_results
                    if site_data:
                        st.session_state.gis_site_data = site_data
            else:
                st.error("GIS Integration module not available. Please install required packages: folium, geopandas, shapely")

        with p2_tab4:
            ml_col, econ_col = st.columns(2)
            with ml_col:
                if ML_AVAILABLE:
                    assert get_ml_surrogate_manager is not None
                    ml_results = get_ml_surrogate_manager().render_ml_interface()
                    if ml_results:
                        st.session_state.ml_results = ml_results
                else:
                    st.error("ML Surrogates module not available. Please install required packages: scikit-learn, shap, lime")

            with econ_col:
                if ECONOMIC_AVAILABLE:
                    assert get_economic_optimizer is not None
                    economic_results = get_economic_optimizer().render_economic_interface()
                    if economic_results:
                        st.session_state.economic_results = economic_results
                else:
                    st.error("Economic Optimization module not available. Please install required packages: pymoo")

        with p2_tab5:
            if IRRIGATION_AVAILABLE:
                assert get_irrigation_physics is not None
                irrigation_results = get_irrigation_physics().render_irrigation_physics_interface()
                if irrigation_results:
                    st.session_state.irrigation_results = irrigation_results
            else:
                st.error("Irrigation Physics module not available. Please install required packages: scipy")

# ==================== PHASE 3: IRRIGATION SCHEDULING (NEW) ====================
with phase3:
    st.subheader("💧 Phase 3: Irrigation Scheduling")
    st.markdown("Estimate root-zone water deficit and irrigation needs from field measurements.")

    if st.session_state.get('simulation_run', False):
        # Initialize irrigation scheduler
        if 'irrigation_scheduler' not in st.session_state:
            st.session_state.irrigation_scheduler = IrrigationScheduler()

        scheduler = st.session_state.irrigation_scheduler

        st.write("**Field Soil-Water and Irrigation Inputs**")
        col1, col2, col3 = st.columns(3)
        with col1:
            soil_water_capacity = st.number_input(
                "Target Soil Water Capacity (mm)",
                value=100.0,
                min_value=0.0,
                key="phase3_soil_water_capacity",
                help="Target water level in the root zone, such as field capacity.",
            )
            soil_moisture = st.number_input(
                "Current Soil Moisture (mm)",
                value=30.0,
                min_value=0.0,
                key="phase3_soil_moisture",
            )
        with col2:
            expected_rainfall = st.number_input(
                "Expected Rainfall (mm)", value=5.0, min_value=0.0, key="phase3_expected_rainfall"
            )
            available_water = st.number_input(
                "Irrigation Supply Available (mm)",
                value=100.0,
                min_value=0.0,
                key="phase3_available_water",
            )
        with col3:
            field_area_m2 = st.number_input(
                "Irrigated Field Area (m²)",
                value=0.0,
                min_value=0.0,
                key="phase3_field_area",
                help="Enter the irrigated area to calculate application duration.",
            )
            flow_rate = st.number_input(
                "Irrigation Flow Rate (L/s)",
                value=2.5,
                min_value=0.01,
                key="phase3_flow_rate",
            )

        if st.button("Generate Irrigation Schedule"):
            schedule = scheduler.calculate_irrigation_schedule(
                soil_water_capacity=soil_water_capacity,
                soil_moisture=soil_moisture,
                rainfall_expected=expected_rainfall,
                water_availability=available_water,
                field_area_m2=field_area_m2,
                flow_rate_lps=flow_rate,
            )

            st.session_state.current_schedule = schedule

            # Store in database
            _execute_db(
                """INSERT INTO irrigation_schedules
                (timestamp, location, crop, irrigation_amount, timing, duration, flow_rate, stress_level, status)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    schedule['timestamp'],
                    country,
                    crop_type,
                    schedule['net_irrigation_mm'],
                    schedule['timing'],
                    schedule['duration_minutes'],
                    schedule['flow_rate'],
                    schedule['stress_level'],
                    'pending'
                )
            )
            db_conn.commit()

        schedule_inputs_changed = (
            'current_schedule' in st.session_state
            and (
                st.session_state.current_schedule.get('soil_moisture_current') != soil_moisture
                or st.session_state.current_schedule.get('rainfall_expected_mm') != expected_rainfall
                or st.session_state.current_schedule.get('water_availability_mm') != available_water
                or st.session_state.current_schedule.get('soil_water_capacity_mm') != soil_water_capacity
                or st.session_state.current_schedule.get('field_area_m2') != field_area_m2
                or st.session_state.current_schedule.get('flow_rate') != flow_rate
            )
        )
        if schedule_inputs_changed:
            st.info("Scheduling inputs changed. Generate a new schedule to update the details.")

        if 'current_schedule' in st.session_state and not schedule_inputs_changed:
            schedule = st.session_state.current_schedule
            st.success("✅ Irrigation schedule generated successfully!")

            st.write("**Irrigation Schedule Details:**")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Irrigation Amount", f"{schedule['net_irrigation_mm']:.1f} mm")
            s2.metric("Timing", schedule['timing'])
            duration = schedule['duration_minutes']
            s3.metric(
                "Duration",
                f"{duration:.1f} min" if duration is not None else "Enter field area",
            )
            s4.metric("Flow Rate", f"{schedule['flow_rate']:.1f} L/s")

            st.metric(
                "Soil-Water Deficit Proxy",
                f"{schedule['stress_level']:.2f}",
                delta=f"Current soil moisture: {schedule['soil_moisture_current']:.1f} mm",
                help="Estimated root-zone deficit fraction; this is not a crop-model stress output.",
            )

            # Visualization
            st.caption("Water-balance estimate; crop-specific stress is not modeled here.")
            stress_data = {
                'Target Soil Water (mm)': schedule['soil_water_capacity_mm'],
                'Current Soil Moisture (mm)': schedule['soil_moisture_current'],
                'Expected Rainfall (mm)': schedule['rainfall_expected_mm'],
                'Recommended Irrigation (mm)': schedule['net_irrigation_mm'],
                'Irrigation Supply (mm)': schedule['water_availability_mm'],
            }

            fig_stress = px.bar(
                x=list(stress_data.keys()),
                y=list(stress_data.values()),
                title="Root-Zone Water Balance",
                color=list(stress_data.values()),
                color_continuous_scale='Blues',
                labels={"x": "Water-balance input", "y": "Water depth (mm)", "color": "Water depth (mm)"},
            )
            st.plotly_chart(fig_stress, use_container_width=True)
    else:
        if is_field_user:
            st.info("Enter field measurements to generate a local irrigation schedule. Research simulations and optimization remain restricted to Admin/Researcher accounts.")
            if "irrigation_scheduler" not in st.session_state:
                st.session_state.irrigation_scheduler = IrrigationScheduler()
            field_col1, field_col2, field_col3 = st.columns(3)
            with field_col1:
                field_capacity = st.number_input(
                    "Target Soil Water Capacity (mm)",
                    min_value=0.0,
                    value=100.0,
                    key="field_soil_water_capacity",
                    help="Target root-zone water level, such as field capacity.",
                )
                field_moisture = st.number_input("Current Soil Moisture (mm)", min_value=0.0, value=25.0, key="field_soil_moisture")
            with field_col2:
                field_rain = st.number_input("Expected Rainfall (mm)", min_value=0.0, value=0.0, key="field_expected_rain")
                field_water = st.number_input("Irrigation Supply Available (mm)", min_value=0.0, value=100.0, key="field_available_water")
            with field_col3:
                field_area = st.number_input(
                    "Irrigated Field Area (m²)",
                    min_value=0.0,
                    value=0.0,
                    key="field_area_m2",
                    help="Enter the irrigated area to calculate application duration.",
                )
                field_flow_rate = st.number_input(
                    "Irrigation Flow Rate (L/s)", min_value=0.01, value=2.5, key="field_flow_rate"
                )
            if st.button("Generate Field Irrigation Schedule", key="field_generate_schedule"):
                field_schedule = st.session_state.irrigation_scheduler.calculate_irrigation_schedule(
                    soil_water_capacity=field_capacity,
                    soil_moisture=field_moisture,
                    rainfall_expected=field_rain,
                    water_availability=field_water,
                    field_area_m2=field_area,
                    flow_rate_lps=field_flow_rate,
                )
                st.session_state.current_schedule = field_schedule
                _execute_db(
                    """INSERT INTO irrigation_schedules
                    (timestamp, location, crop, irrigation_amount, timing, duration, flow_rate, stress_level, status)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        field_schedule["timestamp"], country, crop_type,
                        field_schedule["net_irrigation_mm"], field_schedule["timing"],
                        field_schedule["duration_minutes"], field_schedule["flow_rate"],
                        field_schedule["stress_level"], "pending",
                    ),
                )
                db_conn.commit()
                st.success("Field irrigation schedule generated and saved.")
            if "current_schedule" in st.session_state:
                current_field_schedule = st.session_state.current_schedule
                st.metric("Recommended Irrigation", f"{current_field_schedule['net_irrigation_mm']:.1f} mm")
                duration = current_field_schedule["duration_minutes"]
                duration_text = (
                    f"{duration:.1f} min" if duration is not None else "Enter field area to calculate"
                )
                st.write(
                    f"Timing: **{current_field_schedule['timing']}** · "
                    f"Duration: **{duration_text}** · "
                    f"Flow: **{current_field_schedule['flow_rate']:.1f} L/s**"
                )
            saved_schedules = _read_db_query("SELECT * FROM irrigation_schedules ORDER BY timestamp DESC")
            if not saved_schedules.empty:
                st.subheader("Saved Irrigation Schedules")
                st.dataframe(saved_schedules, use_container_width=True)
        else:
            st.info("Please complete Phase 1 (Drivers & Projections) and Phase 2 (Simulation & Analysis) first.")

# ==================== PHASE 4: FIELD OPERATIONS (NEW) ====================
with phase4:
    st.subheader("🚜 Phase 4: Field Operations")
    st.markdown("NEW: Field communication, SMS messaging, and hardware control for irrigation automation")

    p4_tab1, p4_tab2, p4_tab3, p4_tab4, p4_tab5 = st.tabs(
        [
            "📝 Field Observation Entry",
            "📡 Server-Field Communication",
            "⚙️ Hardware Control",
            "🌱 Agricultural Sensor Monitoring",
            "📱 Mobile App Sync"
        ]
    )

    with p4_tab1:
        st.subheader("📝 Detailed PhD Field Observation Entry")
        with st.form("field"):
            col1, col2 = st.columns(2)
            with col1:
                f_loc = st.text_input("Site Location (e.g., Kulumsa, Wonji)")
                f_soil = st.selectbox("Soil Type", ["Clay", "Sandy Loam", "Silt Clay", "Loam"])
                f_temp = st.number_input("Max Temp Observed (°C)", value=25.0)
                f_hum = st.number_input("Humidity (%)", value=60.0)
            with col2:
                f_rain = st.number_input("Daily Rainfall (mm)", value=0.0)
                f_irr_obs = st.number_input("Irrigation Applied (mm)", value=45.0)
                f_slp_obs = st.number_input("Observed Slope (%)", value=0.1)
                f_stage = st.selectbox("Crop Growth Stage", ["Initial", "Development", "Mid-Season", "Late-Season"])
            f_notes = st.text_area("Field Researcher Notes")
            if st.form_submit_button("Sync Entry to Master Database"):
                _execute_db(
                    """INSERT INTO field_data (timestamp, location, crop, soil_type, temp_max,
                                humidity, rainfall, irrigation, furrow_length, furrow_slope, notes)
                                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                        f_loc,
                        crop_type,
                        f_soil,
                        f_temp,
                        f_hum,
                        f_rain,
                        f_irr_obs,
                        f_len,
                        f_slp_obs,
                        f_notes,
                    ),
                )
                db_conn.commit()
                st.success("Detailed observation synchronized successfully.")

    with p4_tab2:
        st.subheader("📡 Server-Field Communication")
        st.markdown("Manage irrigation requests, instructions, and SMS communication with field users")

        # Initialize communication managers
        if 'field_comm_manager' not in st.session_state:
            st.session_state.field_comm_manager = FieldCommunicationManager()
        if 'sms_module' not in st.session_state:
            st.session_state.sms_module = SMSCommunicationModule()

        comm_manager = st.session_state.field_comm_manager
        sms_module = st.session_state.sms_module

        # Field Request Input
        st.write("**Receive Field Irrigation Request**")
        with st.form("field_request"):
            fr_col1, fr_col2 = st.columns(2)
            with fr_col1:
                farmer_id = st.text_input("Farmer ID", value="FARM001")
                location = st.text_input("Field Location", value="Kulumsa")
            with fr_col2:
                crop_stage = st.selectbox("Crop Stage", ["Initial", "Development", "Mid-Season", "Late-Season"])
                soil_moisture = st.number_input("Current Soil Moisture (mm)", value=25.0)

            if st.form_submit_button("Receive Request"):
                request = comm_manager.add_field_request(farmer_id, location, crop_stage, soil_moisture)
                # Store in database
                _execute_db(
                    """INSERT INTO field_requests
                    (timestamp, farmer_id, location, crop_stage, soil_moisture, status)
                    VALUES (?,?,?,?,?,?)""",
                    (request['timestamp'], farmer_id, location, crop_stage, soil_moisture, 'pending')
                )
                db_conn.commit()
                st.success(f"Request received from {farmer_id} at {location}")

        # Pending Requests
        st.write("**Pending Field Requests**")
        if comm_manager.pending_requests:
            for req in comm_manager.pending_requests[-5:]:  # Show last 5
                st.json(req)
        else:
            st.info("No pending requests")

        # Generate and Send Instructions
        st.write("**Generate Irrigation Instructions**")
        if comm_manager.pending_requests and 'current_schedule' in st.session_state:
            if st.button("Generate Instructions for Pending Requests"):
                for req in comm_manager.pending_requests:
                    instruction = comm_manager.generate_irrigation_instruction(
                        req,
                        st.session_state.current_schedule
                    )
                    st.success(f"Instruction generated for {req['farmer_id']}")

        # SMS Communication
        st.write("**SMS Communication**")
        if comm_manager.irrigation_instructions:
            with st.form("send_sms"):
                phone_number = st.text_input("Phone Number", value="+251955006516")
                if st.form_submit_button("Send SMS Instruction"):
                    for instruction in comm_manager.irrigation_instructions[-1:]:  # Send latest
                        message = sms_module.format_irrigation_instruction_sms(instruction)
                        sms = sms_module.queue_sms(phone_number, message)
                        sent = sms_module.send_sms(sms)
                        st.success(f"SMS sent to {phone_number}")
                        st.text_area("Message Content", message, height=100)

        # SMS History
        st.write("**SMS History**")
        if sms_module.sms_history:
            for sms in sms_module.sms_history[-5:]:
                st.text(f"To: {sms['phone_number']} | Status: {sms['status']}")
                st.text(f"Message: {sms['message'][:50]}...")
                st.divider()

    with p4_tab3:
        st.subheader("⚙️ Hardware Control Interface")
        st.markdown("Control furrow irrigation hardware for automated irrigation")

        # Initialize hardware interface
        if 'hardware_interface' not in st.session_state:
            st.session_state.hardware_interface = HardwareControlInterface()

        hw_interface = st.session_state.hardware_interface

        # Connection Status
        st.write("**Hardware Connection Status**")
        conn_status = hw_interface.hardware_status['connection_status']
        if conn_status == 'connected':
            st.success(f"✅ Connected to Device: {hw_interface.hardware_status.get('device_id', 'Unknown')}")
        else:
            st.warning("⚠️ Hardware not connected")

        # Connect/Disconnect
        with st.form("hardware_connection"):
            device_id = st.text_input("Device ID", value="HW001")
            if st.form_submit_button("Connect Hardware"):
                hw_interface.connect_hardware(device_id)
                # Log to database
                _execute_db(
                    """INSERT INTO hardware_control_log
                    (timestamp, device_id, action, parameters, status)
                    VALUES (?,?,?,?,?)""",
                    (datetime.now().isoformat(), device_id, 'connect', '{}', 'success')
                )
                db_conn.commit()
                st.rerun()

        # Valve Control
        if hw_interface.hardware_status['connection_status'] == 'connected':
            st.write("**Valve Control**")
            col_hw1, col_hw2 = st.columns(2)
            with col_hw1:
                flow_rate = st.number_input("Flow Rate (L/s)", value=2.5, min_value=0.1, max_value=10.0)
                if st.button("Open Valve"):
                    hw_interface.open_valve(flow_rate)
                    _execute_db(
                        """INSERT INTO hardware_control_log
                        (timestamp, device_id, action, parameters, status)
                        VALUES (?,?,?,?,?)""",
                        (datetime.now().isoformat(), device_id, 'open_valve',
                         json.dumps({'flow_rate': flow_rate}), 'success')
                    )
                    db_conn.commit()
                    st.success("Valve opened")

            with col_hw2:
                if st.button("Close Valve"):
                    hw_interface.close_valve()
                    _execute_db(
                        """INSERT INTO hardware_control_log
                        (timestamp, device_id, action, parameters, status)
                        VALUES (?,?,?,?,?)""",
                        (datetime.now().isoformat(), device_id, 'close_valve', '{}', 'success')
                    )
                    db_conn.commit()
                    st.success("Valve closed")

            # Automated Irrigation
            st.write("**Automated Irrigation**")
            if 'current_schedule' in st.session_state:
                schedule = st.session_state.current_schedule
                if schedule['duration_minutes'] is None:
                    st.warning("Enter the irrigated field area in Phase 3 before executing timed irrigation.")
                if st.button(
                    "Execute Automated Irrigation",
                    disabled=schedule['duration_minutes'] is None,
                ):
                    hw_interface.automate_irrigation(
                        schedule['net_irrigation_mm'],
                        schedule['duration_minutes'],
                        schedule['flow_rate']
                    )
                    _execute_db(
                        """INSERT INTO hardware_control_log
                        (timestamp, device_id, action, parameters, status)
                        VALUES (?,?,?,?,?)""",
                        (datetime.now().isoformat(), device_id, 'automate_irrigation',
                         json.dumps(schedule), 'scheduled')
                    )
                    db_conn.commit()
                    st.success("Automated irrigation scheduled")

            # Hardware Status Display
            st.write("**Current Hardware Status**")
            status_df = pd.DataFrame([hw_interface.hardware_status])
            st.dataframe(status_df, use_container_width=True)

            # Control Log
            st.write("**Control Log**")
            if hw_interface.control_log:
                log_df = pd.DataFrame(hw_interface.control_log[-10:])
                st.dataframe(log_df, use_container_width=True)

    # ==========================================
    # AGRICULTURAL SENSOR MONITORING
    # Step 5.3.3 - Read-only SQLite Integration
    # ==========================================
    @st.fragment(run_every="15s")
    def render_sensor_monitoring():
        st.subheader("🌱 Agricultural Sensor Monitoring")
        st.markdown(
            "Monitor soil moisture and weather observations "
            "collected from agricultural field sensors."
        )

        st.info(
            "Data source: TOKUMA persistent SQLite sensor database. "
            "Initial measurements are simulated test observations."
        )

        if st.button("🔄 Refresh Sensor Data", key="refresh_sensor_dashboard"):
            st.rerun()

        sensor_conn = None

        try:
            # Open the sensor database in read-only mode.
            sensor_conn = sqlite3.connect(
                "file:tokuma_sensor_data.db?mode=ro",
                uri=True
            )

            # Retrieve available experimental fields.
            fields_df = pd.read_sql_query(
                """
                SELECT DISTINCT field_id
                FROM sensor_readings
                ORDER BY field_id
                """,
                sensor_conn
            )

            if fields_df.empty:
                st.warning("No sensor readings are available yet.")

            else:
                field_options = fields_df["field_id"].tolist()

                selected_field = st.selectbox(
                    "Select Experimental Field",
                    field_options,
                    key="sensor_monitoring_field"
                )

                # Retrieve historical readings for the selected field.
                history_df = pd.read_sql_query(
                    """
                    SELECT *
                    FROM sensor_readings
                    WHERE field_id = ?
                    ORDER BY timestamp ASC
                    """,
                    sensor_conn,
                    params=(selected_field,)
                )

                if history_df.empty:
                    st.warning(
                        f"No sensor records found for {selected_field}."
                    )

                else:
                    latest = history_df.iloc[-1]

                    # STEP 5.3.11 - Prepare shared sensor history
                    # for anomaly detection and irrigation decisions.
                    sensor_readings = history_df.to_dict(
                        orient="records"
                    )

                    st.markdown("---")
                    st.markdown(
                        f"### 📡 Latest Sensor Readings: {selected_field}"
                    )

                    def sensor_display(value, unit):
                        if pd.isna(value):
                            return "—"
                        return f"{float(value):.1f} {unit}"

                    # Display the four sensor measurements.
                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        "Soil Moisture",
                        sensor_display(latest["soil_moisture"], "%")
                    )

                    col2.metric(
                        "Air Temperature",
                        sensor_display(latest["air_temperature"], "°C")
                    )

                    col3.metric(
                        "Relative Humidity",
                        sensor_display(latest["relative_humidity"], "%")
                    )

                    col4.metric(
                        "Rainfall",
                        sensor_display(latest["rainfall"], "mm")
                    )

                    st.caption(
                        f"Latest observation timestamp: {latest['timestamp']}"
                    )

                    # ==========================================
                    # STEP 5.3.10 - SENSOR ANOMALY DETECTION
                    # ==========================================

                    st.markdown("---")
                    st.markdown(
                        "### 🚨 Sensor Anomaly Detection and Early Warning"
                    )

                    st.caption(
                        "Anomaly detection is performed using historical "
                        "readings from the selected experimental field. "
                        "Detection thresholds are provisional software "
                        "testing parameters."
                    )

                    try:
                        # Run all four anomaly detection methods.
                        anomaly_results = run_sensor_anomaly_detection(
                            sensor_readings
                        )

                        anomaly_summary = anomaly_results["summary"]

                        severity_counts = (
                            anomaly_summary["severity_counts"]
                        )

                        # Display anomaly summary.
                        acol1, acol2, acol3, acol4 = st.columns(4)

                        acol1.metric(
                            "Total Anomalies",
                            anomaly_summary["total_anomalies"]
                        )

                        acol2.metric(
                            "Warnings",
                            severity_counts["WARNING"]
                        )

                        acol3.metric(
                            "Critical Alerts",
                            severity_counts["CRITICAL"]
                        )

                        acol4.metric(
                            "Affected Sensors",
                            anomaly_summary["affected_sensor_count"]
                        )

                        # ==========================================
                        # DISPLAY ANOMALIES BY SEVERITY
                        # ==========================================

                        anomaly_df = pd.DataFrame(
                            anomaly_results["anomalies"]
                        )

                        if not anomaly_df.empty:

                            info_df = anomaly_df[
                                anomaly_df["severity"] == "INFO"
                            ]

                            warning_df = anomaly_df[
                                anomaly_df["severity"] == "WARNING"
                            ]

                            critical_df = anomaly_df[
                                anomaly_df["severity"] == "CRITICAL"
                            ]

                            display_columns = [
                                "sensor_id",
                                "parameter",
                                "observed_value",
                                "reference_value",
                                "anomaly_type",
                                "severity",
                                "message",
                                "recommendation"
                            ]

                            # Historical statistical information.
                            if not info_df.empty:

                                st.info(
                                    f"{len(info_df)} historical "
                                    "statistical deviation(s) detected. "
                                    "These require interpretation and "
                                    "do not independently confirm "
                                    "sensor malfunction."
                                )

                                st.markdown(
                                    "**Historical Statistical "
                                    "Observations**"
                                )

                                st.dataframe(
                                    info_df[display_columns],
                                    use_container_width=True
                                )

                            # Operational warnings.
                            if not warning_df.empty:

                                st.warning(
                                    f"{len(warning_df)} operational "
                                    "warning(s) require investigation."
                                )

                                st.markdown(
                                    "**Operational Warnings**"
                                )

                                st.dataframe(
                                    warning_df[display_columns],
                                    use_container_width=True
                                )

                            # Critical alerts.
                            if not critical_df.empty:

                                st.error(
                                    f"{len(critical_df)} critical "
                                    "alert(s) require immediate review."
                                )

                                st.markdown(
                                    "**Critical Alerts**"
                                )

                                st.dataframe(
                                    critical_df[display_columns],
                                    use_container_width=True
                                )

                            affected_sensors = (
                                anomaly_summary["affected_sensors"]
                            )

                            st.caption(
                                "Sensors with historical or current "
                                "detection records: "
                                + ", ".join(affected_sensors)
                            )

                        else:

                            st.success(
                                "No anomalies detected in the "
                                "available sensor history."
                            )
                        st.caption(
                            "Detection methods: sudden change, statistical "
                            "outlier, flatline, and delayed reading."
                        )

                    except Exception as error:

                        st.error(
                            "Unable to complete sensor anomaly detection: "
                            f"{error}"
                        )

                    # ==========================================
                    # END STEP 5.3.10 ANOMALY DASHBOARD
                    # ==========================================
                    # ==========================================
                    # STEP 5.3.11 - SENSOR-BASED IRRIGATION
                    # DECISION SUPPORT
                    # ==========================================

                    st.markdown("---")
                    st.markdown(
                        "### Sensor-Based Irrigation Decision Support"
                    )

                    try:
                        latest_reading = latest.to_dict()

                        decision_result = (
                            evaluate_sensor_irrigation_decision(
                                latest_reading,
                                sensor_readings
                            )
                        )

                        decision_status = decision_result.get(
                            "decision_status",
                            "INSUFFICIENT_DATA"
                        )

                        if decision_status == "SENSOR_DATA_ACCEPTABLE":
                            st.success(
                                "Sensor observation passed the current "
                                "decision-engine checks."
                            )

                        elif decision_status == "SENSOR_REVIEW_REQUIRED":
                            st.warning(
                                "Sensor observation requires review "
                                "before irrigation assessment."
                            )

                        else:
                            st.error(
                                "Insufficient data for a reliable "
                                "sensor-based decision."
                            )

                        # Display decision and observation information.
                        dcol1, dcol2, dcol3, dcol4 = st.columns(4)

                        dcol1.metric(
                            "Decision Status",
                            decision_status
                        )

                        dcol2.metric(
                            "Quality Status",
                            decision_result.get(
                                "quality_status",
                                "NOT ASSESSED"
                            )
                        )

                        dcol3.metric(
                            "Anomaly Status",
                            decision_result.get(
                                "anomaly_status",
                                "NOT ASSESSED"
                            )
                        )

                        observation_age = decision_result.get(
                            "observation_age_seconds"
                        )

                        age_display = (
                            f"{observation_age:.1f} s"
                            if observation_age is not None
                            else "Not available"
                        )

                        dcol4.metric(
                            "Observation Age",
                            age_display
                        )

                        # Display irrigation outputs only when available.
                        irrigation_amount = decision_result.get(
                            "irrigation_amount_mm"
                        )

                        irrigation_duration = decision_result.get(
                            "irrigation_duration_minutes"
                        )

                        st.markdown("**Irrigation Assessment**")

                        icol1, icol2 = st.columns(2)

                        icol1.metric(
                            "Estimated Irrigation Amount",
                            (
                                f"{irrigation_amount:.2f} mm"
                                if irrigation_amount is not None
                                else "Not available"
                            )
                        )

                        icol2.metric(
                            "Estimated Irrigation Duration",
                            (
                                f"{irrigation_duration:.1f} minutes"
                                if irrigation_duration is not None
                                else "Not available"
                            )
                        )

                        # Explain the decision.
                        decision_reasons = decision_result.get(
                            "decision_reasons",
                            []
                        )

                        st.markdown("**Decision Explanation**")

                        if decision_reasons:
                            for reason in decision_reasons:
                                st.write(f"- {reason}")
                        else:
                            st.write(
                                "No additional decision explanation "
                                "was returned."
                            )

                        st.info(
                            "Scientific limitation: current soil-moisture "
                            "readings are treated as uncalibrated percentages. "
                            "They are not converted into volumetric water "
                            "content or soil-water depth. Calibrated soil "
                            "water-retention parameters and crop-specific "
                            "information are required before calculating "
                            "irrigation depth or duration."
                        )

                        st.caption(
                            "Decision support only. Automatic irrigation "
                            "is disabled; no valve or pump is operated."
                        )

                    except Exception as error:
                        st.error(
                            "Unable to complete sensor-based irrigation "
                            "decision support: "
                            f"{error}"
                        )

                    # ==========================================
                    # END STEP 5.3.11 IRRIGATION DECISION SUPPORT
                    # ==========================================

                    # ==========================================
                    # STEP 5.3.8 - SENSOR QUALITY AND RELIABILITY
                    # ==========================================

                    st.markdown("---")
                    st.markdown(
                        "### 🛡️ Sensor Data Quality and Reliability"
                    )

                    quality_log_path = (
                        Path(__file__).resolve().parent
                        / "tokuma_sensor_quality_log.jsonl"
                    )

                    quality_records = []

                    if quality_log_path.exists():

                        try:
                            with open(
                                quality_log_path,
                                "r",
                                encoding="utf-8"
                            ) as quality_file:

                                for line_number, line in enumerate(
                                    quality_file,
                                    start=1
                                ):
                                    if not line.strip():
                                        continue

                                    try:
                                        record = json.loads(line)

                                        if (
                                            record.get("field_id")
                                            == selected_field
                                        ):
                                            quality_records.append(record)

                                    except json.JSONDecodeError:
                                        st.warning(
                                            "A malformed quality-log entry "
                                            f"was skipped (line {line_number})."
                                        )

                        except OSError as error:
                            st.error(
                                f"Unable to read sensor quality log: {error}"
                            )

                    else:
                        st.info(
                            "No sensor quality assessment log is available yet."
                        )

                    if quality_records:

                        quality_df = pd.DataFrame(quality_records)

                        # Normalize quality status values.
                        quality_df["quality_status"] = (
                            quality_df["quality_status"]
                            .fillna("UNKNOWN")
                            .astype(str)
                            .str.upper()
                        )

                        acceptable_count = int(
                            (quality_df["quality_status"] == "ACCEPTABLE").sum()
                        )

                        incomplete_count = int(
                            (quality_df["quality_status"] == "INCOMPLETE").sum()
                        )

                        review_count = int(
                            (quality_df["quality_status"] == "REVIEW").sum()
                        )

                        total_quality = len(quality_df)

                        acceptance_rate = (
                            acceptable_count / total_quality
                        ) * 100

                        # Quality assessment summary.
                        st.markdown(
                            f"**Quality assessment summary: {selected_field}**"
                        )

                        qcol1, qcol2, qcol3, qcol4 = st.columns(4)

                        qcol1.metric(
                            "Total Assessments",
                            total_quality
                        )

                        qcol2.metric(
                            "Acceptable",
                            acceptable_count
                        )

                        qcol3.metric(
                            "Incomplete",
                            incomplete_count
                        )

                        qcol4.metric(
                            "Review Required",
                            review_count
                        )

                        st.metric(
                            "Acceptable Assessment Rate",
                            f"{acceptance_rate:.1f}%"
                        )

                        # Explain quality results.
                        if review_count > 0:
                            st.error(
                                f"{review_count} assessment(s) require "
                                "quality review."
                            )

                        if incomplete_count > 0:
                            st.warning(
                                f"{incomplete_count} assessment(s) contain "
                                "incomplete optional sensor information."
                            )

                        if review_count == 0 and incomplete_count == 0:
                            st.success(
                                "All recorded assessments are acceptable."
                            )

                        # Display quality flags.
                        st.markdown("#### Quality Flags and Warnings")

                        flag_records = []

                        for record in quality_records:

                            flags = record.get("quality_flags", [])

                            if not isinstance(flags, list):
                                flags = []

                            for flag in flags:
                                flag_records.append({
                                    "Sensor ID": record.get(
                                        "sensor_id", "Unknown"
                                    ),
                                    "Quality Status": record.get(
                                        "quality_status", "UNKNOWN"
                                    ),
                                    "Quality Flag": str(flag),
                                    "Assessment Time": record.get(
                                        "assessment_timestamp", "Unknown"
                                    )
                                })

                        if flag_records:
                            flags_df = pd.DataFrame(flag_records)

                            st.dataframe(
                                flags_df,
                                use_container_width=True,
                                hide_index=True
                            )

                        else:
                            st.info(
                                "No quality flags were recorded for this field."
                            )
                        # ==========================================
                        # STEP 5.3.9.2 - SENSOR-LEVEL RELIABILITY
                        # ==========================================

                        st.markdown("---")
                        st.markdown(
                            "### Individual Sensor Reliability Analysis"
                        )

                        # Prepare sensor identification.
                        reliability_df = quality_df.copy()

                        reliability_df["sensor_id"] = (
                            reliability_df["sensor_id"]
                            .fillna("Unknown")
                            .astype(str)
                            .replace("", "Unknown")
                        )

                        # Calculate quality indicators per sensor.
                        reliability_records = []

                        for sensor_id, sensor_group in (
                            reliability_df.groupby("sensor_id")
                        ):

                            sensor_total = len(sensor_group)

                            sensor_acceptable = int(
                                (
                                    sensor_group["quality_status"]
                                    == "ACCEPTABLE"
                                ).sum()
                            )

                            sensor_incomplete = int(
                                (
                                    sensor_group["quality_status"]
                                    == "INCOMPLETE"
                                ).sum()
                            )

                            sensor_review = int(
                                (
                                    sensor_group["quality_status"]
                                    == "REVIEW"
                                ).sum()
                            )

                            sensor_acceptance_rate = (
                                sensor_acceptable / sensor_total
                            ) * 100 if sensor_total > 0 else 0

                            # Generate a descriptive monitoring note.
                            if sensor_review > 0:
                                monitoring_note = (
                                    "Review flags recorded"
                                )

                            elif sensor_incomplete > 0:
                                monitoring_note = (
                                    "Incomplete optional data recorded"
                                )

                            else:
                                monitoring_note = (
                                    "No review or incomplete flags recorded"
                                )

                            reliability_records.append({

                                "Sensor ID": sensor_id,

                                "Total Assessments": sensor_total,

                                "Acceptable": sensor_acceptable,

                                "Incomplete": sensor_incomplete,

                                "Review Required": sensor_review,

                                "Acceptance Rate (%)": round(
                                    sensor_acceptance_rate, 1
                                ),

                                "Monitoring Note": monitoring_note

                            })

                        # Display sensor-level reliability table.
                        if reliability_records:

                            sensor_reliability_df = pd.DataFrame(
                                reliability_records
                            )

                            st.markdown(
                                "#### Sensor-Level Quality Performance"
                            )

                            st.dataframe(
                                sensor_reliability_df,
                                use_container_width=True,
                                hide_index=True
                            )

                            # Compare acceptance rates across sensors.
                            st.markdown(
                                "#### Sensor Acceptance Rate Comparison"
                            )

                            reliability_chart = px.bar(
                                sensor_reliability_df,
                                x="Sensor ID",
                                y="Acceptance Rate (%)",
                                color="Acceptance Rate (%)",
                                hover_data=[
                                    "Total Assessments",
                                    "Acceptable",
                                    "Incomplete",
                                    "Review Required"
                                ],
                                title=(
                                    "Acceptable Assessment Rate "
                                    "by Sensor"
                                ),
                                text="Acceptance Rate (%)"
                            )

                            reliability_chart.update_layout(
                                yaxis_title="Acceptable Assessments (%)",
                                xaxis_title="Sensor Identification",
                                yaxis_range=[0, 100]
                            )

                            reliability_chart.update_traces(
                                texttemplate="%{text:.1f}%",
                                textposition="outside"
                            )

                            st.plotly_chart(
                                reliability_chart,
                                use_container_width=True
                            )

                            st.caption(
                                "Interpretation: Acceptance rate is the "
                                "percentage of recorded assessments "
                                "classified as ACCEPTABLE. It is a "
                                "descriptive quality indicator, not a "
                                "direct measurement of sensor accuracy. "
                                "Interpret percentages alongside the "
                                "number of assessments available."
                            )

                        else:

                            st.info(
                                "No sensor-level reliability records "
                                "are available for this field."
                            )
                        # ==========================================
                        # STEP 5.3.8.9 - INDIVIDUAL QUALITY TREND
                        # ==========================================

                        quality_df["assessment_timestamp"] = pd.to_datetime(
                            quality_df["assessment_timestamp"],
                            errors="coerce",
                            utc=True
                        )

                        trend_df = (
                            quality_df
                            .dropna(subset=["assessment_timestamp"])
                            .sort_values("assessment_timestamp")
                            .copy()
                        )

                        if not trend_df.empty:

                            # Convert quality flags into readable text.
                            trend_df["quality_flags_display"] = [
                                ", ".join(flags)
                                if isinstance(flags, list) and flags
                                else "No quality flags"
                                for flags in trend_df["quality_flags"]
                            ]

                            # Display each assessment individually.
                            quality_chart = px.scatter(
                                trend_df,
                                x="assessment_timestamp",
                                y="quality_status",
                                color="quality_status",
                                category_orders={
                                    "quality_status": [
                                        "ACCEPTABLE",
                                        "INCOMPLETE",
                                        "REVIEW"
                                    ]
                                },
                                color_discrete_map={
                                    "ACCEPTABLE": "#16A34A",
                                    "INCOMPLETE": "#E5A000",
                                    "REVIEW": "#DC2626"
                                },
                                hover_data={
                                    "sensor_id": True,
                                    "quality_flags_display": True,
                                    "flag_count": True,
                                    "assessment_timestamp": False
                                },
                                title=(
                                    "Individual Sensor Quality "
                                    "Assessments Over Time"
                                ),
                                labels={
                                    "assessment_timestamp": (
                                        "Assessment Time (UTC)"
                                    ),
                                    "quality_status": "Quality Status",
                                    "sensor_id": "Sensor ID",
                                    "quality_flags_display": "Quality Flags",
                                    "flag_count": "Number of Flags"
                                }
                            )

                            quality_chart.update_traces(
                                marker=dict(size=13),
                                selector=dict(mode="markers")
                            )

                            quality_chart.update_yaxes(
                                categoryorder="array",
                                categoryarray=[
                                    "ACCEPTABLE",
                                    "INCOMPLETE",
                                    "REVIEW"
                                ],
                                title="Quality Classification"
                            )

                            quality_chart.update_xaxes(
                                title="Assessment Time (UTC)"
                            )

                            quality_chart.update_layout(
                                hovermode="closest"
                            )

                            st.plotly_chart(
                                quality_chart,
                                use_container_width=True
                            )

                        else:
                            st.info(
                                "No valid assessment timestamps are "
                                "available for historical plotting."
                            )


                    else:
                        st.info(
                            f"No quality assessments are currently "
                            f"available for {selected_field}."
                        )

                    st.markdown("---")
                    st.markdown("### 📈 Historical Sensor Trends")

                    # Convert timestamps for chronological plotting.
                    history_df["timestamp"] = pd.to_datetime(
                        history_df["timestamp"],
                        errors="coerce",
                        utc=True
                    )
                    # Soil moisture trend - corrected visualization.
                    soil_plot_df = history_df.copy()

                    soil_plot_df["soil_moisture"] = pd.to_numeric(
                        soil_plot_df["soil_moisture"],
                        errors="coerce"
                    )

                    soil_plot_df = soil_plot_df.dropna(
                        subset=["timestamp", "soil_moisture"]
                    ).sort_values("timestamp")

                    if not soil_plot_df.empty:
                        soil_min = soil_plot_df["soil_moisture"].min()
                        soil_max = soil_plot_df["soil_moisture"].max()

                        # Provide a visible vertical range.
                        padding = max((soil_max - soil_min) * 0.15, 1.0)

                        soil_chart = px.line(
                            soil_plot_df,
                            x="timestamp",
                            y="soil_moisture",
                            markers=True,
                            text="soil_moisture",
                            title="Soil Moisture Trend",
                            labels={
                                "timestamp": "Observation Time (UTC)",
                                "soil_moisture": "Soil Moisture (%)"
                            }
                        )

                        soil_chart.update_traces(
                            mode="lines+markers+text",
                            texttemplate="%{text:.1f}%",
                            textposition="top center",
                            marker=dict(size=10)
                        )

                        soil_chart.update_yaxes(
                            range=[
                                max(0, soil_min - padding),
                                soil_max + padding
                            ],
                            title="Soil Moisture (%)"
                        )

                        soil_chart.update_layout(
                            hovermode="x unified"
                        )

                        st.plotly_chart(
                            soil_chart,
                            use_container_width=True
                        )
                    else:
                        st.warning(
                            "No valid soil moisture observations available."
                        )


                    # Air temperature trend.
                    temperature_chart = px.line(
                        history_df,
                        x="timestamp",
                        y="air_temperature",
                        markers=True,
                        title="Air Temperature Trend",
                        labels={
                            "timestamp": "Observation Time",
                            "air_temperature": "Temperature (°C)"
                        }
                    )

                    st.plotly_chart(
                        temperature_chart,
                        use_container_width=True
                    )

                    # Relative humidity trend.
                    humidity_chart = px.line(
                        history_df,
                        x="timestamp",
                        y="relative_humidity",
                        markers=True,
                        title="Relative Humidity Trend",
                        labels={
                            "timestamp": "Observation Time",
                            "relative_humidity": "Relative Humidity (%)"
                        }
                    )

                    st.plotly_chart(
                        humidity_chart,
                        use_container_width=True
                    )

                    # Rainfall trend.
                    rainfall_chart = px.bar(
                        history_df,
                        x="timestamp",
                        y="rainfall",
                        title="Recorded Rainfall",
                        labels={
                            "timestamp": "Observation Time",
                            "rainfall": "Rainfall (mm)"
                        }
                    )

                    st.plotly_chart(
                        rainfall_chart,
                        use_container_width=True
                    )

                    st.markdown("---")
                    st.markdown("### 📋 Sensor Reading History")

                    display_df = history_df.sort_values(
                        "timestamp",
                        ascending=False
                    )

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.success(
                        f"Successfully retrieved {len(history_df)} "
                        f"sensor reading(s) for {selected_field}."
                    )

        except (sqlite3.Error, pd.errors.DatabaseError) as error:
            st.error(
                f"Unable to read the sensor database: {error}"
            )

        except Exception as error:
            st.error(
                f"Sensor monitoring error: {error}"
            )

        finally:
            if sensor_conn is not None:
                sensor_conn.close()
            # TAB 4: Agricultural Sensor Monitoring
    with p4_tab4:
        render_sensor_monitoring()

    # TAB 5: Offline Mobile App Sync
    with p4_tab5:
        st.subheader("📱 Offline Mobile App Sync")
        st.markdown(
            "Field users in rural areas can sync irrigation requests and receive instructions "
            "via an offline-capable mobile app when internet is unavailable; SMS remains the fallback channel."
        )

        st.info(
            "**Architecture:** Server ↔ Mobile App (offline queue) ↔ SMS gateway ↔ Furrow hardware. "
            "DSSIS-inspired workflow; Tokuma retains climate/population/GA optimization originality."
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write("**Simulated Mobile Request (offline queue → server)**")
            with st.form("mobile_sync"):
                mobile_farmer_id = st.text_input("Farmer ID (mobile)", value="MOB001")
                mobile_location = st.text_input("GPS / Field Location", value="Wonji")
                mobile_soil = st.number_input("Soil Moisture from App (mm)", value=28.0)
                mobile_phone = st.text_input("Phone for SMS fallback", value="+251955006516")
                if st.form_submit_button("Queue Mobile Sync Request"):
                    if 'field_comm_manager' not in st.session_state:
                        st.session_state.field_comm_manager = FieldCommunicationManager()
                    req = st.session_state.field_comm_manager.add_field_request(
                        mobile_farmer_id, mobile_location, "Mid-Season", mobile_soil
                    )
                    _execute_db(
                        """INSERT INTO field_requests
                        (timestamp, farmer_id, location, crop_stage, soil_moisture, status)
                        VALUES (?,?,?,?,?,?)""",
                        (req['timestamp'], mobile_farmer_id, mobile_location, "Mid-Season", mobile_soil, 'mobile_queued')
                    )
                    db_conn.commit()
                    st.session_state.last_mobile_phone = mobile_phone
                    st.success("Mobile request queued for server sync (simulated).")

        with col_m2:
            st.write("**Server Response to Mobile / SMS**")
            if 'current_schedule' in st.session_state and st.session_state.get('field_comm_manager'):
                schedule = st.session_state.current_schedule
                payload = {
                    "endpoint": "POST /api/v1/field/sync",
                    "irrigation_amount_mm": round(schedule['net_irrigation_mm'], 1),
                    "timing": schedule['timing'],
                    "duration_minutes": (
                        round(schedule['duration_minutes'], 1)
                        if schedule['duration_minutes'] is not None
                        else None
                    ),
                    "flow_rate_lps": schedule['flow_rate'],
                    "drivers": {
                        "country": country,
                        "crop": crop_type,
                        "schedule_method": st.session_state.get('schedule_method', 'ET-based'),
                    },
                }
                st.json(payload)
                if schedule['duration_minutes'] is None:
                    st.warning("Enter the irrigated field area in Phase 3 before sending a timed instruction.")
                if st.button(
                    "Push Instruction to Mobile + SMS",
                    disabled=schedule['duration_minutes'] is None,
                ):
                    if 'sms_module' not in st.session_state:
                        st.session_state.sms_module = SMSCommunicationModule()
                    phone = st.session_state.get('last_mobile_phone', '+251955006516')
                    instruction = {
                        'irrigation_amount': schedule['net_irrigation_mm'],
                        'duration_minutes': schedule['duration_minutes'],
                        'flow_rate': schedule['flow_rate'],
                        'irrigation_timing': schedule['timing'],
                        'request_id': datetime.now().isoformat(),
                    }
                    message = st.session_state.sms_module.format_irrigation_instruction_sms(instruction)
                    sms = st.session_state.sms_module.queue_sms(phone, message)
                    st.session_state.sms_module.send_sms(sms)
                    st.success(f"Instruction pushed to mobile queue and SMS sent to {phone}")
            else:
                st.warning("Complete Phase 3 (generate schedule) and queue a mobile request first.")

        st.caption(
            "Production: deploy REST API (e.g. FastAPI on port 8000) and Flutter/React Native offline app. "
            "Current build simulates sync for research and field-testing workflows."
        )

if is_admin:
    assert phase5 is not None
    # ==================== PHASE 5: REPORTING & EXPORT ====================
    with phase5:
        st.subheader("📊 Phase 5: Reporting & Export")
        st.markdown("Generate reports, export data, and access API connectors")

        p5_tab1, p5_tab2, p5_tab3, p5_tab4 = st.tabs([
            "Reports",
            "API Connectors",
            "Export",
            "Experimental Design",
        ])

        with p5_tab1:
            st.subheader("📄 Report Generation")

            # Generate report content (only if simulation has been run)
            if st.session_state.get('simulation_run', False):
                results = st.session_state.simulation_results
                wp = results['wp']
                yld = results['yld']
                demand = results['demand']
                temp = results['temp']
                country = results['country']
                target_year = results['target_year']
                crop_type = results['crop_type']
                use_file_coupling = results['use_file_coupling']

                report_content = f"""
                TOKUMA 3-in-1 INTEGRATED RESEARCH REPORT
                ----------------------------------------
                Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}
                Researcher: {st.session_state.user}
                Institution: {st.session_state.inst}
                Study Area: {country}
                Target Year: {target_year}
                Crop: {crop_type}
                Coupling: {"file-based subprocess" if use_file_coupling else "integrated proxy"}

                KEY METRICS:
                - Water Productivity: {wp:.3f} kg/m3
                - Projected Yield: {yld:.2f} t/ha
                - Water Demand: {demand:.2f} BCM
                - Temperature Change: +{temp:.2f} C

                IRRIGATION SCHEDULING:
                - Schedule Generated: {'Yes' if 'current_schedule' in st.session_state else 'No'}
                """

                if 'current_schedule' in st.session_state:
                    schedule = st.session_state.current_schedule
                    duration = schedule['duration_minutes']
                    duration_text = (
                        f"{duration:.1f} min" if duration is not None else "Not calculated (field area required)"
                    )
                    report_content += f"""
                - Irrigation Amount: {schedule['net_irrigation_mm']:.1f} mm
                - Timing: {schedule['timing']}
                - Duration: {duration_text}
                - Soil-Water Deficit Proxy: {schedule['stress_level']:.2f}
                    """

                st.download_button(
                    label="📥 Download Research Report",
                    data=report_content,
                    file_name=f"Report_{country}_{target_year}.txt",
                    mime="text/plain",
                )

                # Advanced report generation if available
                if REPORT_AVAILABLE:
                    if st.button("Generate Advanced Report"):
                        assert get_report_generator is not None
                        report_results = get_report_generator().render_report_generator_interface()
                        if report_results:
                            st.session_state.report_results = report_results
                else:
                    st.info("Advanced Report Generator module not available")
            else:
                st.info("Please run simulation first to generate reports")

        with p5_tab2:
            st.subheader("📡 API Connectors")
            if API_AVAILABLE:
                assert get_api_manager is not None
                api_results = get_api_manager().render_api_interface()
                if api_results:
                    st.session_state.api_results = api_results
            else:
                st.error("API Connectors module not available. Please install required packages: requests")

        with p5_tab3:
            st.subheader("📂 Export Research Data")
            if st.button("Archive to 13-Column Log"):
                st.success("Simulation Archived.")

            logs = _read_db_query("SELECT * FROM research_logs")
            st.write("**Research Log History**")
            st.dataframe(logs, use_container_width=True)
            if not logs.empty:
                st.download_button(
                    label="📊 Download Research Logs (CSV)",
                    data=logs.to_csv(index=False),
                    file_name="Research_Logs.csv",
                    mime="text/csv",
                )

            field_logs = _read_db_query("SELECT * FROM field_data")
            st.write("**Field Observation Records**")
            st.dataframe(field_logs, use_container_width=True)
            if not field_logs.empty:
                st.download_button(
                    label="📝 Download Field Records (CSV)",
                    data=field_logs.to_csv(index=False),
                    file_name="Field_Observations.csv",
                    mime="text/csv",
                )

            # NEW: Export irrigation schedules
            schedule_logs = _read_db_query("SELECT * FROM irrigation_schedules")
            st.write("**Irrigation Schedule History**")
            st.dataframe(schedule_logs, use_container_width=True)
            if not schedule_logs.empty:
                st.download_button(
                    label="💧 Download Irrigation Schedules (CSV)",
                    data=schedule_logs.to_csv(index=False),
                    file_name="Irrigation_Schedules.csv",
                    mime="text/csv",
                )

            # NEW: Export field requests
            request_logs = _read_db_query("SELECT * FROM field_requests")
            st.write("**Field Request History**")
            st.dataframe(request_logs, use_container_width=True)
            if not request_logs.empty:
                st.download_button(
                    label="📡 Download Field Requests (CSV)",
                    data=request_logs.to_csv(index=False),
                    file_name="Field_Requests.csv",
                    mime="text/csv",
                )

            # NEW: Export hardware control logs
            hardware_logs = _read_db_query("SELECT * FROM hardware_control_log")
            st.write("**Hardware Control Log**")
            st.dataframe(hardware_logs, use_container_width=True)
            if not hardware_logs.empty:
                st.download_button(
                    label="⚙️ Download Hardware Logs (CSV)",
                    data=hardware_logs.to_csv(index=False),
                    file_name="Hardware_Control_Log.csv",
                    mime="text/csv",
                )

            # Monte Carlo download only available if simulation has been run
            if st.session_state.get('simulation_run', False):
                mc_df = st.session_state.simulation_results['mc_df']
                st.download_button(
                    "Download Monte Carlo table (CSV)",
                    data=mc_df.to_csv(index=False),
                    file_name="monte_carlo_nexus.csv",
                    mime="text/csv",
                )
            else:
                st.info("Run simulation first to download Monte Carlo data")

        # ==========================================================
        # STEP 5.3.12.3.1 - EXPERIMENTAL DESIGN INSPECTION
        # Read-only reporting: no registration or record modification.
        # ==========================================================
        with p5_tab4:
            st.subheader("Registered Experimental Design")

            st.caption(
                "Read-only inspection of the registered HARC and MARC "
                "experimental design. This interface does not register, "
                "edit, or delete experimental records."
            )

            st.warning(
                "Design records represent the proposed experimental "
                "framework. They do not establish surveyed field boundaries, "
                "physical plot randomization, or completed field operations."
            )

            try:
                experimental_inventory = get_experimental_design_inventory(
                    db_conn,
                    st.session_state.get("db_type", "sqlite"),
                )

                inventory_counts = experimental_inventory["counts"]

                # Key experimental design metrics.
                metric_columns = st.columns(4)

                metric_columns[0].metric(
                    "Research Sites",
                    inventory_counts["experimental_sites"],
                )
                metric_columns[1].metric(
                    "Experimental Blocks",
                    inventory_counts["experimental_blocks"],
                )
                metric_columns[2].metric(
                    "Main Plots",
                    inventory_counts["experimental_main_plots"],
                )
                metric_columns[3].metric(
                    "Crop Subplots",
                    inventory_counts["experimental_subplots"],
                )

                st.divider()

                # Research site inventory.
                st.markdown("#### 1. Research Sites")

                sites_df = pd.DataFrame(
                    experimental_inventory["sites"]
                )

                if not sites_df.empty:
                    site_columns = [
                        column for column in [
                            "site_code",
                            "site_name",
                            "region",
                            "latitude",
                            "longitude",
                            "altitude_m",
                        ]
                        if column in sites_df.columns
                    ]

                    st.dataframe(
                        sites_df[site_columns],
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.caption(
                        "Site coordinates are approximate proposal-based "
                        "locations, not surveyed experimental plot coordinates."
                    )
                else:
                    st.info("No research sites are registered.")

                # Irrigation treatment distribution.
                st.markdown("#### 2. Irrigation Treatment Distribution")

                main_plots_df = pd.DataFrame(
                    experimental_inventory["main_plots"]
                )

                if (
                    not main_plots_df.empty
                    and "irrigation_treatment" in main_plots_df.columns
                ):
                    treatment_summary = (
                        main_plots_df["irrigation_treatment"]
                        .value_counts()
                        .rename_axis("Irrigation Treatment")
                        .reset_index(name="Main Plot Count")
                    )

                    st.dataframe(
                        treatment_summary,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No irrigation main plots are registered.")

                # Crop subplot distribution.
                st.markdown("#### 3. Crop Subplot Distribution")

                subplots_df = pd.DataFrame(
                    experimental_inventory["subplots"]
                )

                if (
                    not subplots_df.empty
                    and "crop_code" in subplots_df.columns
                ):
                    crop_summary = (
                        subplots_df["crop_code"]
                        .value_counts()
                        .rename_axis("Crop")
                        .reset_index(name="Subplot Count")
                    )

                    st.dataframe(
                        crop_summary,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No crop subplots are registered.")

                # Detailed registered subplot inventory.
                st.markdown("#### 4. Detailed Subplot Inventory")

                if not subplots_df.empty:
                    st.dataframe(
                        subplots_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        label="Download Experimental Subplot Inventory (CSV)",
                        data=subplots_df.to_csv(index=False).encode("utf-8"),
                        file_name="TOKUMA_Experimental_Subplot_Inventory.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("There are no subplot records to export.")

                # Structural validation.
                st.markdown("#### 5. Experimental Design Validation")

                validation_result = experimental_inventory["validation"]

                if validation_result["valid"]:
                    st.success(
                        "Experimental design structural validation passed."
                    )
                else:
                    st.error(
                        "Experimental design structural validation failed."
                    )

                    for validation_error in validation_result.get(
                        "errors", []
                    ):
                        st.write(f"- {validation_error}")

                # Sensor installation status.
                st.markdown("#### 6. Sensor Deployment Status")

                deployment_count = inventory_counts.get(
                    "sensor_deployments", 0
                )

                st.metric(
                    "Registered Sensor Deployments",
                    deployment_count,
                )

                if deployment_count == 0:
                    st.info(
                        "No physical sensor deployments are currently "
                        "registered. Sensor installation should only be "
                        "recorded after actual field deployment is verified."
                    )

            except Exception as error:
                st.error(
                    "Unable to retrieve the experimental design inventory."
                )
                st.exception(error)

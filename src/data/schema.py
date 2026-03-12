"""
HAI Dataset Schema Definitions

HAI = Hardware-In-the-Loop Augmented ICS Security Dataset
Industrial boiler/steam system simulation with cyberattack scenarios.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ── Column category definitions for HAI-23.05 ────────────────────────────────

TIMESTAMP_COL = "timestamp"

LABEL_COL = "Attack"  # Binary: 0=normal, 1=attack

# Process P1: Water treatment / boiler feed
P1_PRESSURE_SENSORS = ["P1_PIT01", "P1_PIT02"]
P1_FLOW_SENSORS = ["P1_FT01", "P1_FT02", "P1_FT03"]
P1_TEMP_SENSORS = ["P1_TIT01", "P1_TIT02", "P1_TIT03"]
P1_LEVEL_SENSORS = ["P1_LIT01"]
P1_CONTROL_VALVES = ["P1_FCV01D", "P1_FCV01Z", "P1_FCV02D", "P1_FCV02Z",
                      "P1_FCV03D", "P1_FCV03Z"]
P1_PUMP_COLS = ["P1_PP01AD", "P1_PP01AR", "P1_PP02D", "P1_PP02R", "P1_PP04"]
P1_VALVE_COLS = ["P1_SOL01D", "P1_SOL03D"]

# Process P2: Secondary system / monitoring
P2_SENSORS = ["P2_SIT01"]
P2_STATUS_COLS = ["P2_24Vdc", "P2_OnOff", "P2_AutoGO", "P2_Emerg",
                   "P2_VTR", "P2_VIBTR"]

# Process P3: Feedwater / Level control
P3_PRESSURE_SENSORS = ["P3_PIT01"]
P3_FLOW_SENSORS = ["P3_FIT01"]
P3_LEVEL_SENSORS = ["P3_LIT01"]
P3_LEVEL_CONTROL = ["P3_LCV01D", "P3_LCV01Z", "P3_LL01", "P3_LH01"]

# Process P4: Steam generation
P4_SENSORS = ["P4_ST_PT01", "P4_ST_TT01", "P4_ST_FT01", "P4_LD", "P4_ST_PS01"]

# Setpoint / demand columns (ending in Z or D)
SETPOINT_COLS_SUFFIXES = ["Z", "SP"]
DEMAND_COLS_SUFFIXES = ["D"]

# All sensor categories combined
ALL_SENSOR_GROUPS = {
    "P1_Pressure": P1_PRESSURE_SENSORS,
    "P1_Flow": P1_FLOW_SENSORS,
    "P1_Temperature": P1_TEMP_SENSORS,
    "P1_Level": P1_LEVEL_SENSORS,
    "P1_ControlValves": P1_CONTROL_VALVES,
    "P1_Pumps": P1_PUMP_COLS,
    "P1_Valves": P1_VALVE_COLS,
    "P2_Sensors": P2_SENSORS,
    "P2_Status": P2_STATUS_COLS,
    "P3_Pressure": P3_PRESSURE_SENSORS,
    "P3_Flow": P3_FLOW_SENSORS,
    "P3_Level": P3_LEVEL_SENSORS,
    "P3_LevelControl": P3_LEVEL_CONTROL,
    "P4_Steam": P4_SENSORS,
}

# Known binary columns (0/1 states)
BINARY_COLS = [
    "P1_SOL01D", "P1_SOL03D", "P1_PP01AD", "P1_PP01AR", "P1_PP02D", "P1_PP02R",
    "P2_OnOff", "P2_AutoGO", "P2_Emerg", "P3_LL01", "P3_LH01", "P2_VTR", "P2_VIBTR"
]

# Critical sensors most associated with attack detection (from domain knowledge)
CRITICAL_SENSORS = [
    "P1_FT01", "P1_FT02", "P1_PIT01", "P1_PIT02", "P1_LIT01",
    "P2_SIT01", "P3_FIT01", "P3_LIT01", "P4_ST_PT01", "P4_ST_TT01"
]


@dataclass
class HAISchema:
    """Schema for HAI dataset version."""
    version: str
    n_sensor_cols: int
    has_embedded_labels: bool
    label_cols: List[str]
    timestamp_col: str
    delimiter: str
    encoding: str = "utf-8"
    description: str = ""


HAI_VERSIONS = {
    "hai-20.07": HAISchema(
        version="hai-20.07",
        n_sensor_cols=62,
        has_embedded_labels=True,
        label_cols=["attack", "attack_P1", "attack_P2", "attack_P3"],
        timestamp_col="timestamp",
        delimiter=";",
        description="Original 2019 dataset, semicolon-delimited, multi-process attack labels"
    ),
    "hai-21.03": HAISchema(
        version="hai-21.03",
        n_sensor_cols=82,
        has_embedded_labels=True,
        label_cols=["attack", "attack_P1", "attack_P2", "attack_P3"],
        timestamp_col="timestamp",
        delimiter=",",
        description="2020-2021 dataset, expanded columns, 5 test scenarios"
    ),
    "hai-22.04": HAISchema(
        version="hai-22.04",
        n_sensor_cols=86,
        has_embedded_labels=True,
        label_cols=["Attack"],
        timestamp_col="timestamp",
        delimiter=",",
        description="2021 dataset, detailed attack summaries, 6 train + 4 test files"
    ),
    "hai-23.05": HAISchema(
        version="hai-23.05",
        n_sensor_cols=86,
        has_embedded_labels=False,
        label_cols=["Attack"],
        timestamp_col="timestamp",
        delimiter=",",
        description="2022 dataset, separate label files, 4 train + 2 test files"
    ),
    "haiend-23.05": HAISchema(
        version="haiend-23.05",
        n_sensor_cols=224,
        has_embedded_labels=False,
        label_cols=["Attack"],
        timestamp_col="timestamp",
        delimiter=",",
        description="2022 extended dataset with DCS node outputs, 226 total columns"
    ),
}


DATA_DICTIONARY = {
    "timestamp": "ISO 8601 datetime timestamp at 1-second intervals",
    "P1_PIT01": "Process 1 - Pressure Indicator Transmitter 01 (bar)",
    "P1_PIT02": "Process 1 - Pressure Indicator Transmitter 02 (bar)",
    "P1_FT01": "Process 1 - Flow Transmitter 01 (LPM)",
    "P1_FT02": "Process 1 - Flow Transmitter 02 (LPM)",
    "P1_FT03": "Process 1 - Flow Transmitter 03 (LPM)",
    "P1_TIT01": "Process 1 - Temperature Indicator Transmitter 01 (°C)",
    "P1_TIT02": "Process 1 - Temperature Indicator Transmitter 02 (°C)",
    "P1_TIT03": "Process 1 - Temperature Indicator Transmitter 03 (°C)",
    "P1_LIT01": "Process 1 - Level Indicator Transmitter 01 (mm)",
    "P1_FCV01D": "Process 1 - Flow Control Valve 01 Demand (% open)",
    "P1_FCV01Z": "Process 1 - Flow Control Valve 01 Setpoint (%)",
    "P1_FCV02D": "Process 1 - Flow Control Valve 02 Demand (% open)",
    "P1_FCV02Z": "Process 1 - Flow Control Valve 02 Setpoint (%)",
    "P1_FCV03D": "Process 1 - Flow Control Valve 03 Demand (% open)",
    "P1_FCV03Z": "Process 1 - Flow Control Valve 03 Setpoint (%)",
    "P1_PP01AD": "Process 1 - Pump 01A Demand (binary: 0/1)",
    "P1_PP01AR": "Process 1 - Pump 01A Running status (binary)",
    "P1_PP02D": "Process 1 - Pump 02 Demand (binary)",
    "P1_PP02R": "Process 1 - Pump 02 Running status (binary)",
    "P1_PP04": "Process 1 - Pump 04 Speed (RPM)",
    "P1_SOL01D": "Process 1 - Solenoid Valve 01 Demand (binary)",
    "P1_SOL03D": "Process 1 - Solenoid Valve 03 Demand (binary)",
    "P2_SIT01": "Process 2 - Speed/Signal Indicator Transmitter 01",
    "P2_24Vdc": "Process 2 - 24V DC power status",
    "P2_OnOff": "Process 2 - On/Off control state",
    "P2_AutoGO": "Process 2 - Automatic mode active",
    "P2_Emerg": "Process 2 - Emergency stop status",
    "P2_VTR": "Process 2 - Vibration trip relay",
    "P2_VIBTR": "Process 2 - Vibration transmitter",
    "P3_PIT01": "Process 3 - Pressure Indicator Transmitter 01 (bar)",
    "P3_FIT01": "Process 3 - Flow Indicator Transmitter 01 (LPM)",
    "P3_LIT01": "Process 3 - Level Indicator Transmitter 01 (mm)",
    "P3_LCV01D": "Process 3 - Level Control Valve 01 Demand (%)",
    "P3_LCV01Z": "Process 3 - Level Control Valve 01 Setpoint (%)",
    "P3_LL01": "Process 3 - Level Low 01 alarm (binary)",
    "P3_LH01": "Process 3 - Level High 01 alarm (binary)",
    "P4_ST_PT01": "Process 4 - Steam Pressure Transmitter 01 (bar)",
    "P4_ST_TT01": "Process 4 - Steam Temperature Transmitter 01 (°C)",
    "P4_ST_FT01": "Process 4 - Steam Flow Transmitter 01",
    "P4_LD": "Process 4 - Load demand",
    "P4_ST_PS01": "Process 4 - Steam Pressure Switch 01",
    "Attack": "Binary attack label: 0=Normal operation, 1=Cyberattack active",
}

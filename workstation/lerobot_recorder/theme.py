"""Modern flat dark theme + status colors for the recorder/replay GUIs."""

from __future__ import annotations

# semantic colors (GitHub-dark-ish)
BG = "#0d1117"
PANEL = "#161b22"
TEXT = "#c9d1d9"
MUTED = "#8b949e"
ACCENT = "#58a6ff"
OK = "#2ea043"
WARN = "#d29922"
BAD = "#f85149"
IDLE = "#30363d"

# status banner color per state
STATE_COLORS = {
    "IDLE": IDLE,
    "ARMED": ACCENT,
    "ENGAGED": OK,
    "REC": OK,
    "HOMING": WARN,
    "REVIEW": WARN,
    "ERROR": BAD,
}

QSS = f"""
* {{ font-family: -apple-system, "Segoe UI", "Noto Sans", sans-serif; font-size: 26px; }}
QWidget {{ background: {BG}; color: {TEXT}; }}
QGroupBox {{ background: {PANEL}; border: 1px solid #30363d; border-radius: 10px; margin-top: 20px; padding: 16px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {MUTED}; }}
QLabel {{ background: transparent; }}
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
    background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 12px 14px; color: {TEXT}; font-size: 26px;
}}
QComboBox[pickerCombo="true"] {{ border-color: {ACCENT}; padding-right: 92px; }}
QComboBox[pickerCombo="true"] QLineEdit {{ background: transparent; border: 0; padding: 12px 14px; color: {TEXT}; }}
QComboBox::drop-down {{
    background: #21262d; border: 0; border-left: 1px solid #30363d;
    border-top-right-radius: 8px; border-bottom-right-radius: 8px; width: 56px;
}}
QComboBox::down-arrow {{ width: 18px; height: 18px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid #30363d; selection-background-color: {ACCENT}; }}
QToolButton#comboDropButton {{
    background: {ACCENT}; border: 0; border-left: 1px solid #30363d;
    border-top-right-radius: 8px; border-bottom-right-radius: 8px;
    color: white; font-size: 28px; font-weight: 700; padding: 0;
}}
QToolButton#comboDropButton:hover {{ background: #6cb1ff; }}
QToolButton#comboDropButton:pressed {{ background: #388bfd; }}
QToolButton#comboDropButton:disabled {{ background: #21262d; color: #484f58; }}
QPushButton {{
    background: #21262d; border: 1px solid #30363d; border-radius: 8px; padding: 16px 26px; color: {TEXT}; font-size: 26px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: #30363d; }}
QPushButton:disabled {{ color: #484f58; }}
QPushButton:checked {{ background: {OK}; border-color: {OK}; color: white; }}
QPushButton#start {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-size: 40px; font-weight: 700; padding: 22px 40px; }}
QPushButton#start:hover {{ background: #6cb1ff; }}
QPushButton#save {{ background: #8ecbff; border-color: #8ecbff; color: #06111f; font-weight: 700; }}
QPushButton#save:hover {{ background: #a6d8ff; border-color: #a6d8ff; }}
QPushButton#save:pressed {{ background: #74bdf2; border-color: #74bdf2; }}
QPushButton#save:disabled {{ background: #21262d; border-color: #30363d; color: #484f58; font-weight: 400; }}
QPushButton#estop {{ font-size: 26px; font-weight: 700; }}
QCheckBox {{ spacing: 10px; font-size: 26px; }}
"""


def banner_style(color: str) -> str:
    return f"background: {color}; color: white; border-radius: 10px; padding: 16px; font-size: 44px; font-weight: 600;"


def dot(ok: bool) -> str:
    """Inline HTML colored dot for the health strip."""
    return f'<span style="color:{OK if ok else BAD};">●</span>'

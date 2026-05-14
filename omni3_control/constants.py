"""
omni3_control/constants.py
Paylasilan sabitler — kinematik ve encoder.
"""

import math

# ── Encoder ──────────────────────────────────────────────────────────────────
COUNTS_PER_REV = 750
CPR2RAD  = 2.0 * math.pi / COUNTS_PER_REV
RAD2QPPS = COUNTS_PER_REV / (2.0 * math.pi)

# ── Kinematik (varsayilan; ROS parametreleri ile ezilebilir) ─────────────────
WHEEL_RADIUS = 0.05   # [m]
ROBOT_RADIUS = 0.27   # [m]
WHEEL_BETAS  = (-60.0, 60.0, 180.0)  # [deg]

# ── Encoder watchdog ─────────────────────────────────────────────────────────
ENC_STALE_SEC = 0.2   # bu kadar saniye taze veri yoksa motorlari durdur

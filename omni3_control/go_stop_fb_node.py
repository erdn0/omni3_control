#!/usr/bin/env python3
"""
go_stop_fb_node.py — GO-STOP quintic trayektori takibi (feedforward + geribildirim).

go_stop_node ile ayni trayektori planlamasini kullanir; fark:
  - Her adimda konum/yon hatasi P-kontrolcusu ile duzeltilir (FF + P geribildirim).
  - Trayektori bittikten sonra hedef konumda P-kontrol ile tutucu moda gecer.

Baslangic (0, 0, 0) noktasindan (X_REF, Y_REF, PHI_REF) hedefine
T_TOTAL saniyede rest-to-rest quintic polinom ile gider.

Wheel haritasi (move_1m_node / go_stop_node ile ayni):
  Wheel1  beta=-60   0x80 M2  (sag on)
  Wheel2  beta=+60   0x80 M1  (sol on)
  Wheel3  beta=180   0x81 M2  (arka)
"""

import math
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

from omni3_control.roboclaw import Roboclaw, RoboClawError
from omni3_control.kinematics import OmniKinematics, OmniParams
from omni3_control.quintic import QuinticTrajectory
from omni3_control.constants import (
    WHEEL_RADIUS, ROBOT_RADIUS, WHEEL_BETAS,
    CPR2RAD, RAD2QPPS, ENC_STALE_SEC,
)

# ── DONANIM ──────────────────────────────────────────────────────────────────
PORT_A    = '/dev/roboclaw_front'
PORT_B    = '/dev/roboclaw_rear'
BAUDRATE  = 38400
ADDR_A    = 0x80
ADDR_B    = 0x81

DIR_W1    = -1
DIR_W2    = -1
DIR_W3    = -1

PID_P     = 3
PID_I     = 0
PID_D     = 0
QPPS_MAX  = 3000

# ── GO-STOP HEDEF ────────────────────────────────────────────────────────────
X_REF    = 1.0    # [m]
Y_REF    = 0.0    # [m]
PHI_REF  = 0.0    # [rad]
T_TOTAL  = 5.0    # [s]   trayektori suresi
DT       = 0.05   # [s]   kontrol periyodu

# ── SANAL ENGEL (daire) ──────────────────────────────────────────────────────
OBS_X      = 0.5
OBS_Y      = 0.0
OBS_R      = 0.10
OBS_MARGIN = 0.10
OBS_ENABLE = False

# ── GERIBILDIRIM KAZANIMI ─────────────────────────────────────────────────────
KP_XY       = 1.0   # [1/s]  trayektori sirasinda konum hatasi kazanimi (xy)
KP_PHI      = 2.0   # [1/s]  trayektori sirasinda yon hatasi kazanimi
KP_HOLD     = 1.5   # [1/s]  tutucu modda konum hatasi kazanimi
KP_HOLD_PHI = 2.0   # [1/s]  tutucu modda yon hatasi kazanimi

# Tutucu mod cozulmus kabul edilecek esik degerler
HOLD_XY_TOL  = 0.005  # [m]
HOLD_PHI_TOL = 0.01   # [rad]


class GoStopFbNode(Node):

    def __init__(self):
        super().__init__('go_stop_fb_node')

        self.kin = OmniKinematics(OmniParams(
            wheel_radius=WHEEL_RADIUS,
            robot_radius=ROBOT_RADIUS,
            beta=WHEEL_BETAS,
        ))

        try:
            self.rc_a = Roboclaw(PORT_A, BAUDRATE, timeout=0.1)
            self.get_logger().info(f'RoboClaw A acildi: {PORT_A}')
        except Exception as e:
            self.get_logger().fatal(f'RoboClaw A acilamadi ({PORT_A}): {e}')
            raise RuntimeError('serial error')

        try:
            self.rc_b = Roboclaw(PORT_B, BAUDRATE, timeout=0.1)
            self.get_logger().info(f'RoboClaw B acildi: {PORT_B}')
        except Exception as e:
            self.get_logger().fatal(f'RoboClaw B acilamadi ({PORT_B}): {e}')
            raise RuntimeError('serial error')

        self.rc_a.SetM1VelocityPID(ADDR_A, PID_P, PID_I, PID_D, QPPS_MAX)
        self.rc_a.SetM2VelocityPID(ADDR_A, PID_P, PID_I, PID_D, QPPS_MAX)
        self.rc_b.SetM2VelocityPID(ADDR_B, PID_P, PID_I, PID_D, QPPS_MAX)
        self.rc_a.ResetEncoders(ADDR_A)
        self.rc_b.ResetEncoders(ADDR_B)
        time.sleep(0.1)

        self._enc_lock    = threading.Lock()
        self._enc_counts  = [0, 0, 0]
        self._enc_last_ts = 0.0
        self._enc_ready   = False

        self._running    = True
        self._enc_thread = threading.Thread(target=self._enc_reader, daemon=True)
        self._enc_thread.start()

        while not self._enc_ready:
            time.sleep(0.01)
        with self._enc_lock:
            self._prev_enc = list(self._enc_counts)

        self.pose    = np.zeros(3)
        self._held   = False   # tutucu modda cozuldu mu

        # Quintic plan
        S = np.array([0.0, 0.0])
        G = np.array([X_REF, Y_REF])
        V = None
        if OBS_ENABLE:
            C      = np.array([OBS_X, OBS_Y])
            r_safe = OBS_R + OBS_MARGIN
            V = self._plan_via_point(S, G, C, r_safe)

        if V is None:
            self._segments = [(
                QuinticTrajectory(0.0, X_REF, T_TOTAL),
                QuinticTrajectory(0.0, Y_REF, T_TOTAL),
                0.0, T_TOTAL,
            )]
            plan_info = 'tek segment (engel yok / hizada degil)'
        else:
            d1 = float(np.linalg.norm(V - S))
            d2 = float(np.linalg.norm(G - V))
            T1 = T_TOTAL * d1 / (d1 + d2)
            T2 = T_TOTAL - T1
            self._segments = [
                (QuinticTrajectory(0.0,    V[0],  T1),
                 QuinticTrajectory(0.0,    V[1],  T1),
                 0.0, T1),
                (QuinticTrajectory(V[0],   X_REF, T2),
                 QuinticTrajectory(V[1],   Y_REF, T2),
                 T1, T_TOTAL),
            ]
            plan_info = (f'iki segment | via=({V[0]:+.3f}, {V[1]:+.3f})  '
                         f'd1={d1:.3f} d2={d2:.3f}  T1={T1:.2f} T2={T2:.2f}')

        self.tphi = QuinticTrajectory(0.0, PHI_REF, T_TOTAL)
        self._t0  = time.monotonic()

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.create_timer(DT, self._control_loop)

        self.get_logger().info('GoStopFbNode basladi (FF + P geribildirim).')
        self.get_logger().info(
            f'Hedef: x={X_REF} m, y={Y_REF} m, phi={PHI_REF} rad  |  T={T_TOTAL} s'
        )
        self.get_logger().info(
            f'Kazanim: KP_XY={KP_XY}  KP_PHI={KP_PHI}  '
            f'KP_HOLD={KP_HOLD}  KP_HOLD_PHI={KP_HOLD_PHI}'
        )
        if OBS_ENABLE:
            self.get_logger().info(
                f'Engel: ({OBS_X}, {OBS_Y}) r={OBS_R}  margin={OBS_MARGIN}  → {plan_info}'
            )

    # ── Engel etrafi yol planlama ────────────────────────────────────────────
    @staticmethod
    def _plan_via_point(S: np.ndarray, G: np.ndarray, C: np.ndarray, r_safe: float):
        SG = G - S
        L = float(np.linalg.norm(SG))
        if L < 1e-9:
            return None
        t_proj = float(np.dot(C - S, SG) / (L * L))
        if t_proj <= 0.0 or t_proj >= 1.0:
            return None
        P = S + t_proj * SG
        d = float(np.linalg.norm(C - P))
        if d >= r_safe:
            return None
        SG_hat = SG / L
        n_perp = np.array([-SG_hat[1], SG_hat[0]])
        V_a = C + r_safe * n_perp
        V_b = C - r_safe * n_perp
        len_a = float(np.linalg.norm(V_a - S) + np.linalg.norm(G - V_a))
        len_b = float(np.linalg.norm(V_b - S) + np.linalg.norm(G - V_b))
        return V_a if len_a <= len_b else V_b

    # ── Encoder thread ───────────────────────────────────────────────────────
    @staticmethod
    def _delta_i32(cur: int, prev: int) -> int:
        return ((cur - prev + 0x80000000) & 0xFFFFFFFF) - 0x80000000

    def _enc_reader(self):
        while self._running:
            try:
                w1 = self.rc_a.ReadEncM2(ADDR_A)
                w2 = self.rc_a.ReadEncM1(ADDR_A)
                w3 = self.rc_b.ReadEncM2(ADDR_B)
                now = time.monotonic()
                with self._enc_lock:
                    self._enc_counts  = [w1, w2, w3]
                    self._enc_last_ts = now
                    self._enc_ready   = True
            except Exception as e:
                self.get_logger().warn(f'Encoder okuma hatasi: {e}', throttle_duration_sec=2.0)
            time.sleep(0.02)

    # ── 20 Hz kontrol ────────────────────────────────────────────────────────
    def _control_loop(self):
        if self._held:
            return

        with self._enc_lock:
            cur     = list(self._enc_counts)
            last_ts = self._enc_last_ts

        if time.monotonic() - last_ts > ENC_STALE_SEC:
            self.get_logger().error(
                f'Encoder verisi bayat ({time.monotonic() - last_ts:.2f}s) — motorlar durduruluyor',
                throttle_duration_sec=1.0,
            )
            self._stop()
            return

        # Odometri
        dc = [self._delta_i32(cur[i], self._prev_enc[i]) for i in range(3)]
        self._prev_enc = cur

        dphi = np.array([
            dc[0] * DIR_W1 * CPR2RAD,
            dc[1] * DIR_W2 * CPR2RAD,
            dc[2] * DIR_W3 * CPR2RAD,
        ])
        theta     = self.pose[2]
        disp_body = self.kin.J_inv @ (dphi * self.kin.p.wheel_radius)
        c, s      = math.cos(theta), math.sin(theta)
        self.pose += np.array([
            c * disp_body[0] - s * disp_body[1],
            s * disp_body[0] + c * disp_body[1],
            disp_body[2],
        ])
        self._publish_odom()

        t = time.monotonic() - self._t0

        # ── Tutucu mod (trayektori sonrasi) ──────────────────────────────────
        if t >= T_TOTAL:
            ex   = X_REF   - self.pose[0]
            ey   = Y_REF   - self.pose[1]
            ephi = PHI_REF - self.pose[2]

            dist = math.hypot(ex, ey)
            if dist < HOLD_XY_TOL and abs(ephi) < HOLD_PHI_TOL:
                self.get_logger().info('Hedef konuma ulasildi, motorlar durduruluyor.')
                self._stop()
                self._held = True
                return

            vx_cmd = KP_HOLD     * ex
            vy_cmd = KP_HOLD     * ey
            w_cmd  = KP_HOLD_PHI * ephi

            phi_dot = self.kin.forward_world(
                np.array([vx_cmd, vy_cmd, w_cmd]), self.pose[2]
            )
            self._send(phi_dot)

            self.get_logger().info(
                f'── HOLD  hata: dx={ex:+.3f}  dy={ey:+.3f}  dphi={math.degrees(ephi):+.1f}°\n'
                f'   GERCEK x={self.pose[0]:+.3f}  y={self.pose[1]:+.3f}  '
                f'phi={math.degrees(self.pose[2]):+.1f}°',
                throttle_duration_sec=0.5,
            )
            return

        # ── Trayektori modu (FF + P geribildirim) ────────────────────────────
        tx_seg, ty_seg, t_start = self._segments[-1][0], self._segments[-1][1], self._segments[-1][2]
        for tx_s, ty_s, ts, te in self._segments:
            if t < te:
                tx_seg, ty_seg, t_start = tx_s, ty_s, ts
                break
        seg_t = t - t_start

        x_r, y_r   = tx_seg.s(seg_t),     ty_seg.s(seg_t)
        vx_r, vy_r = tx_seg.s_dot(seg_t), ty_seg.s_dot(seg_t)
        phi_r, w_r = self.tphi.s(t),      self.tphi.s_dot(t)

        # Konum ve yon hatasi
        ex   = x_r   - self.pose[0]
        ey   = y_r   - self.pose[1]
        ephi = phi_r - self.pose[2]

        # FF + P: referans hiz + hata duzeltmesi
        vx_cmd = vx_r + KP_XY  * ex
        vy_cmd = vy_r + KP_XY  * ey
        w_cmd  = w_r  + KP_PHI * ephi

        # Gercek yon (pose[2]) kullanilarak world → teker hizi donusumu
        phi_dot = self.kin.forward_world(
            np.array([vx_cmd, vy_cmd, w_cmd]), self.pose[2]
        )
        self._send(phi_dot)

        self.get_logger().info(
            f'── t={t:5.2f}/{T_TOTAL:.2f} s\n'
            f'   PLAN   x={x_r:+.3f}  y={y_r:+.3f}  phi={math.degrees(phi_r):+.1f}°  '
            f'vx={vx_r:+.3f}  vy={vy_r:+.3f}  w={w_r:+.3f}\n'
            f'   GERCEK x={self.pose[0]:+.3f}  y={self.pose[1]:+.3f}  '
            f'phi={math.degrees(self.pose[2]):+.1f}°\n'
            f'   HATA   ex={ex:+.4f}  ey={ey:+.4f}  ephi={math.degrees(ephi):+.2f}°'
        )

    # ── Yardimcilar ──────────────────────────────────────────────────────────
    def _send(self, phi_dot: np.ndarray):
        q1 = int(round(phi_dot[0] * DIR_W1 * RAD2QPPS))
        q2 = int(round(phi_dot[1] * DIR_W2 * RAD2QPPS))
        q3 = int(round(phi_dot[2] * DIR_W3 * RAD2QPPS))
        self.rc_a.SpeedM2(ADDR_A, q1)
        self.rc_a.SpeedM1(ADDR_A, q2)
        self.rc_b.SpeedM2(ADDR_B, q3)

    def _stop(self):
        self.rc_a.SpeedM2(ADDR_A, 0)
        self.rc_a.SpeedM1(ADDR_A, 0)
        self.rc_b.SpeedM2(ADDR_B, 0)

    def _publish_odom(self):
        odom = Odometry()
        odom.header.stamp         = self.get_clock().now().to_msg()
        odom.header.frame_id      = 'odom'
        odom.child_frame_id       = 'base_link'
        odom.pose.pose.position.x = float(self.pose[0])
        odom.pose.pose.position.y = float(self.pose[1])
        half = self.pose[2] / 2.0
        odom.pose.pose.orientation.z = float(math.sin(half))
        odom.pose.pose.orientation.w = float(math.cos(half))
        self.odom_pub.publish(odom)

    def destroy_node(self):
        self._running = False
        self._stop()
        self.rc_a.close()
        self.rc_b.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GoStopFbNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

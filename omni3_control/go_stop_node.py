#!/usr/bin/env python3
"""
go_stop_node.py — GO-STOP quintic trayektori takibi (feedforward only).

Baslangic (0, 0, 0) noktasindan (X_REF, Y_REF, PHI_REF) hedefine
T_TOTAL saniyede rest-to-rest quintic polinom ile gider. Her eksen
icin bagimsiz s(t) hesaplanir; s_dot(t) dunya cercevesi hiz olarak
inverse kinematics uzerinden tekerleklere feedforward edilir.

Wheel haritasi (move_1m_node ile ayni):
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
PORT_A    = '/dev/roboclaw_front'   # RoboClaw 1 (0x80) — W1 M2 / W2 M1
PORT_B    = '/dev/roboclaw_rear'    # RoboClaw 2 (0x81) — W3 M2
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
OBS_X      = 0.5    # [m]   engel merkez x
OBS_Y      = 0.0    # [m]   engel merkez y
OBS_R      = 0.10   # [m]   engel yari capi
OBS_MARGIN = 0.10   # [m]   guvenlik tamponu (r_safe = OBS_R + OBS_MARGIN)
OBS_ENABLE = True   # False → engel yok say, dogrudan hedefe git


class GoStopNode(Node):

    def __init__(self):
        super().__init__('go_stop_node')

        # Kinematics
        self.kin = OmniKinematics(OmniParams(
            wheel_radius=WHEEL_RADIUS,
            robot_radius=ROBOT_RADIUS,
            beta=WHEEL_BETAS,
        ))

        # Iki RoboClaw
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

        # Paylasilan encoder verisi
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

        # Durum
        self.pose = np.zeros(3)
        self.done = False

        # Quintic plan: tek segment (dogrudan) veya iki segment (engel etrafindan)
        S = np.array([0.0, 0.0])
        G = np.array([X_REF, Y_REF])
        V = None
        if OBS_ENABLE:
            C      = np.array([OBS_X, OBS_Y])
            r_safe = OBS_R + OBS_MARGIN
            V = self._plan_via_point(S, G, C, r_safe)

        if V is None:
            # Carpisma yok → tek quintic
            self._segments = [(
                QuinticTrajectory(0.0, X_REF, T_TOTAL),
                QuinticTrajectory(0.0, Y_REF, T_TOTAL),
                0.0, T_TOTAL,
            )]
            plan_info = 'tek segment (engel yok / hizada degil)'
        else:
            # Iki segment, zamani uzunluk oranina gore bol
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

        # phi her zaman tek quintic — yon segmentlerden bagimsiz
        self.tphi = QuinticTrajectory(0.0, PHI_REF, T_TOTAL)
        self._t0  = time.monotonic()

        # /odom
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # 20 Hz kontrol
        self.create_timer(DT, self._control_loop)

        self.get_logger().info('GoStopNode basladi.')
        self.get_logger().info(
            f'Hedef: x={X_REF} m, y={Y_REF} m, phi={PHI_REF} rad  |  T={T_TOTAL} s'
        )
        if OBS_ENABLE:
            self.get_logger().info(
                f'Engel: ({OBS_X}, {OBS_Y}) r={OBS_R}  margin={OBS_MARGIN}  → {plan_info}'
            )

    # ── Engel etrafi yol planlama ────────────────────────────────────────────
    @staticmethod
    def _plan_via_point(S: np.ndarray, G: np.ndarray, C: np.ndarray, r_safe: float):
        """
        Daire engel (C, r_safe) S-G dogru parcasini kesiyorsa, dairenin
        iki tarafinda da bir via-point adayi hesaplar ve toplam mesafeyi
        kisaltani dondurur. Carpisma yoksa None.
        """
        SG = G - S
        L = float(np.linalg.norm(SG))
        if L < 1e-9:
            return None
        t_proj = float(np.dot(C - S, SG) / (L * L))
        # Engel SG segmentinin "yaninda" mi? (uclar disindaysa detour gereksiz)
        if t_proj <= 0.0 or t_proj >= 1.0:
            return None
        P = S + t_proj * SG
        d = float(np.linalg.norm(C - P))
        if d >= r_safe:
            return None   # Yol zaten engele temas etmiyor
        SG_hat = SG / L
        n_perp = np.array([-SG_hat[1], SG_hat[0]])   # 90° CCW
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
        if self.done:
            return

        with self._enc_lock:
            cur     = list(self._enc_counts)
            last_ts = self._enc_last_ts

        # Watchdog
        if time.monotonic() - last_ts > ENC_STALE_SEC:
            self.get_logger().error(
                f'Encoder verisi bayat ({time.monotonic() - last_ts:.2f}s) — motorlar durduruluyor',
                throttle_duration_sec=1.0,
            )
            self._stop()
            return

        # Odometri guncelle (sadece izleme/log icin — FF kontrolu degil)
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

        # Quintic referansi — aktif segmenti bul
        t = time.monotonic() - self._t0
        tx_seg, ty_seg, t_start = self._segments[-1][0], self._segments[-1][1], self._segments[-1][2]
        for tx_s, ty_s, ts, te in self._segments:
            if t < te:
                tx_seg, ty_seg, t_start = tx_s, ty_s, ts
                break
        seg_t = t - t_start
        x_r, y_r       = tx_seg.s(seg_t),     ty_seg.s(seg_t)
        vx_r, vy_r     = tx_seg.s_dot(seg_t), ty_seg.s_dot(seg_t)
        phi_r, w_r     = self.tphi.s(t),      self.tphi.s_dot(t)

        self.get_logger().info(
            f'── t={t:5.2f}/{T_TOTAL:.2f} s\n'
            f'   PLAN   x={x_r:+.3f}  y={y_r:+.3f}  phi={math.degrees(phi_r):+.1f}°  '
            f'vx={vx_r:+.3f}  vy={vy_r:+.3f}  w={w_r:+.3f}\n'
            f'   GERCEK x={self.pose[0]:+.3f}  y={self.pose[1]:+.3f}  '
            f'phi={math.degrees(self.pose[2]):+.1f}°'
        )

        # Trayektori bitti mi?
        if t >= T_TOTAL:
            self.get_logger().info('Trayektori tamamlandi.')
            self._stop()
            self.done = True
            return

        # FF: dunya hizi → teker hizi → QPPS
        # theta olarak planlanan phi(t) kullanilir (FF varsayimi: takip mukemmel)
        phi_dot = self.kin.forward_world(np.array([vx_r, vy_r, w_r]), phi_r)
        q1 = int(round(phi_dot[0] * DIR_W1 * RAD2QPPS))
        q2 = int(round(phi_dot[1] * DIR_W2 * RAD2QPPS))
        q3 = int(round(phi_dot[2] * DIR_W3 * RAD2QPPS))

        self.rc_a.SpeedM2(ADDR_A, q1)
        self.rc_a.SpeedM1(ADDR_A, q2)
        self.rc_b.SpeedM2(ADDR_B, q3)

    # ── Yardimcilar ──────────────────────────────────────────────────────────
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

    def _stop(self):
        self.rc_a.SpeedM2(ADDR_A, 0)
        self.rc_a.SpeedM1(ADDR_A, 0)
        self.rc_b.SpeedM2(ADDR_B, 0)

    def destroy_node(self):
        self._running = False
        self._stop()
        self.rc_a.close()
        self.rc_b.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GoStopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

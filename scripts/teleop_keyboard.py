#!/usr/bin/env python3
"""Keyboard teleop for the G1 GR00T WBC policies - publishes /g1/cmd_vel.

Run in a REAL terminal while the headed warehouse sim is up:

    cd ~/Projects/thesis/G1_sim && source .envrc && python scripts/teleop_keyboard.py

Data path: this node -> /g1/cmd_vel (Twist) -> sim OmniGraph subscriber ->
WbcBridge.step(). Command norm > 0.05 (WALK_CMD_DEADBAND) selects the Walk
policy; at/below it Balance runs - so releasing every key publishes zero and
the robot just stands. Deadman by design: a keypress keeps its command alive
for KEY_TTL seconds, then it expires.

Keys (hold to drive):
    W/S     forward / back        vx  +/-0.35 m/s
    A/D     turn left / right     wz  +/-0.60 rad/s
    Q/E     strafe left / right   vy  +/-0.30 m/s
    SPACE   stop immediately
    Ctrl-C  quit (termios restored)

Arrow keys work as WASD equivalents.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

TOPIC_CMD_VEL = "/g1/cmd_vel"
PUBLISH_HZ = 20.0
KEY_TTL = 0.40  # s a keypress keeps commanding after the last repeat
VX, VY, WZ = 0.35, 0.30, 0.60  # patrol-proven vx; conservative vy/wz
WALK_CMD_DEADBAND = 0.05  # mirrors g1_sim/wbc_bridge.py

# key -> (vx, vy, wz) contribution while held
KEYMAP = {
    "w": (+VX, 0.0, 0.0),
    "s": (-VX, 0.0, 0.0),
    "a": (0.0, 0.0, +WZ),
    "d": (0.0, 0.0, -WZ),
    "q": (0.0, -VY, 0.0),
    "e": (0.0, +VY, 0.0),
    # arrows: ESC [ A/B/C/D (composed by _read_keys)
    "\x1b[A": (+VX, 0.0, 0.0),
    "\x1b[B": (-VX, 0.0, 0.0),
    "\x1b[D": (0.0, 0.0, +WZ),
    "\x1b[C": (0.0, 0.0, -WZ),
}


class Teleop(Node):
    def __init__(self) -> None:
        super().__init__("g1_keyboard_teleop")
        self._pub = self.create_publisher(Twist, TOPIC_CMD_VEL, 10)

    def send(self, vx: float, vy: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = wz
        self._pub.publish(msg)


def _read_keys(fd: int) -> list[str]:
    """Drain every pending key; compose ESC-sequences into one token."""
    keys: list[str] = []
    while select.select([fd], [], [], 0)[0]:
        ch = os.read(fd, 1).decode("utf-8", "ignore")
        if ch == "\x1b" and select.select([fd], [], [], 0.005)[0]:
            ch += os.read(fd, 1).decode("utf-8", "ignore")
            if ch[-1] == "[" and select.select([fd], [], [], 0.005)[0]:
                ch += os.read(fd, 1).decode("utf-8", "ignore")
        keys.append(ch)
    return keys


def main() -> None:
    if not sys.stdin.isatty():
        sys.exit("teleop_keyboard needs a real terminal (TTY) - run it interactively, e.g. in konsole/xterm.")

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    rclpy.init()
    node = Teleop()
    held: dict[str, tuple[float, float, float, float]] = {}  # key -> (vx, vy, wz, expires)

    print(__doc__)
    print(f"publishing {TOPIC_CMD_VEL} @ {PUBLISH_HZ:.0f} Hz - focus this window and hold keys")
    tty.setcbreak(fd)
    try:
        next_t = time.monotonic()
        while rclpy.ok():
            now = time.monotonic()
            for key in _read_keys(fd):
                if key == " ":
                    held.clear()
                elif key in KEYMAP:
                    vx, vy, wz = KEYMAP[key]
                    held[key] = (vx, vy, wz, now + KEY_TTL)
            vx = vy = wz = 0.0
            for key, (kx, ky, kz, exp) in list(held.items()):
                if now > exp:
                    del held[key]
                else:
                    vx += kx
                    vy += ky
                    wz += kz
            node.send(vx, vy, wz)
            rclpy.spin_once(node, timeout_sec=0.0)

            mode = "WALK" if (vx * vx + vy * vy + wz * wz) ** 0.5 > WALK_CMD_DEADBAND else "BALANCE"
            sys.stdout.write(f"\r vx={vx:+.2f} m/s  vy={vy:+.2f}  wz={wz:+.2f} rad/s  {mode:<6} ")
            sys.stdout.flush()

            next_t += 1.0 / PUBLISH_HZ
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()  # fell behind; resync
    except KeyboardInterrupt:
        pass
    finally:
        node.send(0.0, 0.0, 0.0)  # leave the robot balancing
        time.sleep(0.1)
        rclpy.spin_once(node, timeout_sec=0.0)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        rclpy.shutdown()
        print("\nzero cmd sent - robot back to Balance. bye")


if __name__ == "__main__":
    main()

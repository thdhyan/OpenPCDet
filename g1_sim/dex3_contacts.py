"""Per-fingertip contact sensing for the Dex3 hands.

One ``IsaacContactSensor`` per fingertip link (thumb_2, index_1, middle_1 on
each hand), published as ``geometry_msgs/WrenchStamped`` on
``/g1/dex3/<side>/<finger>/contact``:

- ``wrench.force``: net contact force on the tip in the World frame, summed
  from the raw contact impulses of the last physics step (impulse / dt).
- ``wrench.torque``: zero (not measured).
- ``header.frame_id``: ``World``; ``header.stamp``: sim time.

This is the simple placeholder for the planned TacSL visuotactile sensors
(docs/TACSL_PLAN.md): the tip list and topic namespace are the integration
points a tactile array will extend.
"""

from __future__ import annotations

import numpy as np

TIP_LINKS = {
    f"{side}_{finger}": f"{side}_hand_{link}_link"
    for side in ("left", "right")
    for finger, link in (("thumb", "thumb_2"), ("index", "index_1"), ("middle", "middle_1"))
}
TOPIC_FMT = "/g1/dex3/{side}/{finger}/contact"


def spawn_tip_contacts(robot_prim_path: str, max_force_n: float = 200.0) -> dict[str, str]:
    """Create a contact sensor under every Dex3 fingertip link.

    Returns ``{tip_name: sensor_prim_path}``; tips missing from the USD are
    skipped, so a handless robot yields an empty dict.
    """
    import omni.usd
    from isaacsim.sensors.experimental.physics import Contact

    stage = omni.usd.get_context().get_stage()
    sensors = {}
    for tip, link in TIP_LINKS.items():
        link_path = f"{robot_prim_path}/{link}"
        if not stage.GetPrimAtPath(link_path).IsValid():
            continue
        path = f"{link_path}/tip_contact"
        # radius -1: report contacts anywhere on the link's collision shape.
        Contact.create(path, min_threshold=0.0, max_threshold=max_force_n, radius=-1.0)
        sensors[tip] = path
    return sensors


def net_force(sensor) -> np.ndarray:
    """Net World-frame contact force on the sensor's body from the last
    physics step: sum of raw contact impulses / dt."""
    force = np.zeros(3)
    for c in sensor.get_raw_data():
        dt = float(c.get("dt", 0.0))
        if dt > 0.0:
            imp = c["impulse"]  # {"x", "y", "z"} dict from the C++ interface
            force += np.array([imp["x"], imp["y"], imp["z"]], dtype=float) / dt
    return force


class TipContactPublisher:
    """Reads the tip contact sensors each step and publishes at ``publish_rate``."""

    def __init__(self, sensor_paths: dict[str, str], publish_rate: float = 60.0):
        from geometry_msgs.msg import WrenchStamped
        from isaacsim.sensors.experimental.physics import ContactSensor
        from rclpy.node import Node

        self._WrenchStamped = WrenchStamped
        self.sensors = {tip: ContactSensor(path) for tip, path in sensor_paths.items()}
        self.node = Node("g1_dex3_tip_contacts")
        self.publishers = {}
        for tip in self.sensors:
            side, finger = tip.split("_")
            self.publishers[tip] = self.node.create_publisher(WrenchStamped, TOPIC_FMT.format(side=side, finger=finger), 10)
        self.period = 1.0 / publish_rate
        self._next = -float("inf")
        self.last_forces: dict[str, np.ndarray] = {tip: np.zeros(3) for tip in self.sensors}

    def publish(self, sim_time: float) -> None:
        if sim_time < self._next:
            return
        self._next = sim_time + self.period
        for tip, sensor in self.sensors.items():
            f = net_force(sensor)
            self.last_forces[tip] = f
            msg = self._WrenchStamped()
            msg.header.frame_id = "World"
            msg.header.stamp.sec = int(sim_time)
            msg.header.stamp.nanosec = int((sim_time - int(sim_time)) * 1e9)
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = (float(v) for v in f)
            self.publishers[tip].publish(msg)

    def destroy(self) -> None:
        self.node.destroy_node()

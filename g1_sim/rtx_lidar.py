"""RTX LiDAR Livox Mid-360 for the G1, using emitter state arrays.

The warp ray-caster only tests rays against a fixed list of static meshes, so
it cannot see the robot or anything spawned later. RTX LiDAR ray-traces the
real rendered scene instead, which means self-occlusion and dynamic objects
come for free.

The Mid-360's sweep is non-repetitive, so it is expressed as a sequence of
``emitterStates`` - one per frame - generated from the real scan pattern by
``scripts/gen_mid360_rtx_config.py``. Because the Hydra API caps one prim at
~5 MB of emitter data, the pattern is split across several prims sharing a
transform; their union is the full pattern.

Requires **Isaac Sim 6.0+**. On 5.1 every RTX sensor renders at the simulation
frame rate regardless of ``tickRate``, so a 10 Hz sensor in a 60 Hz sim fires
6x too often - which both corrupts the point rate and triggers CUDA buffer
races. 6.0 enables multi-tick rendering by default and honours ``tickRate``.

Cost. Every prim is a separate ray-tracing pass each render, so the load
scales with prim count and emitters per state, and RTX LiDAR is expensive on
a small GPU. Three presets trade sweep completeness against startup and frame
time:

===========  ======  ==========  =======  =====================
Preset       Prims   States      Size     Cycle
===========  ======  ==========  =======  =====================
``fast``     4       2 each      5.9 MB   0.2 s  (default)
``light``    4       5 each      15 MB    0.5 s
(full)       8       5 each      30 MB    0.5 s, denser sweep
===========  ======  ==========  =======  =====================

If the GUI stutters, pass ``--num-prims 2`` or add ``--no-camera``; the camera
renders three annotators per frame and is usually the larger cost of the two.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "assets/lidar_configs"

# Maps our generated JSON profile's field names (schema of the deprecated
# ``IsaacSensorCreateRtxLidar`` Kit command's config files) to the real,
# currently-live ``OmniSensorGenericLidarCoreAPI`` USD schema's attribute
# names - confirmed by reading the schema straight from
# ``omni.sensors.nv.common``'s ``generatedSchema.usda`` and a known-good
# example (``omni.cip.mega``'s ``OmniLidarCheckerPass.usda``). Renamed or
# dropped where the two schemas disagree; see ``spawn_mid360``'s docstring
# for why this translation exists at all instead of using the JSON directly.
_PROFILE_TO_CORE_ATTR = {
    "nearRangeM": "nearRangeM",
    "farRangeM": "farRangeM",
    "rangeResolutionM": "rangeResolutionM",
    "rangeAccuracyM": "rangeAccuracyM",
    "minReflectance": "minReflectance",
    "pulseTimeNs": "pulseTimeNs",
    "maxReturns": "maxReturns",
    "scanRateBaseHz": "scanRateBaseHz",
    "patternFiringRateHz": "patternFiringRateHz",
    "numberOfEmitters": "numberOfEmitters",
    "numberOfChannels": "numberOfChannels",
    "rangeCount": "rangeCount",
    "azimuthErrorMean": "azimuthErrorMean",
    "azimuthErrorStd": "azimuthErrorStd",
    "elevationErrorMean": "elevationErrorMean",
    "elevationErrorStd": "elevationErrorStd",
    "validStartAzimuthDeg": "validStartAzimuthDeg",
    "validEndAzimuthDeg": "validEndAzimuthDeg",
    "stateResolutionStep": "stateResolutionStep",
    # Renamed between schemas.
    "minReflectanceRange": "minReflectionRangeM",
    "wavelengthNm": "waveLengthNm",
    # avgPowerW and emitterStateCount have no equivalent in the real schema
    # (peakPowerW exists instead, and state count is implicit in how many
    # emitterState instances are applied) - dropped, not mapped.
}
_TOKEN_VALUES = {
    # JSON profiles store lowercase; the USD schema tokens are uppercase
    # (NVIDIA's own converter authors scanType="ROTARY"/"SOLID_STATE" -
    # Example_Rotary.usda / Example_Solid_State.usda).
    "scanType": {"solidState": "SOLID_STATE", "rotary": "ROTARY"},
    "intensityProcessing": {"normalization": "NORMALIZATION"},
}

# Matches the real device and the generated configs.
SCAN_RATE_HZ = 10.0

# The URDF's mid360_joint (assets/robot/g1_29/g1_29dof.urdf, used by
# convert_g1_urdf_to_usd.py): xyz=(0.0002835, 0.00003, 0.4188),
# rpy=(3.14, 0, 0) — a 180 deg roll, no pitch. This pose is baked into the
# USD's mid360_link prim, which is what the sim now mounts the sensor *as*
# (identity transform) — see the NOTE below.
# NVIDIA's GR00T-WholeBodyControl repo ships a *different* g1_29dof.urdf
# with a different mid360_joint (xyz z=0.40618, rpy=(0, 0.0401, 0), no roll) -
# this file previously copied that one by mistake. Since the USD's frames come
# from the vendored URDF, not GR00T's, the mismatch pointed the sensor's local Z
# into the ceiling instead of the ground. This is now moot for the sim path
# (the sensor uses mid360_link's own pose, identity transform) but still applies
# to any code that mounts on torso_link with these constants.
# Briefly lowered 0.2 m (to 0.2188) 2026-08-10 to test whether the real
# mount's ~9.7 m blind cone explained a narrow-arc-not-ring cloud. Reverted
# to the real URDF mount height per user request - didn't resolve the arc
# issue and the sensor's still-unexplained non-publishing that session took
# priority. If revisiting the blind-cone theory, the math was: blind_radius
# = 8x mount height, so lowering height shrinks the cone and lets closer
# geometry register.
# NOTE (2026-08-11): the sim scripts no longer apply these to the sensor prim.
# The Mid-360 is now spawned *as* the USD's `mid360_link` prim with an identity
# local transform, so the URDF pose already baked into that prim is the single
# source of truth and the ROS frame_id (`mid360_link`) matches the point origin
# exactly. These constants are kept as documentation of the URDF joint and are
# still used by the diag scripts (which mount on torso_link) and for the
# mount-height estimate below.
MID360_POS = (0.0002835, 0.00003, 0.4188)
# rpy=(3.14, 0, 0) -> wxyz quaternion for a 180 deg roll about X-axis.
_MID360_ROLL = __import__("math").pi
MID360_QUAT_WXYZ = (
    __import__("math").cos(_MID360_ROLL / 2),  # w
    __import__("math").sin(_MID360_ROLL / 2),  # x (roll)
    0.0,                                        # y (pitch=0)
    0.0,                                        # z (yaw=0)
)


def install_configs(config_dir: Path | str = CONFIG_DIR) -> list[str]:
    """Copy the generated profiles where Isaac Sim's sensor loader looks.

    ``IsaacSensorCreateRtxLidar`` resolves ``config=`` against the RTX sensor
    extension's own ``data/lidar`` directory, not against an arbitrary path, so
    the JSON files have to be placed there.

    Returns the config names (file stems) that were installed.
    """
    import isaacsim

    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        raise FileNotFoundError(
            f"{config_dir} not found - run scripts/gen_mid360_rtx_config.py first"
        )

    profiles = sorted(config_dir.glob("Livox_Mid360_*.json"))
    if not profiles:
        raise FileNotFoundError(f"no Livox_Mid360_*.json in {config_dir}")

    pkg_root = Path(isaacsim.__file__).parent
    targets = sorted(pkg_root.glob("extscache/omni.sensors.nv.common-*/data/lidar"))
    if not targets:
        raise RuntimeError("could not locate the RTX sensor config directory")

    # These files are ~4 MB each and Isaac Sim re-parses them at load, so only
    # copy when the destination is missing or stale - an unconditional copy of
    # 15 MB visibly slows every launch.
    copied = 0
    for target in targets:
        for profile in profiles:
            dest = target / profile.name
            if dest.exists() and dest.stat().st_mtime >= profile.stat().st_mtime:
                continue
            shutil.copy2(profile, dest)
            copied += 1

    if copied:
        print(f"[RTX] installed {copied} lidar profile(s)")

    return [p.stem for p in profiles]


# (attribute name suffix -> Sdf type name string) for the per-profile scalar
# fields this module actually sets. Everything here is a plain float scalar
# in the real schema except the tokens/uints called out explicitly below.
_UINT_SCALAR_ATTRS = {
    "maxReturns", "scanRateBaseHz", "patternFiringRateHz", "numberOfEmitters",
    "numberOfChannels", "rangeCount", "pulseTimeNs", "stateResolutionStep",
}
_TOKEN_ATTRS = {"scanType", "rayType", "intensityProcessing", "intensityMappingType"}


def _profile_to_attributes(profile: dict) -> list[tuple[str, str, object]]:
    """Translate one generated JSON profile into a flat list of
    ``(attribute_name, sdf_type_name, value)`` for
    ``OmniSensorGenericLidarCoreAPI``/``...EmitterStateAPI``.

    Why this exists at all: ``omni.kit.commands.execute("IsaacSensorCreateRtxLidar",
    config=name, ...)`` resolves ``config`` by name against a directory the
    ``isaacsim.sensors.rtx`` extension appears to index once, at
    extension-enable time - not on every call. Every prim created this way
    logged ``Config '<name>' not found for OmniLidar`` regardless of the
    JSON's name or content (live-verified 2026-08-11 across three separate
    fixes: restructuring emitterStates into elevation-binned lines, renaming
    the config, and moving the install ahead of extension-enable - all
    produced byte-identical output). The prim was silently left at the
    ``OmniSensorGenericLidarCoreAPI`` schema's own built-in defaults the
    entire time: its default ``elevationDeg`` is a repeating 32-value ramp
    from -15 to +10 deg, and the *last two* values of every cycle are
    9.19 and 10.0 - an exact match to what every one of those "fixes"
    still produced. ``isaacsim.sensors.experimental.rtx`` (the
    non-deprecated replacement) resolves named configs to USD assets on
    Nucleus instead of JSON, and Livox isn't among NVIDIA's shipped
    profiles - so this authors the ``omni:sensor:Core:*`` attributes
    directly from our own JSON, sidestepping config-name resolution
    entirely. Field mapping confirmed against
    ``omni.sensors.nv.common``'s ``generatedSchema.usda`` and
    ``omni.cip.mega``'s ``OmniLidarCheckerPass.usda`` (a real validated
    example), not guessed.

    Returns a flat list rather than a dict passed to ``Lidar(attributes=...)``
    because that path routes through ``omni.replicator.core``'s
    ``modify.set_attributes()``, which does ``python_class(*v)`` for any
    Python ``list`` value - star-unpacking every element as a positional
    constructor arg. Fine for a 3-element translate, broken for a
    20,000-element ``azimuthDeg`` (confirmed live 2026-08-11:
    ``Boost.Python.ArgumentError``, no C++ signature takes 20,000 floats).
    Authoring straight through ``Usd.Attribute.Set()`` - the caller's job,
    using this function's output - sidesteps that helper entirely.
    """
    out: list[tuple[str, str, object]] = []

    def add(name: str, sdf_type: str, value) -> None:
        out.append((f"omni:sensor:Core:{name}", sdf_type, value))

    for src, dst in _PROFILE_TO_CORE_ATTR.items():
        if src not in profile:
            continue
        if dst in _UINT_SCALAR_ATTRS:
            add(dst, "uint", int(profile[src]))
        else:
            add(dst, "float", float(profile[src]))

    for field in _TOKEN_ATTRS:
        if field in profile:
            value = _TOKEN_VALUES.get(field, {}).get(profile[field], profile[field])
            add(field, "token", value)

    if "ranges" in profile:
        add("rangesMinM", "float[]", [float(r["min"]) for r in profile["ranges"]])
        add("rangesMaxM", "float[]", [float(r["max"]) for r in profile["ranges"]])

    # numLines/numRaysPerLine aren't in _PROFILE_TO_CORE_ATTR (not a 1:1 rename,
    # numRaysPerLine is an array) - without these the schema default numLines=0
    # stays in place while gen_mid360_rtx_config.py's per-state "bank" values
    # reference line indices 0..39, and the RTX engine rejects every param
    # update with "bankId 0 at index 0 is greater than the profile numLines 0"
    # - confirmed live 2026-08-11 via scripts/diag_warehouse_lidar.py: 0 points
    # collected across 300 steps with this omission, schema confirms
    # omni:sensor:Core:numLines (uint) / omni:sensor:Core:numRaysPerLine
    # (uint[], one entry per line) are the real attribute names.
    if "numLines" in profile:
        add("numLines", "uint", int(profile["numLines"]))
    if "numRaysPerLine" in profile:
        add("numRaysPerLine", "uint[]", [int(n) for n in profile["numRaysPerLine"]])

    # Report rate - NOT in the JSON->USD mapping table above because the
    # generated profiles never carried it, but every NVIDIA-converted
    # solid-state USD authors it (Example_Solid_State.usda:
    # ``uint omni:sensor:Core:reportRateBaseHz = 10``, equal to
    # patternFiringRateHz). It is not even declared in
    # omni.usd.schema.omni_sensors' generatedSchema - plugin-internal - so an
    # unauthored prim leaves it to the plugin's fallback. Without it the
    # engine emitted ~1 return per render out of 5,000 emitters/state
    # (live-isolated 2026-08-24, scripts/lidar_viz_isaacsim.py --enclose).
    if "patternFiringRateHz" in profile:
        add("reportRateBaseHz", "uint", int(profile["patternFiringRateHz"]))

    # Per-emitter optional arrays the NVIDIA converter always writes
    # (Example_Solid_State.usda / Simple_Example_Solid_State.usda). All zeros;
    # matching the reference byte-for-byte removes them as variables.
    # NOTE: states are 1-indexed (s001...) on every NVIDIA-authored USD -
    # s000 is never used by the converter, and the applied
    # OmniSensorGenericLidarCoreEmitterStateAPI instance name must match this
    # index (spawn_mid360 applies :s001..).
    for i, state in enumerate(profile["emitterStates"]):
        n = len(state["azimuthDeg"])
        zeros_f = [0.0] * n
        prefix = f"emitterState:s{i + 1:03d}:"
        add(prefix + "azimuthDeg", "float[]", state["azimuthDeg"])
        add(prefix + "elevationDeg", "float[]", state["elevationDeg"])
        if state.get("fireTimeNs"):
            add(prefix + "fireTimeNs", "uint[]", state["fireTimeNs"])
        # channelId is OPTIONAL - NVIDIA's Example_Rotary.json states carry
        # only azimuthDeg/elevationDeg/fireTimeNs.
        if state.get("channelId"):
            add(prefix + "channelId", "uint[]", state["channelId"])
        if state.get("rangeId"):
            add(prefix + "rangeId", "uint[]", state["rangeId"])
        if state.get("bank"):
            add(prefix + "bank", "uint[]", state["bank"])
        add(prefix + "distanceCorrectionM", "float[]", zeros_f)
        add(prefix + "emitterPeakPowerW", "float[]", zeros_f)
        add(prefix + "focalDistM", "float[]", zeros_f)
        add(prefix + "focalSlope", "float[]", zeros_f)
        add(prefix + "horOffsetM", "float[]", zeros_f)
        add(prefix + "vertOffsetM", "float[]", zeros_f)
        add(prefix + "reportRateDiv", "float[]", zeros_f)

    return out


# Identity quaternion: the URDF's mid360_joint already bakes the 180° roll
# into the USD's mid360_link prim. The sensor prims should mount with
# identity so their local frame == mid360_link frame (points match TF).
_IDENTITY_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)


def spawn_mid360(
    parent_prim_path: str,
    config_dir: Path | str = CONFIG_DIR,
    translation: tuple[float, float, float] = MID360_POS,
    orientation: tuple[float, float, float, float] = _IDENTITY_QUAT_WXYZ,
) -> list[str]:
    """Spawn the Mid-360 as several co-located RTX LiDAR prims.

    Args:
        parent_prim_path: prim to mount under, e.g.
            ``/World/envs/env_0/Robot/torso_link``.
        config_dir: directory holding the generated profiles (from
            ``scripts/gen_mid360_rtx_config.py``).
        translation: sensor offset from the parent, in metres.
        orientation: sensor rotation as a wxyz quaternion, relative to the
            mount. Identity for mid360_link, which already carries the URDF's
            180 deg roll.

    Returns:
        The spawned prim paths, one per profile.
    """
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom

    sdf_types = {
        "float": Sdf.ValueTypeNames.Float,
        "uint": Sdf.ValueTypeNames.UInt,
        "token": Sdf.ValueTypeNames.Token,
        "float[]": Sdf.ValueTypeNames.FloatArray,
        "uint[]": Sdf.ValueTypeNames.UIntArray,
    }

    config_dir = Path(config_dir)
    profiles = sorted(config_dir.glob("Livox_Mid360_*.json"))
    if not profiles:
        raise FileNotFoundError(f"no Livox_Mid360_*.json in {config_dir}")

    stage = omni.usd.get_context().get_stage()
    prim_paths = []
    for profile_path in profiles:
        name = profile_path.stem
        profile = json.loads(profile_path.read_text())["profile"]
        entries = _profile_to_attributes(profile)

        path = f"{parent_prim_path}/{name}"
        # Authored directly via pxr.Usd, not isaacsim.sensors.experimental.rtx's
        # Lidar(attributes=...) - see _profile_to_attributes' docstring for
        # why: that path star-unpacks large arrays and crashes.
        prim = stage.DefinePrim(path, "OmniLidar")
        prim.AddAppliedSchema("OmniSensorGenericLidarCoreAPI")
        # Emitter-state instances are 1-indexed (s001...) - matches
        # _profile_to_attributes' attribute names and every NVIDIA-authored
        # reference USD. See the per-state note there.
        for i in range(len(profile["emitterStates"])):
            prim.AddAppliedSchema(f"OmniSensorGenericLidarCoreEmitterStateAPI:s{i + 1:03d}")

        for attr_name, type_key, value in entries:
            attr = prim.CreateAttribute(attr_name, sdf_types[type_key])
            attr.Set(value)

        # Isaac Sim 6.0 enables multi-tick rendering, so tickRate genuinely
        # limits how often the sensor renders - see rtx_lidar.py's module
        # docstring for why that matters on 6.0 vs 5.1.
        prim.CreateAttribute("omni:sensor:tickRate", Sdf.ValueTypeNames.Float).Set(SCAN_RATE_HZ)
        # The schema default is False ("when true the model will accumulate
        # the outputs until one scan is complete"); the deprecated
        # IsaacSensorCreateRtxLidar command explicitly overrode it to True
        # after creation (isaacsim.sensors.rtx's commands.py, do()) - carried
        # over here since we're not going through that command anymore.
        prim.CreateAttribute("omni:sensor:Core:accumulateOutputs", Sdf.ValueTypeNames.Bool).Set(True)
        # GMO default elementsCoordsType is SPHERICAL: gmo.x=azimuth(deg),
        # gmo.y=elevation(deg), gmo.z=range(m). Switch to CARTESIAN so
        # gmo.x/y/z are metric sensor-frame Cartesian (ISO8855: +x=front,
        # +y=left, +z=up). Without this, treating the spherical fields as
        # x/y/z produces garbage point positions and a broken range filter.
        # Schema token values: "CARTESIAN" | "SPHERICAL" (generatedSchema.usda,
        # omni.sensors.nv.common-3.0.0, confirmed 2026-08-13).
        prim.CreateAttribute(
            "omni:sensor:Core:elementsCoordsType", Sdf.ValueTypeNames.Token
        ).Set("CARTESIAN")

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*orientation))

        prim_paths.append(path)

    return prim_paths


SCAN_PATTERN = REPO / "assets/scan_patterns/mid360.npy"
POINTS_PER_FRAME = 20000


def pattern_frame_deg(frame):
    """One recorded Mid-360 frame (N x 2 radians: azimuth, elevation) as the
    (azimuthDeg, elevationDeg) arrays the solid-state emitter state expects.

    Azimuth is negated: the RTX solid-state engine sweeps clockwise, so
    returns land at -azimuthDeg in the sensor frame (live-measured
    2026-09-23 with per-sector test patterns).
    """
    import numpy as np

    az = np.degrees(frame[:, 0])
    az = np.where(az > 180.0, az - 360.0, az)
    return -az, np.degrees(frame[:, 1])


class ScanPatternCycler:
    """Stream consecutive recorded Mid-360 frames into a solid-state prim.

    The RTX solid-state engine only reproduces authored directions with a
    single emitter state - a 10-state profile squeezed -7..52 deg into
    -20..10 deg (live-measured 2026-09-23). So the prim carries one state and
    this rewrites its azimuth/elevation arrays with the next real frame every
    scan, which keeps the pattern non-repetitive like the device.
    """

    def __init__(self, prim_path: str, pattern: Path | str = SCAN_PATTERN, scan_rate_hz: float = SCAN_RATE_HZ):
        import numpy as np
        import omni.usd

        data = np.load(pattern)
        n_frames = len(data) // POINTS_PER_FRAME
        self.frames = data[: n_frames * POINTS_PER_FRAME].reshape(n_frames, POINTS_PER_FRAME, 2)
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        self._az = prim.GetAttribute("omni:sensor:Core:emitterState:s001:azimuthDeg")
        self._el = prim.GetAttribute("omni:sensor:Core:emitterState:s001:elevationDeg")
        self.period = 1.0 / scan_rate_hz
        self.index = 0
        self._next = -float("inf")

    def step(self, sim_time: float) -> None:
        if sim_time < self._next:
            return
        self._next = sim_time + self.period
        az, el = pattern_frame_deg(self.frames[self.index])
        self._az.Set(az.astype("float32").tolist())
        self._el.Set(el.astype("float32").tolist())
        self.index = (self.index + 1) % len(self.frames)


def attach_ros2_publishers(
    prim_paths: list[str],
    topic: str = "/livox/mid360/points",
    frame_id: str = "mid360_link",
    sim_rate_hz: float = 60.0,
    combine: bool = True,
) -> str:
    """Publish the RTX LiDAR returns as ``sensor_msgs/PointCloud2``.

    Args:
        prim_paths: the sensor prims from :func:`spawn_mid360`.
        topic: topic to publish on. With ``combine``, every prim publishes here
            so subscribers see one merged cloud; otherwise each prim gets its
            own ``topic/a``, ``topic/b``, ... which is useful for debugging
            which part of the sweep a point came from.
        frame_id: TF frame the points are expressed in.
        sim_rate_hz: simulation frame rate, used to derive the publish divisor.
        combine: publish all prims to one topic.

    Returns:
        The graph path holding the publishers.
    """
    import omni.graph.core as og

    graph_path = "/ActionGraph/RtxLidarROS2"

    # Mirrors the graph Isaac Sim's own "ROS 2 OmniGraphs > RTX Lidar" menu
    # builds. Two pieces are easy to miss and leave the topic advertised but
    # silent: a ROS2Context feeding every helper, and render products created
    # *inside* the graph by IsaacCreateRenderProduct rather than beforehand by
    # replicator - the helper reads the graph's own render product path.
    nodes = [
        ("OnTick", "omni.graph.action.OnPlaybackTick"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
        # Renders one frame so the render products are valid before the helpers
        # first read them.
        ("RunOneFrame", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
    ]
    connections = [("OnTick.outputs:tick", "RunOneFrame.inputs:execIn")]
    values = []

    for i, prim_path in enumerate(prim_paths):
        suffix = chr(ord("A") + i)
        rp_node = f"RenderProduct_{suffix}"
        helper = f"Publish_{suffix}"

        nodes += [
            (rp_node, "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            (helper, "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
        ]
        connections += [
            ("RunOneFrame.outputs:step", f"{rp_node}.inputs:execIn"),
            (f"{rp_node}.outputs:execOut", f"{helper}.inputs:execIn"),
            (f"{rp_node}.outputs:renderProductPath", f"{helper}.inputs:renderProductPath"),
            ("Context.outputs:context", f"{helper}.inputs:context"),
        ]
        values += [
            (f"{rp_node}.inputs:cameraPrim", [prim_path]),
            # RTX LiDAR needs only a 1x1 texture: the returns come from the
            # sensor, not from the rendered image.
            (f"{rp_node}.inputs:width", 1),
            (f"{rp_node}.inputs:height", 1),
            (f"{helper}.inputs:type", "point_cloud"),
            (
                f"{helper}.inputs:topicName",
                topic if combine else f"{topic}/{chr(ord('a') + i)}",
            ),
            (f"{helper}.inputs:frameId", frame_id),
            (f"{helper}.inputs:queueSize", 1),
            # frameSkipCount is deprecated on 6.0 - tickRate (set in
            # spawn_mid360) governs the sensor's render rate instead.
            (f"{helper}.inputs:frameSkipCount", 0),
            # fullScan=False emits each frame's returns as they fire, which is
            # what makes the non-repetitive sweep visible over time.
            (f"{helper}.inputs:fullScan", False),
        ]

    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: nodes,
            og.Controller.Keys.CONNECT: connections,
            og.Controller.Keys.SET_VALUES: values,
        },
    )

    return graph_path


def blind_radius(mount_height: float, min_tilt_deg: float = 7.16) -> float:
    """Radius of the Mid-360's blind cone, in metres.

    The steepest downward ray is about -7.2 deg and there is no nadir ray, so
    nothing within roughly 8x the mount height is ever seen.
    """
    import math

    return mount_height / math.tan(math.radians(min_tilt_deg))

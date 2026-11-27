"""Programmatic MuJoCo scene builder using mujoco_menagerie + MjSpec."""

from __future__ import annotations

import mujoco
import mujoco_menagerie as mm

# Menagerie ships the YAM finger servo with kp=100, which is too weak to carry
# the peg reliably; the oracle is tuned against this stiffer jaw servo.
GRIPPER_KP = 800.0


def build_peg_insert_spec(
    socket_pos: tuple[float, float, float] = (0.38, 0.0, 0.02),
    peg_pos: tuple[float, float, float] = (0.28, -0.14, 0.045),
    hole_half_width: float = 0.018,
    wall_thickness: float = 0.015,
    wall_height: float = 0.04,
) -> mujoco.MjSpec:
    """Load i2rt_yam from MuJoCo Menagerie and attach floor, light, table, socket, and peg."""
    # Use the robot-only entry point: menagerie's scene.xml already defines a
    # "floor" geom and light, which would collide with the ones added below.
    spec = mm.get("i2rt_yam").spec("yam")
    world = spec.worldbody

    # Ground plane & overhead light (menagerie robot XMLs omit scene lighting/floor)
    floor = world.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [2.0, 2.0, 0.1]
    floor.pos = [0.0, 0.0, -0.04]
    floor.rgba = [0.95, 0.96, 0.97, 1.0]

    light = world.add_light()
    light.name = "overhead_light"
    light.pos = [0.4, 0.0, 1.5]
    light.dir = [0.0, 0.0, -1.0]
    light.diffuse = [0.85, 0.85, 0.85]

    # Workbench surface
    table = world.add_geom()
    table.name = "workbench_top"
    table.type = mujoco.mjtGeom.mjGEOM_BOX
    table.size = [0.45, 0.45, 0.02]
    table.pos = [0.35, 0.0, -0.02]
    table.rgba = [0.88, 0.90, 0.92, 1.0]

    # Convex 4-wall socket assembly (avoids non-convex mesh collision issues)
    socket_body = world.add_body()
    socket_body.name = "socket_fixture"
    socket_body.pos = list(socket_pos)

    hw = hole_half_width
    wt = wall_thickness
    hh = wall_height / 2.0
    span = hw + wt

    wall_specs = [
        ("socket_wall_n", [span, wt / 2.0, hh], [0.0, hw + wt / 2.0, 0.0]),
        ("socket_wall_s", [span, wt / 2.0, hh], [0.0, -(hw + wt / 2.0), 0.0]),
        ("socket_wall_e", [wt / 2.0, hw, hh], [hw + wt / 2.0, 0.0, 0.0]),
        ("socket_wall_w", [wt / 2.0, hw, hh], [-(hw + wt / 2.0), 0.0, 0.0]),
    ]
    for name, size, pos in wall_specs:
        g = socket_body.add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = size
        g.pos = pos
        g.rgba = [0.18, 0.36, 0.65, 1.0]

    socket_site = socket_body.add_site()
    socket_site.name = "socket_bottom"
    socket_site.pos = [0.0, 0.0, -hh + 0.005]
    socket_site.size = [0.006, 0.006, 0.006]
    socket_site.rgba = [0.1, 0.8, 0.3, 0.8]

    # Graspable peg with freejoint
    peg_body = world.add_body()
    peg_body.name = "peg"
    peg_body.pos = list(peg_pos)
    peg_joint = peg_body.add_freejoint()
    peg_joint.name = "peg_freejoint"

    peg_geom = peg_body.add_geom()
    peg_geom.name = "peg_shaft"
    peg_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    peg_geom.size = [0.012, 0.045, 0.0]
    peg_geom.mass = 0.08
    peg_geom.rgba = [0.85, 0.35, 0.18, 1.0]
    # High sliding/torsional friction so the parallel jaws can hold the shaft
    # without slipping; condim=4 enables the torsional friction term.
    peg_geom.friction = [1.0, 0.05, 0.001]
    peg_geom.condim = 4
    peg_geom.solref = [0.01, 1.0]

    peg_tip = peg_body.add_site()
    peg_tip.name = "peg_tip"
    peg_tip.pos = [0.0, 0.0, -0.045]
    peg_tip.size = [0.005, 0.005, 0.005]
    peg_tip.rgba = [1.0, 0.8, 0.1, 0.9]

    # Overhead camera for future VLA / vision policy extension
    cam = world.add_camera()
    cam.name = "overhead_cam"
    cam.pos = [0.35, 0.0, 0.75]
    # MjSpec cameras expose a quaternion only (no xyaxes on MjsCamera).
    # 180 deg yaw about Z (x->-x, y->-y), looking straight down.
    cam.quat = [0.0, 0.0, 0.0, 1.0]

    return spec


def load_peg_insert_model(**kwargs) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile MjSpec into (MjModel, MjData) ready for simulation."""
    spec = build_peg_insert_spec(**kwargs)
    model = spec.compile()

    # The IK oracle relies on these menagerie-provided assets.
    for obj, name in (
        (mujoco.mjtObj.mjOBJ_SITE, "grasp_site"),
        (mujoco.mjtObj.mjOBJ_KEY, "home"),
    ):
        if mujoco.mj_name2id(model, obj, name) < 0:
            raise RuntimeError(
                f"i2rt_yam model is missing required {obj.name} {name!r}"
            )

    # The menagerie finger servo (kp=100) barely holds the 80 g peg: the jaw
    # normal force stays below ~1 N, so the peg slips out during transport.
    # Stiffen it so the jaws can carry the peg without sliding.
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    model.actuator_gainprm[gripper_id, 0] = GRIPPER_KP
    model.actuator_biasprm[gripper_id, 1] = -GRIPPER_KP

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data

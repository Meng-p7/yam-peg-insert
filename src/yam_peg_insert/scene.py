"""Programmatic MuJoCo scene builder using mujoco_menagerie + MjSpec."""

from __future__ import annotations

import mujoco
import mujoco_menagerie as mm


def build_peg_insert_spec(
    socket_pos: tuple[float, float, float] = (0.38, 0.0, 0.02),
    peg_pos: tuple[float, float, float] = (0.28, -0.14, 0.05),
    hole_half_width: float = 0.018,
    wall_thickness: float = 0.015,
    wall_height: float = 0.04,
) -> mujoco.MjSpec:
    """Load i2rt_yam from MuJoCo Menagerie and attach table, socket, and peg."""
    spec = mm.get("i2rt_yam").spec()

    # Ensure a ground plane and overhead lighting exist
    world = spec.worldbody

    # Simple workbench surface slightly below z=0 or at z=0
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
    peg_geom.friction = [0.9, 0.005, 0.0001]

    peg_tip = peg_body.add_site()
    peg_tip.name = "peg_tip"
    peg_tip.pos = [0.0, 0.0, -0.045]
    peg_tip.size = [0.005, 0.005, 0.005]
    peg_tip.rgba = [1.0, 0.8, 0.1, 0.9]

    # Overhead camera for future VLA / vision policy extension
    cam = world.add_camera()
    cam.name = "overhead_cam"
    cam.pos = [0.35, 0.0, 0.75]
    cam.xyaxes = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0]

    return spec


def load_peg_insert_model(**kwargs) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile MjSpec into (MjModel, MjData) ready for simulation."""
    spec = build_peg_insert_spec(**kwargs)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data

"""Differential IK (Jacobian-pseudoinverse) controller demo on AMDGPU.

Sealant-dispensing application: a UR5e (or Panda) traces a circular
seam around a hatch lid on a metal panel and lays down a continuous
bead of sealant — the kind of pass you'd see on an aircraft access
panel, an automotive engine cover, or a pressure-vessel hatch.

Scene anatomy
-------------
- ``fuselage_panel``  : large aluminium panel (the "skin" the hatch
                       is bolted into).
- ``hatch_lid``       : a slightly raised circular lid (Cylinder).
- ``bead_markers``    : 72 free-floating spheres pre-spawned below
                       the floor and revealed one-by-one as the
                       robot's tool tip passes over them, so the
                       sealant bead actually grows during the
                       first orbit.
- ``nozzle``          : a thin brass-coloured cylinder kinematically
                       teleported to the robot's tool tip every
                       step — looks like a sealant-gun nozzle.

Controller
----------
Damped-least-squares pseudoinverse IK on the 6x6 Jacobian. Gravity
and collisions are disabled so the demo isolates the controller's
tracking behaviour from physics interactions.

The robot is *pre-positioned* at the trajectory start with a single
inverse_kinematics() call before the tracking loop begins, so the
opening frame is calm instead of the arm whipping in from rest.

CLI mirrors grasp_bottle / suction_cup:
    --record / -o / --res / --robot / --orbits

Output mp4 lands in `examples/videos/` by default.
"""

import argparse
import os

import numpy as np

import genesis as gs


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "videos"))


CONFIG = {
    "ur5e":  {"mjcf_file": "xml/universal_robots_ur5e/ur5e.xml",  "end_effector_link": "ee_virtual_link"},
    "panda": {"mjcf_file": "xml/franka_emika_panda/panda.xml",   "end_effector_link": "left_finger"},
}

# Trajectory parameters: target orbits a small circle in the X-Y plane
# at this fixed centre/altitude.
TRAJ_CENTER = np.array([0.5, 0.0, 0.50])
TRAJ_RADIUS = 0.10
TRAJ_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # tool pointing -z

# --- workpiece geometry (all metres) ----------------------------------
# Big aluminium panel under the seam (the "fuselage skin").
PANEL_SIZE = (0.50, 0.40, 0.06)
PANEL_CENTER = (TRAJ_CENTER[0], TRAJ_CENTER[1], 0.42)
PANEL_TOP_Z = PANEL_CENTER[2] + PANEL_SIZE[2] / 2  # = 0.45

# Raised circular hatch lid sitting on top of the panel.
HATCH_RADIUS = 0.085
HATCH_HEIGHT = 0.025
HATCH_CENTER = (TRAJ_CENTER[0], TRAJ_CENTER[1], PANEL_TOP_Z + HATCH_HEIGHT / 2)

# Sealant bead just outside the hatch perimeter, sitting on the panel.
# 72 spheres on a 0.10 m circle → ~8.7 mm spacing; render radius 10 mm
# so the spheres overlap into a continuous bead instead of looking like
# separate dots.
BEAD_RADIUS_VIS = 0.010
BEAD_Z = PANEL_TOP_Z + BEAD_RADIUS_VIS         # so beads rest on panel top
N_BEAD_MARKERS = 72
HIDDEN_Z = -10.0                               # parking spot below floor

# Tool nozzle attached to the EE — its bottom tip should sit ~1 mm
# above the bead z so it reads as "actively dispensing".
NOZZLE_HEIGHT = 0.045
NOZZLE_RADIUS = 0.006

SIM_DT_DEFAULT = 0.01
DAMPING = 1e-4

# Original demo's angular speed: theta = i / 360 * pi  (i=step)
# → period (one full orbit) = 720 steps. We keep that and let the
# user choose how many orbits to render.
STEPS_PER_ORBIT = 720


def _trajectory_point(theta):
    """Position of the IK target at parameter ``theta`` (radians)."""
    return TRAJ_CENTER + np.array(
        [np.cos(theta), np.sin(theta), 0.0]
    ) * TRAJ_RADIUS


def _bead_position(k):
    """Resting position of the k-th sealant bead marker on the panel."""
    theta_k = 2.0 * np.pi * k / N_BEAD_MARKERS
    return np.array([
        TRAJ_CENTER[0] + np.cos(theta_k) * TRAJ_RADIUS,
        TRAJ_CENTER[1] + np.sin(theta_k) * TRAJ_RADIUS,
        BEAD_Z,
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False,
                        help="Force the CPU backend instead of AMDGPU.")
    parser.add_argument("-r", "--robot", choices=list(CONFIG.keys()), default="ur5e",
                        help="Robot model to control.")
    parser.add_argument("--orbits", type=float, default=2.0,
                        help="Number of full orbits the target traces.")
    parser.add_argument("--record", action="store_true",
                        help="Record an offscreen camera to mp4.")
    parser.add_argument("-o", "--video", type=str, default=None,
                        help="Output video path (implies --record).")
    parser.add_argument("--res", type=int, nargs=2, default=(1280, 720),
                        help="Recording resolution (W H).")
    args = parser.parse_args()

    if args.video is not None:
        args.record = True
    if args.record and args.video is None:
        args.video = os.path.join(VIDEOS_DIR, f"diffik_controller_{args.robot}.mp4")
    if args.video is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)

    # ----- init -----
    if args.cpu:
        backend = gs.cpu
    else:
        backend = gs.amdgpu
    gs.init(backend=backend)

    # ----- scene -----
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.10, -0.70, 0.78),
            camera_lookat=(0.50, 0.0, 0.46),
            camera_fov=42,
            max_FPS=200,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=False,
            enable_collision=False,
            gravity=(0.0, 0.0, 0.0),
            dt=SIM_DT_DEFAULT,
        ),
        show_viewer=args.vis,
        show_FPS=False,
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(color=(0.55, 0.55, 0.55, 1.0)),
    )

    robot_config = CONFIG[args.robot]
    robot = scene.add_entity(
        gs.morphs.MJCF(file=robot_config["mjcf_file"]),
    )

    # ----- workpiece: aluminium "fuselage" panel -----
    fuselage_panel = scene.add_entity(
        morph=gs.morphs.Box(
            size=PANEL_SIZE,
            pos=PANEL_CENTER,
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Rough(color=(0.62, 0.64, 0.68, 1.0)),
    )

    # Raised circular hatch lid bolted into the panel
    hatch_lid = scene.add_entity(
        morph=gs.morphs.Cylinder(
            radius=HATCH_RADIUS,
            height=HATCH_HEIGHT,
            pos=HATCH_CENTER,
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Rough(color=(0.18, 0.20, 0.26, 1.0)),
    )

    # ----- sealant bead: free-floating spheres parked below the floor.
    # We'll teleport each one up to its panel-top position the moment
    # the robot's tool tip passes over its angular slot — i.e. the bead
    # actually grows during the first orbit instead of being pre-laid.
    bead_markers = []
    for k in range(N_BEAD_MARKERS):
        bead = scene.add_entity(
            morph=gs.morphs.Sphere(
                radius=BEAD_RADIUS_VIS,
                pos=(TRAJ_CENTER[0], TRAJ_CENTER[1], HIDDEN_Z),
                collision=False,
            ),
            surface=gs.surfaces.Plastic(color=(0.97, 0.42, 0.06, 1.0)),
        )
        bead_markers.append(bead)

    # ----- nozzle: a thin brass cylinder kinematically pinned to the EE.
    # It's a free body (collision off, gravity is already off scene-wide)
    # so we can `set_pos` / `set_quat` it every step.
    nozzle = scene.add_entity(
        morph=gs.morphs.Cylinder(
            radius=NOZZLE_RADIUS,
            height=NOZZLE_HEIGHT,
            pos=(TRAJ_CENTER[0], TRAJ_CENTER[1], TRAJ_CENTER[2] - NOZZLE_HEIGHT / 2),
            collision=False,
        ),
        surface=gs.surfaces.Plastic(color=(0.85, 0.65, 0.20, 1.0)),
    )

    # Tiny axis marker overlaid on the IK target — keeps the controller
    # signal visible without being the dominant visual.
    target_entity = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.05,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=(1.0, 0.5, 0.5, 1.0)),
    )

    cam = None
    if args.record:
        cam = scene.add_camera(
            res=tuple(args.res),
            pos=(1.10, -0.70, 0.78),
            lookat=(0.50, 0.0, 0.46),
            fov=42,
            GUI=False,
        )

    scene.build()

    end_effector_link = robot_config["end_effector_link"]
    ee_link = robot.get_link(end_effector_link)

    diag = DAMPING * np.eye(6)
    n_steps = int(round(args.orbits * STEPS_PER_ORBIT))

    # Pre-position the robot at the trajectory start so the diffik loop
    # opens with ~zero error; otherwise the very first dq is huge and
    # the arm whips in from its rest pose.
    start_target = _trajectory_point(0.0)
    qpos_start = robot.inverse_kinematics(
        link=ee_link,
        pos=start_target,
        quat=TRAJ_QUAT,
    )
    robot.set_qpos(qpos_start)
    robot.control_dofs_position(qpos_start.cpu().numpy())
    target_entity.set_qpos(np.concatenate([start_target, TRAJ_QUAT]))

    # Pin the nozzle to the EE before the very first frame so it
    # appears already attached in the opening still.
    def _pin_nozzle_to_ee():
        ee_pos_now = ee_link.get_pos().cpu().numpy()
        nozzle.set_pos(np.array([
            float(ee_pos_now[0]),
            float(ee_pos_now[1]),
            float(ee_pos_now[2]) - NOZZLE_HEIGHT / 2,
        ]))
        try:
            nozzle.set_dofs_velocity([0.0] * 6)
        except Exception:
            pass

    _pin_nozzle_to_ee()
    scene.step()
    _pin_nozzle_to_ee()

    if cam is not None:
        cam.start_recording()

    try:
        # Brief stationary opening beat so the still frame at t=0 is calm
        for _ in range(20):
            _pin_nozzle_to_ee()
            scene.step()
            if cam is not None:
                cam.render()

        n_revealed = 0  # bead markers already lifted into place
        for i in range(n_steps):
            theta = i / 360.0 * np.pi
            target_pos = _trajectory_point(theta)

            # Progressively lay down the sealant bead during orbit 1.
            # Each marker k corresponds to angular slot 2π·k/N in the
            # CURRENT orbit; we reveal the marker once the tool tip has
            # passed over its slot.
            if i < STEPS_PER_ORBIT:
                deposited_angle = theta % (2.0 * np.pi)
            else:
                deposited_angle = 2.0 * np.pi
            target_revealed = min(
                N_BEAD_MARKERS,
                int(np.ceil(deposited_angle / (2.0 * np.pi) * N_BEAD_MARKERS)),
            )
            while n_revealed < target_revealed:
                bead_markers[n_revealed].set_pos(_bead_position(n_revealed))
                try:
                    bead_markers[n_revealed].set_dofs_velocity([0.0] * 6)
                except Exception:
                    pass
                n_revealed += 1

            target_entity.set_qpos(np.concatenate([target_pos, TRAJ_QUAT]))

            error_pos = target_pos - ee_link.get_pos().cpu().numpy()
            ee_quat = ee_link.get_quat().cpu().numpy()
            error_quat = gs.transform_quat_by_quat(gs.inv_quat(ee_quat), TRAJ_QUAT)
            error_rotvec = gs.quat_to_rotvec(error_quat)
            error = np.concatenate([error_pos, error_rotvec])

            jac = robot.get_jacobian(link=ee_link).cpu().numpy()
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)
            q = robot.get_qpos().cpu().numpy() + dq

            robot.control_dofs_position(q)
            _pin_nozzle_to_ee()
            scene.step()
            if cam is not None:
                cam.render()
    finally:
        if cam is not None:
            fps = max(10, int(round(1.0 / SIM_DT_DEFAULT)))
            cam.stop_recording(save_to_filename=args.video, fps=fps)
            print(f"Saved video to {args.video}")


if __name__ == "__main__":
    main()

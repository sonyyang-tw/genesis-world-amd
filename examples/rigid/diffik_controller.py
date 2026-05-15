"""Differential IK (Jacobian-pseudoinverse) controller demo on AMDGPU.

A UR5e or Panda arm tracks a circular trajectory above a static
workpiece — think welding a circular seam on a metal flange, or
applying a bead of sealant around a circular feature on a panel.

The controller is damped-least-squares pseudoinverse IK applied to
the 6x6 Jacobian; gravity and collisions are disabled so the demo
isolates the controller behaviour.

The robot is *pre-positioned* at the trajectory start with a single
inverse_kinematics() call before the tracking loop begins, so the
opening frame is stable instead of the arm whipping in from its
default rest pose.

The static "weld bead" markers (small bright spheres laid along the
seam) make the industrial intent of the demo immediately readable.

CLI matches grasp_bottle / suction_cup conventions:
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
TRAJ_CENTER = np.array([0.5, 0.0, 0.5])
TRAJ_RADIUS = 0.10
TRAJ_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # tool pointing -z

# Workpiece sits just under the trajectory plane so the seam appears
# to sit on the part's top face.
WORKPIECE_SIZE = (0.26, 0.20, 0.04)
WORKPIECE_CENTER = (TRAJ_CENTER[0], TRAJ_CENTER[1], TRAJ_CENTER[2] - 0.05)

# Number of "weld-bead" markers laid along the trajectory.
N_BEAD_MARKERS = 48
BEAD_RADIUS = 0.006

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
            camera_pos=(1.1, -1.4, 0.95),
            camera_lookat=(0.4, 0.0, 0.5),
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

    # Static metal-look workpiece — the "part" the robot is welding /
    # sealing / inspecting around.
    workpiece = scene.add_entity(
        morph=gs.morphs.Box(
            size=WORKPIECE_SIZE,
            pos=WORKPIECE_CENTER,
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Rough(color=(0.50, 0.52, 0.58, 1.0)),
    )

    # Pre-laid "weld bead" markers along the seam — gives an obvious
    # visual cue of what trajectory the robot is following and what an
    # industrial application would look like (welding, sealant
    # dispensing, glue laying, edge finishing, ...).
    for k in range(N_BEAD_MARKERS):
        theta_k = 2.0 * np.pi * k / N_BEAD_MARKERS
        scene.add_entity(
            morph=gs.morphs.Sphere(
                radius=BEAD_RADIUS,
                pos=tuple(_trajectory_point(theta_k)),
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Plastic(color=(1.0, 0.30, 0.05, 1.0)),
        )

    # Visual axis marker that we'll teleport to the IK target each step.
    # Marked collision=False so the engine skips convex-decomposition on
    # the axis arrow mesh (it crashed during decomp on AMDGPU and the
    # marker is purely visual anyway).
    target_entity = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.08,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=(1.0, 0.5, 0.5, 1.0)),
    )

    cam = None
    if args.record:
        cam = scene.add_camera(
            res=tuple(args.res),
            pos=(1.1, -1.4, 0.95),
            lookat=(0.4, 0.0, 0.5),
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
    scene.step()

    if cam is not None:
        cam.start_recording()

    try:
        # Brief stationary opening beat so the still frame at t=0 is calm
        for _ in range(20):
            scene.step()
            if cam is not None:
                cam.render()

        for i in range(n_steps):
            theta = i / 360.0 * np.pi
            target_pos = _trajectory_point(theta)

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

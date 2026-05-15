"""Heterogeneous simulation demo on AMDGPU.

Four parallel environments each instantiate the SAME entity slot with
a DIFFERENT geometry variant — two box sizes and two sphere radii.
A Franka in every env runs the same scripted grasp & lift sequence,
showing the robot adapting to the per-env object shape.

Variant assignment (with `n_envs == n_variants == 4`):
    Env 0 -> Variant 0  (4 cm box)
    Env 1 -> Variant 1  (2 cm box)
    Env 2 -> Variant 2  (1.5 cm sphere)
    Env 3 -> Variant 3  (2.5 cm sphere)

CLI matches the other AMDGPU demos:
    --record / -o / --res / --n-envs
"""

import argparse
import os

import numpy as np

import genesis as gs


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "videos"))

SIM_DT = 0.01

# Phase lengths in sim steps. Lengthened from the original demo so the
# video shows each stage clearly.
N_HOLD = 100      # 1.0 s settling at pre-grasp pose
N_GRASP = 200     # 2.0 s closing the gripper
N_LIFT = 350      # 3.5 s lifting + holding aloft
N_FINAL = 80      # final hold so the lift is clearly visible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-n", "--n-envs", type=int, default=4,
                        help="Number of parallel environments (4 = one per variant).")
    parser.add_argument("--record", action="store_true",
                        help="Record an offscreen camera to mp4.")
    parser.add_argument("-o", "--video", type=str, default=None,
                        help="Output video path (implies --record).")
    parser.add_argument("--res", type=int, nargs=2, default=(1280, 720),
                        help="Recording resolution (W H).")
    parser.add_argument("--env-spacing", type=float, nargs=2, default=(1.0, 1.0),
                        help="(dx, dy) spacing between environments.")
    args = parser.parse_args()

    if args.video is not None:
        args.record = True
    if args.record and args.video is None:
        args.video = os.path.join(VIDEOS_DIR, "heterogeneous_grasp.mp4")
    if args.video is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)

    if args.n_envs <= 0:
        gs.raise_exception("This demo requires n_envs >= 1; the heterogeneous "
                           "morph list only takes effect on a batched scene.")

    # ----- init -----
    gs.init(backend=gs.amdgpu, precision="32")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.7, -1.7, 1.25),
            camera_lookat=(0.4, 0.0, 0.18),
            camera_fov=50,
            max_FPS=60,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(color=(0.55, 0.55, 0.55, 1.0)),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    morphs_heterogeneous = [
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)),  # Variant 0
        gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.65, 0.0, 0.02)),  # Variant 1
        gs.morphs.Sphere(radius=0.015, pos=(0.65, 0.0, 0.02)),          # Variant 2
        gs.morphs.Sphere(radius=0.025, pos=(0.65, 0.0, 0.02)),          # Variant 3
    ]
    grasping_object = scene.add_entity(
        morph=morphs_heterogeneous,
        # Heterogeneous entities share one surface, so use a bright,
        # high-contrast colour so the box / sphere / size variants pop
        # against the grey robots and floor in the recorded video.
        surface=gs.surfaces.Plastic(color=(1.0, 0.45, 0.05, 1.0)),
    )

    cam = None
    if args.record:
        cam = scene.add_camera(
            res=tuple(args.res),
            pos=(1.7, -1.7, 1.25),
            lookat=(0.4, 0.0, 0.18),
            fov=50,
            GUI=False,
        )

    scene.build(n_envs=args.n_envs, env_spacing=tuple(args.env_spacing))

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)

    l_qpos = [-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]
    franka.set_qpos(np.array([l_qpos] * args.n_envs))
    scene.step()

    aabb = grasping_object.get_AABB()
    mass = grasping_object.get_mass()
    print(f"heterogeneous AABB:\n{aabb}")
    print(f"heterogeneous mass: {mass}")

    end_effector = franka.get_link("hand")

    # Pre-grasp target — same for every env relative to its local frame
    qpos_pre = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([[0.65, 0.0, 0.135]] * args.n_envs),
        quat=np.array([[0.0, 1.0, 0.0, 0.0]] * args.n_envs),
    )
    franka.control_dofs_position(qpos_pre[..., :-2], motors_dof)

    if cam is not None:
        cam.start_recording()

    def _step_with_render():
        scene.step()
        if cam is not None:
            cam.render()

    try:
        # ----- 1) settle at pre-grasp pose -----
        for _ in range(N_HOLD):
            _step_with_render()

        # ----- 2) close the gripper to grasp the variant object -----
        finger_close = 0.0
        finger_targets = np.array([[finger_close, finger_close]] * args.n_envs)
        for _ in range(N_GRASP):
            franka.control_dofs_position(qpos_pre[..., :-2], motors_dof)
            franka.control_dofs_position(finger_targets, fingers_dof)
            _step_with_render()

        # ----- 3) lift the grasped object straight up -----
        qpos_lift = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([[0.65, 0.0, 0.30]] * args.n_envs),
            quat=np.array([[0.0, 1.0, 0.0, 0.0]] * args.n_envs),
        )
        for _ in range(N_LIFT):
            franka.control_dofs_position(qpos_lift[..., :-2], motors_dof)
            franka.control_dofs_position(finger_targets, fingers_dof)
            _step_with_render()

        # ----- 4) final hold so the still frame at the end is clean -----
        for _ in range(N_FINAL):
            franka.control_dofs_position(qpos_lift[..., :-2], motors_dof)
            franka.control_dofs_position(finger_targets, fingers_dof)
            _step_with_render()

    finally:
        if cam is not None:
            fps = max(10, int(round(1.0 / SIM_DT)))
            cam.stop_recording(save_to_filename=args.video, fps=fps)
            print(f"Saved video to {args.video}")


if __name__ == "__main__":
    main()

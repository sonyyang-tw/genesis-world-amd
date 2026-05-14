import argparse
import math

import numpy as np

import genesis as gs


def _interp_keyframes(keys, t):
    """Cosine-eased interpolation across (t, value) keyframes."""
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    for i in range(len(keys) - 1):
        t0, v0 = keys[i]
        t1, v1 = keys[i + 1]
        if t0 <= t <= t1:
            local = (t - t0) / max(1e-9, (t1 - t0))
            eased = 0.5 - 0.5 * math.cos(math.pi * local)
            return tuple(float(a + (b - a) * eased) for a, b in zip(v0, v1))
    return keys[-1][1]


def _drive_past_pose(step_i, total, side):
    """Cinematic camera trajectory: establish -> drive past +x side -> pull back."""
    t = min(1.0, step_i / max(1, total))
    far = max(6.0, side * 1.4)
    pos_keys = [
        (0.00, (far + 2.0, -far - 2.0, far)),
        (0.15, (far - 0.5, -side - 1.5, side)),
        (0.20, (far - 1.0, -side - 1.0, 1.6)),
        (0.75, (far - 1.0, side + 1.0, 1.6)),
        (0.85, (far + 1.0, side + 2.0, side)),
        (1.00, (far + 2.0, far + 2.0, far - 1.0)),
    ]
    look_keys = [
        (0.00, (0.3, 0.0, 0.5)),
        (0.15, (0.3, 0.0, 0.4)),
        (0.20, (0.3, -side + 1.0, 0.4)),
        (0.75, (0.3, side - 1.0, 0.4)),
        (0.85, (0.3, 0.0, 0.6)),
        (1.00, (0.3, 0.0, 0.6)),
    ]
    return _interp_keyframes(pos_keys, t), _interp_keyframes(look_keys, t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-n", "--n_envs", type=int, default=49)
    parser.add_argument("-r", "--record", action="store_true", help="Record an offscreen camera to mp4.")
    parser.add_argument("-o", "--video", type=str, default=None, help="Output video path (implies --record).")
    parser.add_argument("--res", type=int, nargs=2, default=(1920, 1080), help="Recording resolution (W H).")
    parser.add_argument(
        "--randomize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize per-env bottle xy so each Franka does a slightly different motion.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for bottle randomization.")
    parser.add_argument(
        "--static-cam",
        action="store_true",
        help="Disable the drive-past trajectory and use a fixed wide-angle camera (legacy).",
    )
    args = parser.parse_args()

    if args.video is not None:
        args.record = True
    if args.record and args.video is None:
        args.video = f"grasp_bottle_n{args.n_envs}.mp4"

    ########################## init ##########################
    gs.init(backend=gs.amdgpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )

    # When recording, render all parallel envs so the wide shot captures the grid.
    vis_options = None
    if args.record and args.n_envs > 0:
        vis_options = gs.options.VisOptions(rendered_envs_idx=list(range(args.n_envs)))

    scene = gs.Scene(
        viewer_options=viewer_options,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
        ),
        vis_options=vis_options if vis_options is not None else gs.options.VisOptions(),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    bottle = scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.URDF(
            file="urdf/3763/mobility_vhacd.urdf",
            scale=0.09,
            pos=(0.65, 0.0, 0.036),
            euler=(0, 90, 0),
        ),
        # visualize_contact=True,
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    ########################## camera (for recording) ##########################
    cam = None
    # Half-extent of the env grid in world units (env_spacing=1, so side ≈ sqrt(n_envs)/2).
    n_per_row = max(1, math.ceil(math.sqrt(max(args.n_envs, 1))))
    grid_half = max(1.0, (n_per_row - 1) / 2.0)
    if args.record:
        radius = max(4.0, grid_half * 2.0)
        cam = scene.add_camera(
            res=tuple(args.res),
            pos=(radius, -radius, max(4.0, grid_half * 1.5)),
            lookat=(0.0, 0.0, 0.3),
            fov=45,
            GUI=False,
        )

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1, 1))

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # Optional: set control gains
    if args.n_envs == 0:
        franka.set_qpos(np.array([1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]))
    else:
        franka.set_qpos(np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * args.n_envs))
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    end_effector = franka.get_link("hand")

    ########################## per-env bottle randomization ##########################
    # Each env's bottle gets a small (Δx, Δy) jitter so every Franka has to
    # reach a slightly different target. The IK targets below are offset by
    # the same amount, so the resulting motions are all unique.
    n_batched = max(1, args.n_envs)
    bottle_offsets = np.zeros((n_batched, 3), dtype=gs.np_float)
    if args.randomize and args.n_envs > 0:
        rng = np.random.default_rng(args.seed)
        bottle_offsets[:, 0] = rng.uniform(-0.04, 0.04, args.n_envs)
        bottle_offsets[:, 1] = rng.uniform(-0.05, 0.05, args.n_envs)

        base_pos = np.array([0.65, 0.0, 0.036], dtype=gs.np_float)
        bottle.set_pos(base_pos[None] + bottle_offsets)

    def with_offset(base_xyz):
        base = np.asarray(base_xyz, dtype=gs.np_float)
        if args.n_envs == 0:
            return base
        return np.broadcast_to(base, (args.n_envs, 3)) + bottle_offsets[: args.n_envs]

    grasp_quat = (
        np.array([0, 1, 0, 0])
        if args.n_envs == 0
        else np.array([[0, 1, 0, 0]] * args.n_envs)
    )

    ########################## sim-step + camera helper ##########################
    step_counter = [0]
    # Rough length of the scripted sequence; used to time the camera trajectory.
    # plan_path waypoints are unknown until plan_path() returns, so we add it
    # in once we have it.
    cam_total = [30 + 100 + 100 + 1000]  # 1230 fixed + plan_path
    use_dynamic_cam = args.record and not args.static_cam

    def step(n: int = 1):
        for _ in range(n):
            scene.step()
            if cam is not None:
                if use_dynamic_cam:
                    pos, lookat = _drive_past_pose(step_counter[0], cam_total[0], grid_half)
                    cam.set_pose(pos=pos, lookat=lookat)
                cam.render()
            step_counter[0] += 1

    if cam is not None:
        cam.start_recording()

    try:
        # move to pre-grasp pose
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=with_offset([0.65, 0.0, 0.25]),
            quat=grasp_quat,
        )
        qpos[..., -2:] = 0.04

        path = franka.plan_path(qpos)
        cam_total[0] += len(path)  # include plan_path length in the trajectory budget
        for waypoint in path:
            franka.control_dofs_position(waypoint)
            step()
        step(30)

        # reach
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=with_offset([0.65, 0.0, 0.142]),
            quat=grasp_quat,
        )
        franka.control_dofs_position(qpos[..., :-2], motors_dof)
        step(100)

        # grasp
        franka.control_dofs_position(qpos[..., :-2], motors_dof)
        franka.control_dofs_position(
            np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
        )  # you can use position control
        step(100)

        # lift
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=with_offset([0.65, 0.0, 0.3]),
            quat=grasp_quat,
        )
        franka.control_dofs_position(qpos[..., :-2], motors_dof)
        franka.control_dofs_force(
            np.array([-20, -20]) if args.n_envs == 0 else np.array([[-20, -20]] * args.n_envs), fingers_dof
        )  # can also use force control
        step(1000)
    except KeyboardInterrupt:
        print("Interrupted, finalizing recording...")
    finally:
        if cam is not None:
            fps = int(round(1.0 / 0.01))
            cam.stop_recording(save_to_filename=args.video, fps=fps)
            print(f"Saved video to {args.video}")


if __name__ == "__main__":
    main()

"""
# headless recording (49 parallel envs, 1920x1080 mp4)
python examples/rigid/grasp_bottle.py --record

# custom env count and resolution
python examples/rigid/grasp_bottle.py -n 16 --record --res 1280 720 -o grasp_16.mp4

# on-screen viewer (original behaviour)
python examples/rigid/grasp_bottle.py -v
"""

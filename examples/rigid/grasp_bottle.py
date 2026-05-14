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


def _cam_start_x(side):
    """X position the camera occupies at t=0 of the drive-past.

    Chosen to roughly match where the *previous* version of this script had
    the camera at second 8 — i.e. already past the centre of the grid, with
    the row of robots filling the right half of the frame. From here the
    camera continues outward (+x) slowly.
    """
    return side * 0.6


def _camera_pose(step_i, cruise_steps, approach_steps, side):
    """Two-phase cinematic camera path:

    1. Slow horizontal drive-past from the side-on "old sec 8" position out
       past the +x edge of the grid (camera up=+Z, no pitch, no roll).
    2. Smooth arc up and inward to a top-down hero shot of the centre env.

    `cruise_steps` is the number of sim steps for phase 1, `approach_steps`
    for phase 2. After both phases the camera holds the overhead pose.
    """
    drive_y = -side - 0.6     # only ~0.6 m beyond the south row — close
    drive_z = 0.7             # ~at the top of the Franka body
    look_z = 0.7              # SAME as drive_z → horizontal (parallel to floor)

    x_start = _cam_start_x(side)        # ≈ old "second 8" position
    x_cruise_end = side + 1.0           # 1 m past the last column
    x_outro = x_cruise_end + 0.3        # tiny pull-back at the very end

    # End-of-cruise / start-of-approach pose (shared between phases).
    cruise_end_pos = (x_outro, drive_y - 0.2, drive_z + 0.1)
    cruise_end_look = (x_cruise_end, 0.0, look_z)

    # Final overhead hero pose: tight hover above the centre env's lifted
    # arm. Each Franka's gripper sits ~0.65 m in +x from its base so the
    # visual centre of the centre robot is at world (~0.3, 0, ~0.3) — point
    # the lookat there, with the camera shifted slightly south so we don't
    # gimbal-lock straight down.
    overhead_pos = (0.3, -0.5, 1.7)
    overhead_look = (0.3, 0.0, 0.3)

    if step_i < cruise_steps:
        t = step_i / max(1, cruise_steps)
        pos_keys = [
            (0.00, (x_start,      drive_y, drive_z)),
            (0.96, (x_cruise_end, drive_y, drive_z)),
            (1.00, cruise_end_pos),
        ]
        look_keys = [
            (0.00, (x_start,      0.0, look_z)),
            (0.96, (x_cruise_end, 0.0, look_z)),
            (1.00, cruise_end_look),
        ]
    else:
        s = min(1.0, (step_i - cruise_steps) / max(1, approach_steps))
        pos_keys = [
            (0.00, cruise_end_pos),
            (1.00, overhead_pos),
        ]
        look_keys = [
            (0.00, cruise_end_look),
            (1.00, overhead_look),
        ]
        t = s
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
    parser.add_argument(
        "--row-delay",
        type=float,
        default=0.6,
        help="Stagger row start times by this many seconds (each y-line of envs starts later).",
    )
    parser.add_argument(
        "--cam-duration",
        type=float,
        default=14.0,
        help="Drive-past duration in seconds (larger = slower camera).",
    )
    parser.add_argument(
        "--approach-duration",
        type=float,
        default=5.0,
        help="Time (seconds) to arc from drive-past end into the overhead hero shot.",
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
    # Analytical infinite plane with a flat mid-gray surface (no diffuse
    # texture, so the floor reads as truly uniform out to the horizon
    # instead of a finite textured rug + a stripe of default plane).
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(color=(0.55, 0.55, 0.55, 1.0)),
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
        # Initial camera = first keyframe of the drive-past (so there is no
        # establishing-to-cruise cut). Matches what the previous build of
        # this script showed at second 8.
        x_start = _cam_start_x(grid_half)
        cam_init_pos = (x_start, -grid_half - 0.6, 0.7)
        cam_init_look = (x_start, 0.0, 0.7)
        cam = scene.add_camera(
            res=tuple(args.res),
            pos=cam_init_pos,
            lookat=cam_init_look,
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
    DT = 0.01
    ROW_DELAY = max(1, int(round(args.row_delay / DT)))
    CAM_DUR = max(1, int(round(args.cam_duration / DT)))
    APPROACH_DUR = max(0, int(round(args.approach_duration / DT)))
    CAM_TOTAL = CAM_DUR + APPROACH_DUR
    use_dynamic_cam = args.record and not args.static_cam
    step_counter = [0]

    def render_cam_only():
        """Advance the recording camera by one frame (with optional drive-past)."""
        if cam is None:
            return
        if use_dynamic_cam:
            cs = min(step_counter[0], CAM_TOTAL)
            pos, lookat = _camera_pose(cs, CAM_DUR, APPROACH_DUR, grid_half)
            # Always re-pin world up to +Z so the camera never inherits roll
            # from the previous transform (set_pose without `up` falls back
            # to the stored Y-axis, which can be slightly oblique).
            cam.set_pose(pos=pos, lookat=lookat, up=(0.0, 0.0, 1.0))
        cam.render()

    def tick_one():
        scene.step()
        render_cam_only()
        step_counter[0] += 1

    if cam is not None:
        cam.start_recording()

    try:
        if args.n_envs == 0:
            # ----- single-env path: keep the original scripted sequence -----
            qpos = franka.inverse_kinematics(
                link=end_effector, pos=with_offset([0.65, 0.0, 0.25]), quat=grasp_quat,
            )
            qpos[..., -2:] = 0.04
            for waypoint in franka.plan_path(qpos):
                franka.control_dofs_position(waypoint)
                tick_one()
            for _ in range(30):
                tick_one()
            qpos = franka.inverse_kinematics(
                link=end_effector, pos=with_offset([0.65, 0.0, 0.142]), quat=grasp_quat,
            )
            franka.control_dofs_position(qpos[..., :-2], motors_dof)
            for _ in range(100):
                tick_one()
            franka.control_dofs_position(qpos[..., :-2], motors_dof)
            franka.control_dofs_position(np.array([0, 0]), fingers_dof)
            for _ in range(100):
                tick_one()
            qpos = franka.inverse_kinematics(
                link=end_effector, pos=with_offset([0.65, 0.0, 0.3]), quat=grasp_quat,
            )
            franka.control_dofs_position(qpos[..., :-2], motors_dof)
            franka.control_dofs_force(np.array([-20, -20]), fingers_dof)
            for _ in range(1000):
                tick_one()
        else:
            # ----- multi-env path: pre-compute IK + plan_path, then dispatch -----
            # one row at a time so the wave staggers across the grid.
            qpos_pre = franka.inverse_kinematics(
                link=end_effector, pos=with_offset([0.65, 0.0, 0.25]), quat=grasp_quat,
            )
            qpos_pre[..., -2:] = 0.04
            qpos_reach = franka.inverse_kinematics(
                link=end_effector, pos=with_offset([0.65, 0.0, 0.142]), quat=grasp_quat,
            )
            qpos_lift = franka.inverse_kinematics(
                link=end_effector, pos=with_offset([0.65, 0.0, 0.3]), quat=grasp_quat,
            )
            plan_waypoints = list(franka.plan_path(qpos_pre))

            PLAN_LEN = len(plan_waypoints)
            DWELL, REACH, GRASP, LIFT = 30, 100, 100, 1000
            phase_total = PLAN_LEN + DWELL + REACH + GRASP + LIFT

            # With env_spacing=(1, 1) and the row-major layout
            #   offset_x = (env_idx // n_per_row) * sx
            #   offset_y = (env_idx % n_per_row) * sy
            # envs sharing the same `env_idx % n_per_row` lie on the same y-line,
            # i.e. they look like one horizontal row from the camera's POV.
            env_indices = np.arange(args.n_envs)
            n_rows = n_per_row
            envs_per_row = [
                env_indices[(env_indices % n_per_row) == r] for r in range(n_rows)
            ]

            last_row_offset = (n_rows - 1) * ROW_DELAY
            motion_end = phase_total + last_row_offset
            cam_end = CAM_TOTAL if use_dynamic_cam else 0
            TOTAL = max(motion_end, cam_end)

            zeros_2 = np.zeros((1, 2), dtype=gs.np_float)
            lift_force = np.full((1, 2), -20.0, dtype=gs.np_float)

            for t in range(TOTAL):
                for r in range(n_rows):
                    eidx = envs_per_row[r]
                    if eidx.size == 0:
                        continue
                    t_local = t - r * ROW_DELAY
                    if t_local < 0:
                        continue  # row not active yet — stays at initial qpos
                    if t_local < PLAN_LEN:
                        wp = plan_waypoints[t_local]
                        franka.control_dofs_position(wp[eidx], envs_idx=eidx)
                    elif t_local < PLAN_LEN + DWELL:
                        franka.control_dofs_position(qpos_pre[eidx], envs_idx=eidx)
                    elif t_local < PLAN_LEN + DWELL + REACH:
                        franka.control_dofs_position(
                            qpos_reach[eidx, :-2], motors_dof, envs_idx=eidx,
                        )
                    elif t_local < PLAN_LEN + DWELL + REACH + GRASP:
                        franka.control_dofs_position(
                            qpos_reach[eidx, :-2], motors_dof, envs_idx=eidx,
                        )
                        franka.control_dofs_position(
                            np.broadcast_to(zeros_2, (eidx.size, 2)),
                            fingers_dof, envs_idx=eidx,
                        )
                    else:  # in lift or hold-after-lift
                        franka.control_dofs_position(
                            qpos_lift[eidx, :-2], motors_dof, envs_idx=eidx,
                        )
                        franka.control_dofs_force(
                            np.broadcast_to(lift_force, (eidx.size, 2)),
                            fingers_dof, envs_idx=eidx,
                        )
                tick_one()
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

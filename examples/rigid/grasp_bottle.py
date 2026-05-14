import argparse
import math

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-n", "--n_envs", type=int, default=49)
    parser.add_argument("-r", "--record", action="store_true", help="Record an offscreen camera to mp4.")
    parser.add_argument("-o", "--video", type=str, default=None, help="Output video path (implies --record).")
    parser.add_argument("--res", type=int, nargs=2, default=(1920, 1080), help="Recording resolution (W H).")
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
    if args.record:
        # Place camera high & oblique to cover the n_envs grid.
        # Grid is roughly sqrt(n_envs) x sqrt(n_envs) at env_spacing=(1,1).
        side = max(1, math.ceil(math.sqrt(max(args.n_envs, 1))))
        radius = max(4.0, side * 1.3)
        cam = scene.add_camera(
            res=tuple(args.res),
            pos=(radius, -radius, max(4.0, side * 0.9)),
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

    def step(n: int = 1):
        for _ in range(n):
            scene.step()
            if cam is not None:
                cam.render()

    if cam is not None:
        cam.start_recording()

    try:
        # move to pre-grasp pose
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([0.65, 0.0, 0.25]) if args.n_envs == 0 else np.array([[0.65, 0.0, 0.25]] * args.n_envs),
            quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
        )
        qpos[..., -2:] = 0.04

        path = franka.plan_path(qpos)
        for waypoint in path:
            franka.control_dofs_position(waypoint)
            step()
        step(30)

        # reach
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([0.65, 0.0, 0.142]) if args.n_envs == 0 else np.array([[0.65, 0.0, 0.142]] * args.n_envs),
            quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
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
            pos=np.array([0.65, 0.0, 0.3]) if args.n_envs == 0 else np.array([[0.65, 0.0, 0.3]] * args.n_envs),
            quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
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

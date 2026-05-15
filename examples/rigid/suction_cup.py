"""Industrial suction-cup pick & place over a conveyor belt feeder.

A single Franka stands beside a conveyor belt that feeds in colour-coded
cubes. The "suction" end-effector is modelled with a rigid weld
constraint between the gripper's `hand` link and the target cube. The
robot picks each cube off the belt and drops it into a colour-matched
bin on the opposite side, sorting the stream into two piles.

Belt motion is implemented kinematically: as long as a cube is in the
`on_belt` state we move it in -x at constant speed each step. Once the
robot welds onto a cube it follows the gripper (state = `held`). After
release over a bin (state = `dropped`) it falls under gravity. The
belt freezes for the duration of a pick → place → return cycle so the
spacing between cubes stays constant — i.e. the production line runs
stop-and-go like a real station-based assembly line.

AMDGPU backend, headless recording lands in `examples/videos/`.
"""

import argparse
import os

import numpy as np

import genesis as gs


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "videos"))


# ---------------------------------------------------------------------------
# Scene layout (all metres). The robot sits at world origin facing +x.
# ---------------------------------------------------------------------------
SIM_DT = 0.01

PICK_X = 0.55                            # robot picks at this x
BELT_Y_CENTER = 0.00
BELT_HALF_W = 0.10                       # 20 cm wide belt
BELT_THICK = 0.05
BELT_TOP_Z = 0.05                        # top surface height

PICK_Y = BELT_Y_CENTER
PICK_Z_PRE = 0.32                        # safe transit height
PICK_Z_GRAB = 0.18                       # hand height where the weld is added

CUBE_SIZE = 0.04
CUBE_CENTER_Z = BELT_TOP_Z + CUBE_SIZE / 2  # 0.07

# Belt extent is computed at runtime from the cube count so every cube
# starts physically resting on the belt — if a cube spawned past the
# +x edge it would fall to the floor under gravity, and the next
# kinematic set_pos would teleport it 5 cm vertically which crashes
# the rigid constraint solver with NaN.
CUBE_SPACING = 0.35                      # m between consecutive cubes
CUBE_LEAD_OFFSET = 0.30                  # first cube is this far +x of PICK_X
BELT_PAD_FRONT = 0.40                    # belt extends this far past last cube
BELT_PAD_BACK = 0.40                     # ...and this far back from PICK_X

BIN_X = 0.32
BIN_Y_OFFSET = BELT_HALF_W + 0.18        # bins sit 18 cm off each side
BIN_SIZE = 0.18                          # outer bin footprint (square)
BIN_WALL_THICK = 0.012
BIN_WALL_HEIGHT = 0.05                   # short walls so cubes don't roll out
BIN_BASE_DROP_Z = 0.18                   # hand height for the *first* cube
BIN_STACK_DH = CUBE_SIZE                 # raise hand by one cube per stacked item

BIN_POS = {
    "red":  (BIN_X, BELT_Y_CENTER - BIN_Y_OFFSET, 0.02),
    "blue": (BIN_X, BELT_Y_CENTER + BIN_Y_OFFSET, 0.02),
}

GRAB_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # gripper pointing straight down

# Belt kinematics
DEFAULT_BELT_SPEED = 0.18                # m/s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cube_pos(cube_entity):
    """Read entity position back as a numpy (3,) array on the host."""
    p = cube_entity.get_pos()
    if hasattr(p, "cpu"):
        p = p.cpu().numpy()
    return np.asarray(p, dtype=np.float64).reshape(-1)[:3]


def _set_cube_pos(cube_entity, pos):
    cube_entity.set_pos(np.asarray(pos, dtype=np.float64))
    # Kill drift each step so the kinematic cube doesn't carry over
    # contact-induced velocity from previous frames.
    try:
        cube_entity.set_dofs_velocity([0.0] * 6)
    except Exception:
        pass


def _advance_belt(cubes, dt, speed):
    """Slide every `on_belt` cube by -speed*dt in x."""
    for cube in cubes:
        if cube["state"] != "on_belt":
            continue
        pos = _cube_pos(cube["entity"])
        new_x = float(pos[0]) - speed * dt
        _set_cube_pos(cube["entity"], (new_x, BELT_Y_CENTER, CUBE_CENTER_Z))


def _wait_for_cube(scene, cam, recording, cubes, target_idx, dt, speed, max_steps=4000):
    """Advance the belt until cubes[target_idx] reaches PICK_X."""
    cube = cubes[target_idx]
    for _ in range(max_steps):
        pos = _cube_pos(cube["entity"])
        if pos[0] <= PICK_X:
            # Snap exactly onto the pick point and freeze.
            _set_cube_pos(cube["entity"], (PICK_X, PICK_Y, CUBE_CENTER_Z))
            cube["state"] = "waiting_pick"
            return
        _advance_belt(cubes, dt, speed)
        scene.step()
        if recording:
            cam.render()


def _drive_qpos(franka, qpos_target, n_steps, scene, cam, recording, fingers_dof):
    """Hold a position target for n_steps, ticking sim + render each step."""
    finger_force = np.array([0.5, 0.5])
    for _ in range(n_steps):
        franka.control_dofs_position(qpos_target)
        franka.control_dofs_force(finger_force, fingers_dof)
        scene.step()
        if recording:
            cam.render()


def _execute_path(franka, fingers_dof, path, scene, cam, recording):
    finger_force = np.array([0.5, 0.5])
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        franka.control_dofs_force(finger_force, fingers_dof)
        scene.step()
        if recording:
            cam.render()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-r", "--record", action="store_true",
                        help="Record an offscreen camera to mp4.")
    parser.add_argument("-o", "--video", type=str, default=None,
                        help="Output video path (implies --record).")
    parser.add_argument("--res", type=int, nargs=2, default=(1280, 720),
                        help="Recording resolution (W H).")
    parser.add_argument("--n-cubes", type=int, default=4,
                        help="Number of cubes the belt feeds in.")
    parser.add_argument("--belt-speed", type=float, default=DEFAULT_BELT_SPEED,
                        help="Belt linear speed in m/s.")
    args = parser.parse_args()

    if args.video is not None:
        args.record = True
    if args.record and args.video is None:
        args.video = os.path.join(VIDEOS_DIR, "suction_cup_conveyor.mp4")
    if args.video is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)

    # ----- init -----
    gs.init(backend=gs.amdgpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=SIM_DT,
            # Two substeps stabilise the rigid solver under the
            # combination of (a) kinematic set_pos updates on the
            # on-belt cubes and (b) the weld constraint that appears
            # / disappears on every pick & place cycle.
            substeps=2,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
            # The default 50 iterations/50 ls_iterations sometimes fails
            # to converge cleanly when a weld constraint is added on
            # top of multiple kinematic-contact constraints, producing
            # 'nan' constraint forces. Bump both counts and add a few
            # noslip iterations (the docs recommend this specifically
            # for manipulation tasks).
            iterations=200,
            ls_iterations=100,
            noslip_iterations=5,
            integrator=gs.integrator.implicitfast,
            max_dynamic_constraints=16,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.4, -1.7, 1.25),
            camera_lookat=(0.95, 0.0, 0.15),
            camera_fov=45,
            max_FPS=60,
        ),
        show_viewer=args.vis,
    )

    # Belt extent — derived so every cube spawns *on* the belt surface.
    last_cube_x = PICK_X + CUBE_LEAD_OFFSET + (args.n_cubes - 1) * CUBE_SPACING
    belt_x_min = PICK_X - BELT_PAD_BACK
    belt_x_max = last_cube_x + BELT_PAD_FRONT
    belt_center_x = (belt_x_min + belt_x_max) / 2

    # ----- static scenery -----
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(color=(0.55, 0.55, 0.55, 1.0)),
    )

    # Conveyor belt (static box)
    belt = scene.add_entity(
        morph=gs.morphs.Box(
            size=(belt_x_max - belt_x_min, 2 * BELT_HALF_W, BELT_THICK),
            pos=(belt_center_x, BELT_Y_CENTER, BELT_TOP_Z - BELT_THICK / 2),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(color=(0.22, 0.22, 0.26, 1.0)),
    )

    # Side rails along the belt — slim raised strips so it reads as a
    # conveyor rather than just a flat platform.
    rail_thick = 0.02
    rail_height = 0.04
    for sign in (-1, +1):
        scene.add_entity(
            morph=gs.morphs.Box(
                size=(belt_x_max - belt_x_min, rail_thick, rail_height),
                pos=(
                    belt_center_x,
                    BELT_Y_CENTER + sign * (BELT_HALF_W + rail_thick / 2),
                    BELT_TOP_Z + rail_height / 2,
                ),
                fixed=True,
            ),
            surface=gs.surfaces.Rough(color=(0.85, 0.65, 0.10, 1.0)),
        )

    # Sort bins (coloured floor pads + four short walls off either side of
    # the belt). The walls stop a stacked cube from rolling out if it
    # doesn't land squarely on the cube already in the bin.
    for color, rgba in (("red", (0.75, 0.15, 0.15, 1.0)),
                       ("blue", (0.15, 0.25, 0.80, 1.0))):
        bx, by, _ = BIN_POS[color]
        # Bin floor
        scene.add_entity(
            morph=gs.morphs.Box(
                size=(BIN_SIZE, BIN_SIZE, 0.012),
                pos=(bx, by, 0.006),
                fixed=True,
            ),
            surface=gs.surfaces.Rough(color=rgba),
        )
        # Four perimeter walls
        wall_z = 0.012 + BIN_WALL_HEIGHT / 2
        wall_outer = (BIN_SIZE - BIN_WALL_THICK) / 2
        wall_specs = (
            # (size, pos)
            ((BIN_SIZE, BIN_WALL_THICK, BIN_WALL_HEIGHT), (bx, by - wall_outer, wall_z)),
            ((BIN_SIZE, BIN_WALL_THICK, BIN_WALL_HEIGHT), (bx, by + wall_outer, wall_z)),
            ((BIN_WALL_THICK, BIN_SIZE, BIN_WALL_HEIGHT), (bx - wall_outer, by, wall_z)),
            ((BIN_WALL_THICK, BIN_SIZE, BIN_WALL_HEIGHT), (bx + wall_outer, by, wall_z)),
        )
        for size, pos in wall_specs:
            scene.add_entity(
                morph=gs.morphs.Box(size=size, pos=pos, fixed=True),
                surface=gs.surfaces.Rough(color=rgba),
            )

    # ----- cubes (free rigid bodies, alternating colour) -----
    # Every cube spawns on the belt so it rests stably under gravity
    # until the kinematic belt motion slides it toward PICK_X.
    cubes = []
    colors = ["red", "blue"]
    for i in range(args.n_cubes):
        color = colors[i % 2]
        rgba = (0.92, 0.12, 0.12, 1.0) if color == "red" else (0.12, 0.28, 0.92, 1.0)
        x0 = PICK_X + CUBE_LEAD_OFFSET + i * CUBE_SPACING
        cube_ent = scene.add_entity(
            morph=gs.morphs.Box(
                size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                pos=(x0, BELT_Y_CENTER, CUBE_CENTER_Z),
            ),
            surface=gs.surfaces.Plastic(color=rgba),
        )
        cubes.append({
            "entity": cube_ent,
            "color": color,
            "init_x": x0,
            "state": "on_belt",
        })

    # ----- robot -----
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    # ----- camera (recording) -----
    cam = None
    if args.record:
        cam = scene.add_camera(
            res=tuple(args.res),
            pos=(2.4, -1.7, 1.25),
            lookat=(0.95, 0.0, 0.15),
            fov=45,
            GUI=False,
        )

    # ----- build -----
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")

    # Control gains (same as original suction_cup demo)
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    # ----- home pose: hovering above the pick zone, gripper open -----
    qpos_home = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([PICK_X, PICK_Y, PICK_Z_PRE]),
        quat=GRAB_QUAT,
    )
    qpos_home[-2:] = 0.04  # gripper open

    rigid_solver = scene.sim.rigid_solver
    link_franka = end_effector.idx

    if cam is not None:
        cam.start_recording()

    try:
        # Plan smoothly from default qpos to home pose
        home_path = franka.plan_path(qpos_goal=qpos_home, num_waypoints=120)
        _execute_path(franka, fingers_dof, home_path, scene, cam, args.record)

        # Settle a moment so the home pose is stable
        _drive_qpos(franka, qpos_home, 40, scene, cam, args.record, fingers_dof)

        # Per-bin stack counter so each subsequent cube is released
        # one cube-height higher — otherwise the second red cube would
        # be released *inside* the first red cube already sitting in
        # the bin, and the constraint solver would shoot it sideways
        # out of the bin.
        bin_stack_count = {"red": 0, "blue": 0}

        # ----- main pick & place loop -----
        for idx, cube in enumerate(cubes):
            # 1) Belt advances until this cube reaches the pick station
            _wait_for_cube(
                scene, cam, args.record, cubes, idx,
                dt=SIM_DT, speed=args.belt_speed,
            )

            # 2) Lower onto cube
            qpos_grab = franka.inverse_kinematics(
                link=end_effector,
                pos=np.array([PICK_X, PICK_Y, PICK_Z_GRAB]),
                quat=GRAB_QUAT,
            )
            _drive_qpos(franka, qpos_grab, 60, scene, cam, args.record, fingers_dof)

            # 3) Engage suction (weld cube to hand link)
            link_cube = cube["entity"].get_link("box_baselink").idx
            rigid_solver.add_weld_constraint(link_cube, link_franka)
            cube["state"] = "held"

            # 4) Lift back to transit height
            _drive_qpos(franka, qpos_home, 60, scene, cam, args.record, fingers_dof)

            # 5) Carry to the colour-matched bin (smooth plan)
            bin_pos = BIN_POS[cube["color"]]
            qpos_above_bin = franka.inverse_kinematics(
                link=end_effector,
                pos=np.array([bin_pos[0], bin_pos[1], PICK_Z_PRE]),
                quat=GRAB_QUAT,
            )
            path_to_bin = franka.plan_path(qpos_goal=qpos_above_bin, num_waypoints=120)
            _execute_path(franka, fingers_dof, path_to_bin, scene, cam, args.record)

            # 6) Lower toward the bin — release height grows with the
            # stack so the cube is dropped *just above* the previous one
            stack_idx = bin_stack_count[cube["color"]]
            release_z = BIN_BASE_DROP_Z + stack_idx * BIN_STACK_DH
            qpos_release = franka.inverse_kinematics(
                link=end_effector,
                pos=np.array([bin_pos[0], bin_pos[1], release_z]),
                quat=GRAB_QUAT,
            )
            _drive_qpos(franka, qpos_release, 50, scene, cam, args.record, fingers_dof)

            # 7) Release suction → cube falls into the bin under gravity
            rigid_solver.delete_weld_constraint(link_cube, link_franka)
            cube["state"] = "dropped"
            bin_stack_count[cube["color"]] += 1
            # Hold a beat at release height so the cube settles before
            # the gripper sweeps away (otherwise the lift can drag the
            # just-dropped cube sideways)
            _drive_qpos(franka, qpos_release, 35, scene, cam, args.record, fingers_dof)

            # 8) Lift away then plan back to the home pose
            _drive_qpos(franka, qpos_above_bin, 40, scene, cam, args.record, fingers_dof)
            path_home = franka.plan_path(qpos_goal=qpos_home, num_waypoints=120)
            _execute_path(franka, fingers_dof, path_home, scene, cam, args.record)

        # Final hold so the last drop fully settles in the bin
        _drive_qpos(franka, qpos_home, 100, scene, cam, args.record, fingers_dof)

    finally:
        if cam is not None:
            fps = max(10, int(round(1.0 / SIM_DT)))
            cam.stop_recording(save_to_filename=args.video, fps=fps)
            print(f"Saved video to {args.video}")


if __name__ == "__main__":
    main()

import argparse

import torch
from go2_env import Go2Env

import genesis as gs


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 70.0,
        "kd": 3.0,
        # termination
        "termination_if_roll_greater_than": 1000,  # degree
        "termination_if_pitch_greater_than": 1000,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.35],
        "base_init_quat": [0.0, 0.0, 0.0, 1.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.5,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 60,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "reward_scales": {},
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0, 0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


class BackflipEnv(Go2Env):
    def get_observations(self):
        phase = torch.pi * self.episode_length_buf[:, None] / self.max_episode_length
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                self.last_actions,  # 12
                torch.sin(phase),
                torch.cos(phase),
                torch.sin(phase / 2),
                torch.cos(phase / 2),
                torch.sin(phase / 4),
                torch.cos(phase / 4),
            ],
            axis=-1,
        )

        return self.obs_buf

    def step(self, actions):
        super().step(actions)
        self.get_observations()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="single")
    parser.add_argument("-r", "--record", action="store_true", help="Record an offscreen camera to mp4.")
    parser.add_argument("-o", "--video", type=str, default=None, help="Output video path (implies --record).")
    parser.add_argument("--no-viewer", action="store_true", help="Run headless (no on-screen viewer window).")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of env steps to run. Defaults to ~2 episodes when recording, infinite otherwise.",
    )
    parser.add_argument("--res", type=int, nargs=2, default=(1280, 720), help="Recording resolution (W H).")
    args = parser.parse_args()

    if args.video is not None:
        args.record = True
    if args.record and args.video is None:
        args.video = f"go2_backflip_{args.exp_name}.mp4"

    gs.init(backend=gs.amdgpu)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    if args.exp_name == "single":
        env_cfg["episode_length_s"] = 2
    elif args.exp_name == "double":
        env_cfg["episode_length_s"] = 3
    else:
        raise RuntimeError

    camera_kwargs = None
    if args.record:
        camera_kwargs = dict(
            res=tuple(args.res),
            pos=(2.5, 1.5, 1.2),
            lookat=(0.0, 0.0, 0.3),
            fov=40,
            GUI=False,
        )

    env = BackflipEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.no_viewer,
        camera_kwargs=camera_kwargs,
    )

    if args.record:
        env.cam.follow_entity(env.robot, fix_orientation=False)

    policy = torch.jit.load(f"./backflip/{args.exp_name}.pt")
    policy.to(device=gs.device)

    fps = int(round(1.0 / env.dt))
    if args.steps is None:
        if args.record:
            args.steps = 2 * env.max_episode_length
        else:
            args.steps = -1

    if args.record:
        env.cam.start_recording()

    obs = env.reset()
    try:
        with torch.no_grad():
            step_i = 0
            while args.steps < 0 or step_i < args.steps:
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                if args.record:
                    env.cam.render()
                step_i += 1
    except KeyboardInterrupt:
        print("Interrupted, finalizing recording...")
    finally:
        if args.record:
            env.cam.stop_recording(save_to_filename=args.video, fps=fps)
            print(f"Saved video to {args.video}")


if __name__ == "__main__":
    main()

"""
# evaluation (on-screen viewer)
python examples/locomotion/go2_backflip.py -e single
python examples/locomotion/go2_backflip.py -e double

# evaluation (record to mp4, no viewer)
python examples/locomotion/go2_backflip.py -e single  --record --no-viewer
python examples/locomotion/go2_backflip.py -e double  --record --no-viewer -o backflip_double.mp4
"""

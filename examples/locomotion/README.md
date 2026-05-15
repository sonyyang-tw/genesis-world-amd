# Go2 Locomotion on AMD ROCm

Genesis Go2 locomotion examples adapted for **AMD ROCm / `gs.amdgpu`**, with an
extended `go2_backflip.py` that supports **headless offscreen recording** to mp4
and a step-by-step Jupyter notebook tutorial.

Tested on Ryzen AI MAX (Radeon 8060S, 96 GB VRAM) inside the
`docker/Dockerfile.auplc` image (`genesis-world-amd:latest`).

## Files

| File | Purpose |
|---|---|
| `go2_backflip_tutorial.ipynb` | **Tutorial notebook**: runs the backflip policy end-to-end with embedded video playback. Best place to start. |
| `go2_env.py` | Base `Go2Env` (rigid Go2 + plane, PD control, observations). Adds an optional `camera_kwargs` for offscreen recording. |
| `go2_backflip.py` | CLI version: loads a pretrained `.pt` policy and runs single / double backflip. Supports on-screen viewer **and** headless mp4 recording. |
| `go2_train.py` | PPO training (needs `rsl-rl-lib>=5.0.0`). Uses `gs.amdgpu`. |
| `go2_eval.py` | Evaluate a trained walking policy. Uses `gs.amdgpu`. |
| `backflip/` | Pretrained backflip policies (`single.pt`, `double.pt`). See `backflip/readme.md` for the download link. |

All four `gs.init(...)` entry points use `backend=gs.amdgpu` so they pick up the
ROCm path automatically.

---

## 1. Build the docker image

From the repo root:

```bash
docker build -f docker/Dockerfile.auplc -t genesis-world-amd:latest .
```

This produces the `genesis-world-amd:latest` image with Genesis 0.4.6,
PyTorch 2.9.1+rocm7.11.0, JupyterLab 4.5, and all the ROCm GL/EGL libs the
notebook needs. Build takes ~10 min the first time.

## 2. Get the backflip checkpoints

The pretrained policies are not redistributed with this repo. Follow
`backflip/readme.md`:

1. Download `single.pt` and `double.pt` from the
   [Google Drive folder](https://drive.google.com/drive/folders/1ZxBaDP4_Br0ZhriQx_8A3JIwZZLnxix4?usp=drive_link).
2. Place them under `examples/locomotion/backflip/`.

Final layout:

```
examples/locomotion/backflip/
├── single.pt
├── double.pt
└── readme.md
```

## 3. Run — `docker run` launches the notebook

From the repo root, this single command spins up the container, mounts the
repo, forwards the GPU, and starts JupyterLab on port **8888** with the
tutorial notebook ready to open:

```bash
docker rm -f gw-notebook 2>/dev/null
docker run -d --name gw-notebook \
  --privileged --group-add dialout --group-add video \
  --ipc=host --shm-size=8g \
  --device=/dev/kfd --device=/dev/dri \
  -p 8888:8888 \
  -v "$(pwd)":/opt/workspace/genesis-world \
  -w /opt/workspace/genesis-world/examples/locomotion \
  genesis-world-amd:latest \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --ServerApp.token=backflip --ServerApp.password='' \
    --ServerApp.root_dir=/opt/workspace/genesis-world
```

Now open the notebook in your browser:

<http://127.0.0.1:8888/lab/tree/examples/locomotion/go2_backflip_tutorial.ipynb?token=backflip>

Hit **Run All** — the notebook walks through nine pedagogical sections
(Genesis init → cfgs → phase-aware observation override → scene build → policy
load → rollout + recording → inline video playback). The last cell embeds the
recorded MP4 right inside the notebook. Toggle the `exp_name` cell between
`"single"` and `"double"` to switch policies and re-run.

Stream the logs / stop the server with:

```bash
docker logs -f gw-notebook     # watch the JupyterLab logs
docker stop gw-notebook        # stop the server
docker rm   gw-notebook        # remove the container
```

> Want a shell inside the running container instead of the notebook?
> `docker exec -it gw-notebook bash` puts you in `examples/locomotion/`
> with the same env, ready to run any of the CLI variants in §4.

## 4. CLI variants of `go2_backflip.py`

If you'd rather skip JupyterLab and drive the script directly, exec into the
container (or any container built from the same image) and run the script.
`go2_backflip.py` reads `./backflip/<exp>.pt` as a **relative path**, so always
run it from `examples/locomotion/`.

```bash
docker exec -it gw-notebook bash
cd /opt/workspace/genesis-world/examples/locomotion
```

### A) On-screen viewer (default)

Needs a display / EGL context. Inside a desktop session or with X11 forwarded:

```bash
python3 go2_backflip.py -e single   # 2 s episode
python3 go2_backflip.py -e double   # 3 s episode
```

Press `Ctrl+C` in the terminal to exit (the script loops forever by default
when not recording).

### B) Headless mp4 recording (recommended on remote / SSH boxes)

```bash
# Single backflip → ./go2_backflip_single.mp4
python3 go2_backflip.py -e single --record --no-viewer

# Double backflip, custom output path
python3 go2_backflip.py -e double --record --no-viewer -o backflip_double.mp4
```

Defaults when `--record` is enabled:

- resolution `1280x720`
- camera follows the Go2 robot
- runs for `2 * episode_length` (single ≈ 200 frames / 4 s, double ≈ 300 frames / 6 s)
- output FPS = 50 (matches `dt=0.02`)

### C) All CLI flags

```text
-e/--exp_name {single,double}   which policy to load   [default: single]
-r/--record                     enable mp4 recording
-o/--video PATH                 output file (implies --record)
--no-viewer                     disable the on-screen viewer (headless)
--steps N                       number of env steps to run
                                (default: 2 episodes when recording, infinite otherwise)
--res W H                       recording resolution (pixels)
```

### D) Custom shots

```bash
# 1080p, 6 s of single backflip
python3 go2_backflip.py -e single --record --no-viewer --res 1920 1080 --steps 300

# both policies, back-to-back, for a comparison reel
python3 go2_backflip.py -e single --record --no-viewer -o single.mp4
python3 go2_backflip.py -e double --record --no-viewer -o double.mp4
```

## 5. Training / Eval (walking policy)

Walking uses `rsl-rl-lib`. Inside the container:

```bash
pip install "rsl-rl-lib>=5.0.0" tensorboard
```

Then:

```bash
# Train (large batch, headless)
python3 go2_train.py -e go2-walking -B 4096 --max_iterations 100

# Evaluate a checkpoint (opens viewer)
python3 go2_eval.py -e go2-walking --ckpt 100
```

`go2_eval.py` currently spawns the on-screen viewer; if you need a headless mp4
the same pattern as `go2_backflip.py` can be applied (extend `Go2Env` via
`camera_kwargs`, then `env.cam.start_recording()` / `render()` / `stop_recording()`).

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: tensordict` | `pip install tensordict` (already pinned in `Dockerfile.auplc`). |
| `ValueError: not enough values to unpack` on `env.reset()` | Make sure you are on this fork — `Go2Env.reset()` returns a single tensor, the demo here unpacks it accordingly. |
| `EGL` / `libGL.so` errors | Verify `libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev libgbm1` are installed; the AUPLC Dockerfile already does this. |
| Long startup the first time | Genesis JIT-compiles ROCm kernels (~5–10 s); cached afterwards. |
| `Neutral robot position (qpos0) exceeds joint limits` warning | Cosmetic warning from the Go2 URDF; safe to ignore. |
| Notebook kernel dies during `scene.build()` | Make sure you launched JupyterLab from `genesis-world-amd:latest` (the auplc image). Other variants ship a different `quadrants` / `gstaichi` combination that this notebook does not target. |
| `port is already allocated` when starting `gw-notebook` | Another container is on `8888`. Run `docker rm -f gw-notebook` or change `-p 8888:8888` to `-p 8889:8888` (then visit `http://127.0.0.1:8889/...`). |
| `ModuleNotFoundError: go2_env` in the notebook | Make sure you opened the notebook from `examples/locomotion/`. Cell 2 prepends `os.getcwd()` to `sys.path`, but only if the notebook was launched from that directory. |

## 7. Related

- Upstream backflip training code (domain randomization, reference rewards):
  <https://github.com/ziyanx02/Genesis-backflip>
- Genesis docs: <https://genesis-world.readthedocs.io/>

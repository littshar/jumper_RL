# Jumper — Three Motions with a Nailong Skin

This is a lightweight release bundle for three policies trained on the
**Jumper** 22-DoF crab-shaped hexapod, together with the Nailong-themed
appearance used in the demonstrations.

The bundle contains weights, the appearance files and this documentation. The
tasks themselves live in the repository this directory belongs to
(`tasks/jumper/dance_sway/`, `tasks/jumper/dance_pitch/`,
`tasks/jumper/backflip/`), so play and training commands are run from the
**repository root**.

## Contents

```
release/
├── README.md
├── README.zh.md
├── SHA256SUMS
├── checkpoints/
│   ├── you_stink_model_17649.pt
│   ├── nailong_belly_laugh_model_23648.pt
│   └── xiaoliang_big_move_model_9297.pt
└── nailong_skin/
    ├── jumper.xml
    ├── upper_shell_link.obj
    ├── upper_shell_link.mtl
    └── upper_shell_link_basecolor.png
```

## The three motions

| Chinese title | English title | Task id | Checkpoint | Motion |
|---|---|---|---|---|
| 你很臭啊 | You Stink! | `jumper.dance_sway` | `you_stink_model_17649.pt` | Standing on four legs, the body sways side-to-side on a 2-second near-square rhythm while the right claw waves at four times the sway frequency. |
| 奶龙捧腹大笑 | Nailong's Belly Laugh | `jumper.dance_pitch` | `nailong_belly_laugh_model_23648.pt` | A 9-second looping pitch dance: the body nods between 20°, −10°, 5° and back to 20°, with both shoulders swinging at a 0.5-second period. |
| 小亮给他们整个活 | Xiaoliang Shows Them a Trick | `jumper.backflip` | `xiaoliang_big_move_model_9297.pt` | A 360° backward flip: crouch, jump to about 0.30 m, one full backward rotation, feet-first landing. |

`dance_sway` was trained on the default model. `dance_pitch` and `backflip`
were trained on the red-shell model (`assets/jumper/jumper_red.xml`), which is
physics-identical to the default `jumper.xml` — same bodies, joints, ranges,
masses and inertias in the same order — so all three checkpoints replay on the
Nailong skin without any conversion.

## Quick start

### Requirements

- Linux, Python 3.10–3.13.
- An NVIDIA GPU is recommended. Training and replay use
  [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) through
  `mjlab`; the `native` backend can run on CPU.
- The repository's own setup guide is `docs/AGENT_SETUP.md`.

### Install the repository

```bash
git clone <this-repository>
cd <this-repository>
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

`pip install -e .` is required: the vendored `mjlab` and `rsl_rl` live under
`rl/`, which is not on `sys.path` otherwise.

### Replay a motion

```bash
# 你很臭啊 / You Stink!
.venv/bin/python scripts/play.py --task jumper.dance_sway \
    --checkpoint release/checkpoints/you_stink_model_17649.pt

# 奶龙捧腹大笑 / Nailong's Belly Laugh
.venv/bin/python scripts/play.py --task jumper.dance_pitch \
    --checkpoint release/checkpoints/nailong_belly_laugh_model_23648.pt

# 小亮给他们整个活 / Xiaoliang Shows Them a Trick
.venv/bin/python scripts/play.py --task jumper.backflip \
    --checkpoint release/checkpoints/xiaoliang_big_move_model_9297.pt
```

No `--model` is passed, so the default `assets/jumper/jumper.xml` is used. If
the Nailong skin is already installed (see below), that is what you see.

To record a headless video instead of opening the viewer:

```bash
.venv/bin/python scripts/play.py --task jumper.backflip \
    --checkpoint release/checkpoints/xiaoliang_big_move_model_9297.pt \
    --headless --steps 650 --video backflip.mp4 --video-fps 50
```

`--steps 650` covers one 3-second episode at the task's 200 Hz control rate.
The backflip episode ends by itself at step 600, so `--stop-on-done` also works.

## The Nailong skin

The skin is appearance-only. Collision geometry and dynamics are unchanged, so
the policies behave exactly as they did on the red training model.

Files:

- `nailong_skin/jumper.xml` — the complete model file, with the Nailong colours
  and the textured upper shell wired in.
- `upper_shell_link.obj`, `.mtl`, `_basecolor.png` — the textured shell mesh.
- The XML still references `meshes/upper_shell_link_col.STL` for collision;
  that file is already in the repository.

Install it by replacing the model file and copying the shell into the visual
mesh directory (back up the original first):

```bash
cp assets/jumper/jumper.xml assets/jumper/jumper.xml.orig
cp release/nailong_skin/jumper.xml assets/jumper/jumper.xml
cp release/nailong_skin/upper_shell_link.obj \
   release/nailong_skin/upper_shell_link.mtl \
   release/nailong_skin/upper_shell_link_basecolor.png \
   assets/jumper/urdf/jumper/meshes/visual/
```

To replay on the red shell used for training instead, pass
`--model jumper_red`.

## Training (optional)

The checkpoints are the result of continued training; these are the commands
that reproduce the setup.

```bash
# 你很臭啊 — parametric sway dance, 2048 environments
.venv/bin/python scripts/train.py --task jumper.dance_sway \
    --num_envs 2048 --headless

# 奶龙捧腹大笑 — warm-started from the sway dance, 2048 environments.
# Use the newest sway checkpoint; train the sway dance first if logs/ is empty.
SWAY=$(ls -t logs/jumper/jumper.dance_sway/*/model_*.pt | head -1)
.venv/bin/python scripts/train.py --task jumper.dance_pitch \
    --model jumper_red --checkpoint "$SWAY" --num_envs 2048 --headless

# 小亮给他们整个活 — 512 environments on an 8 GB laptop GPU
.venv/bin/python scripts/train.py --task jumper.backflip \
    --model jumper_red \
    --checkpoint tasks/jumper/ref_free_jump/out/example/model_5999.pt \
    --num_envs 512 --headless
```

`--max-iterations` is the number of iterations **this run performs**, so on a
resumed run it is how many more to do on top of the checkpoint.

TensorBoard serves `logs/<model>/<task>/` during training at
<http://localhost:6006/>.

## Export and deployment

The policy checkpoints are PyTorch full-state files. To produce an ONNX policy
for deployment:

```bash
.venv/bin/python scripts/export.py --task jumper.backflip \
    --checkpoint release/checkpoints/xiaoliang_big_move_model_9297.pt
```

See `deploy/README.md` for the controller and bundle workflow.

## Known limitations

- **Backflip landing.** The trained policy lands on about three of its six feet
  on average and does not return to the HOME pose afterwards; the episode times
  out three seconds after the jump starts. It is a reliable single backflip,
  not a repeated acrobatic routine.
- **Dances are parametric.** They are not motion-capture clips; the policy
  tracks a periodic reference built into the task. They run for as long as the
  policy is asked to.
- **Simulation only.** The checkpoints are validated in `mjlab` / MuJoCo. Real
  robot deployment needs the servo model, the controller manifest and the
  hardware calibration described in `deploy/README.md`.

## License

The repository is licensed under the Apache License 2.0; see `LICENSE` and
`NOTICE` in the repository root. Third-party components and their licences are
listed in `NOTICE`.

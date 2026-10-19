<!-- tracks: README.md @ sha256:5bf5b729a067b666 -->
# Jumper — 三个动作与奶龙皮肤

这是一个轻量发布包，包含在 **Jumper** 22 自由度螃蟹形六足机器人上训练的
三个策略，以及演示中使用的奶龙主题外观。

发布包只包含权重、外观文件和本文档；三个任务本身在对应的仓库中
（`tasks/jumper/dance_sway/`、`tasks/jumper/dance_pitch/`、
`tasks/jumper/backflip/`），因此播放和训练命令都在**仓库根目录**执行。

## 目录内容

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

## 三个动作

| 中文名 | 英文名 | 任务 id | 权重 | 动作内容 |
|---|---|---|---|---|
| 你很臭啊 | You Stink! | `jumper.dance_sway` | `you_stink_model_17649.pt` | 四条腿站立，身体按 2 秒一个周期的近方波左右平移，同时右爪以 4 倍频率挥动。 |
| 奶龙捧腹大笑 | Nailong's Belly Laugh | `jumper.dance_pitch` | `nailong_belly_laugh_model_23648.pt` | 9 秒循环的俯仰舞：身体在 20°、−10°、5° 之间点头再回到 20°，双肩以 0.5 秒周期摆动。 |
| 小亮给他们整个活 | Xiaoliang Shows Them a Trick | `jumper.backflip` | `xiaoliang_big_move_model_9297.pt` | 360° 后空翻：下蹲、起跳到约 0.30 m、完整后仰一圈、脚先落地。 |

`dance_sway` 在默认模型上训练；`dance_pitch` 和 `backflip` 在红色外壳模型
（`assets/jumper/jumper_red.xml`）上训练。红模与默认的 `jumper.xml`
物理完全一致——相同的 body、joint、限位、质量和惯量，顺序也相同——所以三个
权重都可以直接在奶龙皮肤上回放，不需要任何转换。

## 快速开始

### 环境要求

- Linux，Python 3.10–3.13。
- 推荐 NVIDIA 显卡。训练和回放通过 `mjlab` 使用
  [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)；`native`
  后端可以用 CPU 运行。
- 仓库自带的安装指南见 `docs/AGENT_SETUP.md`。

### 安装仓库

```bash
git clone <this-repository>
cd <this-repository>
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

必须执行 `pip install -e .`：vendored 的 `mjlab` 和 `rsl_rl` 位于 `rl/`
下，否则不在 `sys.path` 中。

### 回放动作

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

这里不传 `--model`，因此使用默认的 `assets/jumper/jumper.xml`。如果已经
装好奶龙皮肤（见下文），看到的就是奶龙外观。

如果要无窗口录制视频而不是打开 viewer：

```bash
.venv/bin/python scripts/play.py --task jumper.backflip \
    --checkpoint release/checkpoints/xiaoliang_big_move_model_9297.pt \
    --headless --steps 650 --video backflip.mp4 --video-fps 50
```

任务控制频率为 200 Hz，`--steps 650` 覆盖一个 3 秒 episode。后空翻 episode
会在第 600 步自行结束，因此也可以用 `--stop-on-done`。

## 奶龙皮肤

皮肤只改外观，碰撞几何和动力学不变，所以策略表现与红色训练模型完全一致。

文件说明：

- `nailong_skin/jumper.xml` — 完整模型文件，包含奶龙配色和带贴图的上壳。
- `upper_shell_link.obj`、`.mtl`、`_basecolor.png` — 带贴图的外壳网格。
- XML 仍通过 `meshes/upper_shell_link_col.STL` 做碰撞，该文件仓库中已有。

安装方式（先备份原模型，再替换并复制外壳）：

```bash
cp assets/jumper/jumper.xml assets/jumper/jumper.xml.orig
cp release/nailong_skin/jumper.xml assets/jumper/jumper.xml
cp release/nailong_skin/upper_shell_link.obj \
   release/nailong_skin/upper_shell_link.mtl \
   release/nailong_skin/upper_shell_link_basecolor.png \
   assets/jumper/urdf/jumper/meshes/visual/
```

如果想改回训练时用的红色外壳回放，传 `--model jumper_red`。

## 训练（可选）

这些权重是继续训练得到的；以下命令对应当时的设置。

```bash
# 你很臭啊 — 参数化摇摆舞，2048 个环境
.venv/bin/python scripts/train.py --task jumper.dance_sway \
    --num_envs 2048 --headless

# 奶龙捧腹大笑 — 从摇摆舞热启动，2048 个环境。
# 取最新的摇摆舞 checkpoint；如果 logs/ 是空的，先训练摇摆舞。
SWAY=$(ls -t logs/jumper/jumper.dance_sway/*/model_*.pt | head -1)
.venv/bin/python scripts/train.py --task jumper.dance_pitch \
    --model jumper_red --checkpoint "$SWAY" --num_envs 2048 --headless

# 小亮给他们整个活 — 8 GB 笔记本显卡上使用 512 个环境
.venv/bin/python scripts/train.py --task jumper.backflip \
    --model jumper_red \
    --checkpoint tasks/jumper/ref_free_jump/out/example/model_5999.pt \
    --num_envs 512 --headless
```

`--max-iterations` 的含义是**本次运行执行多少轮**，所以在续训时它表示在
checkpoint 基础上再跑多少轮。

训练期间 TensorBoard 会服务 `logs/<model>/<task>/`，地址是
<http://localhost:6006/>。

## 导出与部署

策略权重是 PyTorch 全状态文件。要生成用于部署的 ONNX：

```bash
.venv/bin/python scripts/export.py --task jumper.backflip \
    --checkpoint release/checkpoints/xiaoliang_big_move_model_9297.pt
```

控制器与 bundle 流程见 `deploy/README.md`。

## 已知限制

- **后空翻落地。** 训练后的策略平均大约用三只脚着地（共六只），落地后不会
  回到 HOME 姿态；episode 在起跳三秒后超时结束。它是一次可靠的后空翻，
  不是连续杂技。
- **舞蹈是参数化的。** 它们不是动作捕捉片段，而是跟踪任务内置的周期参考。
  只要策略在运行，动作就会持续循环。
- **仅限仿真验证。** 权重在 `mjlab` / MuJoCo 中验证过。上实机还需要伺服
  模型、控制器 manifest 和硬件标定，见 `deploy/README.md`。

## 许可证

仓库使用 Apache License 2.0，见仓库根目录的 `LICENSE` 和 `NOTICE`。
第三方组件及其许可证列在 `NOTICE` 中。

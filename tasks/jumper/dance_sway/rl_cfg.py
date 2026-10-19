"""Hyper-parameters for jumper.dance_sway.

**Only this file changes this task's training hyper-parameters**; no other task
is affected. The baseline is `jumper.flat`'s, with one departure.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

#: The exploration std the actor starts at. The swing note's reasoning, at the
#: stand task's number: the dance holds a stance that forgives nothing like a
#: fall, and the phase observation means the policy does not need noise to
#: find the rhythm -- the clock tells it where in the wave it is, and what
#: remains is to track. 0.2 is +-0.05 rad of stance jitter, enough to find
#: the sway's size, small enough that early episodes survive their first
#: periods. If the tracking plateaus shy of the amplitude, this is the first
#: number to raise.
INIT_STD = 0.2


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config.

    `experiment_name` is the middle segment of the log path: the full structure is
    `logs/<model>/jumper.dance_sway/<date-time>`, with the outermost segment decided by
    `--model`.
    """
    return jumper_ppo_baseline(
        experiment_name="jumper.dance_sway",
        # ── Network ──
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=INIT_STD,
        # ── Symmetry ──
        # **Off: the dance is chiral.** The left claw holds while the right
        # waves; mirrored, that is a different dance, and the sway phase's
        # mirror -- a half-period shift on the sway clock -- does not even
        # transport consistently onto the claw's 4x clock (a half sway period
        # is two full claw periods, so the claw's phase would need to stay
        # while the sway's flips). `gesture_bow` is the precedent for a
        # gesture that lives on one side only.
        symmetry=False,
        use_data_augmentation=False,
        use_mirror_loss=False,
        # ── PPO ──
        entropy_coef=0.01,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        # ── Runner ──
        num_steps_per_env=24,
        # **Iterations this run performs, not a total** -- on `--resume` it is
        # how many more, on top of the checkpoint's. 6000 more: the standing
        # base converges by ~1500 on a warm start, and the last revision's
        # lesson is that the sway curve shows whether the amplitude is being
        # found within a few hundred -- there is no case for 10000.
        max_iterations=6000,
    )


# No `runner_cls`: the velocity runner logs tracking metrics against a twist
# command this task does not have, and the ladders it checkpoints do not exist
# here. `registry.load_runner_cls` treats a missing function as "use rsl_rl's
# default".


def play_status():
    """The line `scripts/play.py` prints while replaying.

    The dance's quality is the phase lock, so the line prints the body's
    lateral offset beside the reference it should be tracking, the wrist
    joint beside its band's reference, and the fore-aft drift -- a replay
    watched without these is a robot wiggling ambiguously.
    """

    def status(entity, index: int, env) -> str:
        import math

        import torch

        from ..dance_sway.env_cfg import CLAW_JOINT, SUPPORT_LEGS

        unwrapped = getattr(env, "unwrapped", env)
        with torch.inference_mode():
            cfg = SceneEntityCfg("robot", site_names=SUPPORT_LEGS, preserve_order=True)
            cfg.resolve(unwrapped.scene)
            lat = float(lateral_offset(unwrapped, cfg)[index])
            t = float(unwrapped.episode_length_buf[index]) * unwrapped.step_dt
            names = list(entity.joint_names)
            claw = float(
                entity.data.joint_pos[index, names.index(CLAW_JOINT)]
            )
            phase = 2 * math.pi * t / 2.0
            lr = LATERAL_AMPLITUDE * square_sine(torch.tensor(phase), SQUARE_K)
            cr = CLAW_CENTER + CLAW_AMPLITUDE * square_sine(
                torch.tensor(4.0 * phase), CLAW_K
            )
            drift = float(entity.data.root_link_lin_vel_b[index, 0])
        return (
            f"sway {lat * 100:6.1f} / ref {float(lr) * 100:6.1f} cm   "
            f"claw {claw:6.2f} / ref {float(cr):6.2f} rad   "
            f"vx {drift:5.3f} m/s"
        )

    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from ..dance_sway.env_cfg import (
        CLAW_AMPLITUDE,
        CLAW_CENTER,
        LATERAL_AMPLITUDE,
    )
    from ..dance_sway.mdp.rewards import (
        CLAW_K,
        SQUARE_K,
        lateral_offset,
        square_sine,
    )

    return status

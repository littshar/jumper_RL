"""Hyper-parameters for jumper.dance_pitch.

**Only this file changes this task's training hyper-parameters**; no other task
is affected. The baseline is `jumper.flat`'s, with `jumper.dance_sway`'s
departures carried over.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

#: The exploration std the actor starts at. `dance_sway`'s reasoning at its
#: number: the dance holds a stance that forgives nothing like a fall, and
#: the phase observation means the policy does not need noise to find the
#: rhythm -- the clock tells it where in the loop it is, and what remains is
#: to track. If the tracking plateaus shy of the pitch swings, this is the
#: first number to raise.
INIT_STD = 0.2


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config.

    `experiment_name` is the middle segment of the log path: the full structure is
    `logs/<model>/jumper.dance_pitch/<date-time>`, with the outermost segment decided by
    `--model`.
    """
    return jumper_ppo_baseline(
        experiment_name="jumper.dance_pitch",
        # ── Network ──
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=INIT_STD,
        # ── Symmetry ──
        # **Off: the dance is chiral.** The shoulders wave around different
        # centres (-0.6 against -1.0), so mirrored the wave would sit in the
        # wrong band on one side; and the loop clock's mirror does not
        # transport onto the wobble clock any more than `dance_sway`'s did.
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
        # how many more, on top of the checkpoint's. 6000 more: the plan is a
        # warm start from `dance_sway`'s standing base, whose own run had the
        # stance converged by ~1500; the pitch loop is a bigger ask than the
        # sway was, so the budget is the same rather than smaller.
        max_iterations=6000,
    )


# No `runner_cls`: the velocity runner logs tracking metrics against a twist
# command this task does not have (see `dance_sway`'s note).


def play_status():
    """The line `scripts/play.py` prints while replaying.

    The dance's quality is the phase lock, so the line prints the body's
    pitch beside the loop's reference, both shoulders beside their wave's,
    and the fore-aft drift.
    """

    def status(entity, index: int, env) -> str:
        import math

        import torch

        from ..dance_pitch.env_cfg import J1_CENTERS, J1_JOINTS
        from ..dance_pitch.mdp.rewards import (
            LOOP_PERIOD,
            measured_pitch,
            pitch_ref,
        )

        unwrapped = getattr(env, "unwrapped", env)
        with torch.inference_mode():
            pitch = float(measured_pitch(unwrapped)[index])
            ref = float(pitch_ref(unwrapped)[index])
            t = float(unwrapped.episode_length_buf[index]) * unwrapped.step_dt
            t = math.remainder(t, LOOP_PERIOD)
            names = list(entity.joint_names)
            j1 = [float(entity.data.joint_pos[index, names.index(n)]) for n in J1_JOINTS]
            j1_ref = [
                c + math.radians(5.0) * math.sin(4.0 * math.pi * t - math.pi / 2)
                for c in J1_CENTERS
            ]
            drift = float(entity.data.root_link_lin_vel_b[index, 0])
        return (
            f"pitch {math.degrees(pitch):6.1f} / ref {math.degrees(ref):6.1f} deg   "
            f"LF_J1 {j1[0]:6.2f} / ref {j1_ref[0]:5.2f}   "
            f"RF_J1 {j1[1]:6.2f} / ref {j1_ref[1]:5.2f} rad   "
            f"vx {drift:5.3f} m/s"
        )

    return status

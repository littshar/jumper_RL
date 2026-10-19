"""Hyper-parameters for jumper.stand.

**Only this file changes this task's training hyper-parameters**; no other task
is affected. The baseline is `jumper.flat`'s, and two numbers depart from it,
each with its reason below.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

#: The exploration std the actor starts at, against the velocity tasks' 1.0.
#:
#: `jumper.swing` measured the failure the 1.0 default produces on a robot with
#: nowhere to err: sigma 1.0 is +/-0.25 rad on twenty joints at once, and on a
#: task whose episodes end by falling over that is a policy knocking itself down
#: before it has stood up. An early stand task needs episodes that *last*,
#: because survival is both the reward and the push ladder's only evidence.
#:
#: **0.08 here, and it is the second value this constant took.** The first
#: run launched at swing's reasoning -- 0.2, +/-0.05 rad, small enough not to
#: knock itself over -- and 70 iterations in, episode outcomes were flat:
#: `toppled` counts unmoved, survival 0.5%. The flaw was in comparing the
#: noise to the stance's *strength* rather than to its own motion. The
#: batch-settled PD hold oscillates at +/-10 deg (0.17 rad) **with zero
#: noise** -- it is a marginally damped pendulum, and that oscillation is the
#: behaviour the policy starts from. Exploration at 0.05 rad per joint across
#: twenty joints is a perturbation three times that size, applied on top; the
#: episodes died of the noise before any gradient about balancing could form.
#:
#: 0.08 is +/-0.02 rad -- a tenth of the hold's own oscillation, so zero
#: action remains the good behaviour it measurably is (the batch settle held
#: 5 s, 28 deg worst excursion), and the policy's exploration perturbs a
#: working stance instead of drowning it. The cost is slower exploration of
#: large postures; if the policy stalls having learnt only to freeze, this is
#: the first number to raise.
INIT_STD = 0.08


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config.

    `experiment_name` is the middle segment of the log path: the full structure is
    `logs/<model>/jumper.stand/<date-time>`, with the outermost segment decided by
    `--model`.
    """
    return jumper_ppo_baseline(
        experiment_name="jumper.stand",
        # ── Network ──
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=INIT_STD,
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
        max_iterations=10_000,
    )


def runner_cls() -> type:
    """`CurriculumRunner`, so the push ladder's level rides in the checkpoints.

    A resume that dropped the ladder to rung 0 would retrain the policy on pushes
    it has outgrown -- the exact failure `CheckpointedLadder` exists for. The
    runner is the velocity one underneath, so it logs the tracking metrics too;
    at a zero command they read as the drift rulers they here are.
    """
    from ..common.runner import CurriculumRunner

    return CurriculumRunner


def play_status():
    """The line `scripts/play.py` prints while replaying.

    Not the velocity tasks' line -- there is no command to achieve a fraction of.
    What a stand replay has to answer: is it at height, is it level, is it
    drifting, and is it swaying. The height is printed **beside** `STAND_Z`
    rather than as an error, because the error that matters is signed -- a policy
    settled 5 mm low is crouching its way to stability, which a magnitude would
    hide and the pairing shows.
    """

    def status(entity, index: int, env) -> str:
        import torch

        del env
        with torch.inference_mode():
            z = float(entity.data.root_link_pos_w[index, 2])
            gravity_b = entity.data.projected_gravity_b[index]
            tilt = float(
                torch.rad2deg(torch.arccos((-gravity_b[2]).clamp(-1.0, 1.0)))
            )
            drift = float(entity.data.root_link_lin_vel_b[index, :2].norm())
            sway = float(entity.data.root_link_ang_vel_b[index].norm())
        return (
            f"height {z * 1000:6.1f} mm (STAND_Z {0.10647 * 1000:.1f})   "
            f"tilt {tilt:5.2f} deg   "
            f"drift {drift:6.3f} m/s   "
            f"sway {sway:5.2f} rad/s"
        )

    return status

"""PPO hyperparameters for jumper.backflip.

These are `jumper.ref_free_jump`'s, deliberately: the published `model_5999` was
trained with them, the observation layout is unchanged, and the action layout is
unchanged, so that checkpoint is a valid pretrained initialization. The actor is
406 observations wide (3 base angular velocity + 3 projected gravity + four
history terms x 20 driven joints x 5 frames) and 20 actions; the critic is 435
(the actor's terms, base linear velocity, and the four flight/height
bookkeeping terms). Adding or reordering an observation term would change those
shapes and stop the checkpoint from loading.

The schedule is the jump's: 48 control steps per update at 200 Hz, a fourth-root
discount for the 200 Hz step, and the low entropy coefficient the jump task
needed once the motion was being learnt rather than explored.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

#: Environment steps per policy update. The jump task's 48, so the pretrained
#: checkpoint's optimizer and iteration counter line up with this rollout.
NUM_STEPS_PER_ENV = 48


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.backflip",
        # The pretrained checkpoint's network: four layers, 512/256/128/64.
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        # Actions are in units of ACTION_SCALE = 0.5 rad, so 0.3 is 0.15 rad of
        # exploration, as in the jump task.
        init_std=0.3,
        entropy_coef=0.001,
        # 200 Hz: keep the jump task's 0.5 s discount horizon.
        gamma=0.99**0.25,
        lam=0.95**0.25,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        # model_5999 is the pretrained start; this leaves room for a further
        # ~1000 iterations by default. `--max-iterations` overrides it and, on a
        # resumed run, is how many more iterations to perform.
        max_iterations=7000,
    )

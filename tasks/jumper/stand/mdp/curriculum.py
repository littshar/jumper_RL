"""The push ladder: how hard shoving is, as a function of surviving shoving.

The one curriculum this task owns. The velocity tasks widen a *command* as the
policy learns to track it; there is no command here to widen, so what widens is
the **disturbance** -- the only axis this task has to climb.

The gate is **survival**, not an error average. A tracking curriculum can read
tracking error off the command; this task's command is identically zero, so the
error ruler does not exist. What does exist is the thing the task is for: an
environment reset either because it fell over (`fell_over`) or because it
outlived the episode. The share that outlived it is the sample; a promotion asks
for most of them to have survived the current rung before the robot is asked to
survive a harder one.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.curriculum_manager import CurriculumTermCfg

from ...common.mdp.curriculum import CheckpointedLadder

#: The ladder's rungs, as +/- m/s on each axis of `push_by_setting_velocity`'s
#: kick, rescaled for the rearing stance. The six-foot ladder's middle rung was
#: `PUSH_VELOCITY_RANGE` itself, sized against a robot standing on six feet;
#: this stance stands 0.18 m tall on a line polygon and balances actively, so
#: the whole ladder sits an order below -- the top rung here is a quarter of
#: the old top, which is already a real shove for a 55 deg pendulum.
#:
#: The magnitudes are a starting guess, not a measurement, and the failure is
#: not silent: survival falls, the gate holds the level, and
#: `Curriculum/push/survival` shows it. If the ladder parks at rung 1 for a
#: long time, that number is the reason and the top rung is the thing to
#: revisit, not the bar.
PUSH_LEVELS_XY = (0.02, 0.04, 0.07)
PUSH_LEVELS_Z = (0.02, 0.04, 0.07)

#: The survival share a rung must reach before the next one opens. A tracking
#: ladder's 0.3 *range* has no analogue here -- survival is already a fraction --
#: so this is written directly. 0.8 keeps a fifth of the environments at the
#: current rung as a working population for the next one: promote at 1.0 and the
#: first steps of a new rung are taken by every environment at once, which is
#: not evidence of anything, and promote at 0.5 and the ladder climbs on a
#: coin flip.
SURVIVAL_BAR = 0.8

#: No promotion before this many environment steps, whatever the survival says.
#: At the flat baseline's 24-step rollout and the skeleton's 20 s episodes the
#: first resets arrive near iteration 42 (1000 steps an episode); 100 iterations
#: of warmup puts the earliest promotion well past the first full episodes, which
#: is the exact trap `jumper.posture` recorded -- its first promotion landed at
#: iteration 25, decided entirely by environments that had ended early.
WARMUP_ITERATIONS = 100
WARMUP_STEPS = WARMUP_ITERATIONS * 24


class PushStrengthCurriculum(CheckpointedLadder):
    """Widen the push as survival at the current width improves.

    A class term rather than a function because the level and the survival
    average have to live between calls, and across `--resume` -- hence the
    `CheckpointedLadder` mixin, which `common/runner.py`'s `CurriculumRunner`
    saves into every checkpoint.

    **The range is written through the event manager, every call.** Like
    `CommandRangeCurriculum` writes the std through the reward manager: the
    manager deep-copies its config at construction, so an assignment to
    `env.cfg.events[...]` changes an object nothing reads. `_apply` is idempotent
    -- it rewrites three tuples into one dict -- so running it on every call
    rather than only on promotion keeps the level in force even if a promotion
    and a checkpoint restore land in the same step, which is exactly the corner
    `load_state_dict`'s `_applied = False` exists to re-enter.
    """

    #: The environment variable that sets the level by hand.
    LEVEL_ENV = "MJRL_PUSH_LEVEL"

    #: Restored with the level. The survival average is evidence *about one
    #: rung*: carried onto a wider rung it would decide the next promotion on
    #: measurements of an easier disturbance.
    EVIDENCE = ("survival_ema", "last_promotion_step")

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
        del env
        params = getattr(cfg, "params", None) or {}
        top = len(params.get("levels_xy", PUSH_LEVELS_XY)) - 1
        self.level = self._level_from_env(top)
        self._start_level = self.level
        self.survival_ema: float | None = None
        # Samples pooled across resets until `min_samples` have arrived -- the
        # same arithmetic `CommandRangeCurriculum` was measured to need: at a
        # 1000-step episode the reset batch is one environment nearly always, and
        # a gate that demanded them in one batch would never fire at all.
        self._pending = 0.0
        self._pending_n = 0
        self.last_promotion_step = 0
        self._applied = False

    def _apply(
        self, env: ManagerBasedRlEnv, levels_xy: tuple[float, ...],
        levels_z: tuple[float, ...],
    ) -> float:
        xy = levels_xy[self.level]
        z = levels_z[self.level]
        velocity_range = env.event_manager.get_term_cfg("push_robot").params[
            "velocity_range"
        ]
        velocity_range["x"] = (-xy, xy)
        velocity_range["y"] = (-xy, xy)
        velocity_range["z"] = (-z, z)
        return xy

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        levels_xy: tuple[float, ...] = PUSH_LEVELS_XY,
        levels_z: tuple[float, ...] = PUSH_LEVELS_Z,
        survival_bar: float = SURVIVAL_BAR,
        ema_alpha: float = 0.05,
        dwell_steps: int = 24 * 100,
        min_steps: int = 50,
        min_samples: int = 8,
        warmup_steps: int = WARMUP_STEPS,
    ) -> dict[str, torch.Tensor | float]:
        """Advance at most one level per call; return the state for logging.

        Args:
            survival_bar: the survival EMA a rung must reach before the next
                opens; `SURVIVAL_BAR` has why 0.8.
            ema_alpha: smoothing across reset batches. Each call sees only the
                envs resetting on this step.
            dwell_steps: minimum environment steps between promotions. Survival
                at a new rung starts at 1.0 -- nothing has fallen *yet* -- and
                only separates from it once pushes have had time to land, so a
                dwell is what keeps the ladder from climbing on that artifact.
            min_steps: ignore envs whose episode was shorter than this. An env
                that fell in the first second was most likely still settling from
                its reset pose and says nothing about surviving pushes.
            min_samples: pool reset batches until this many valid envs have
                contributed. See the class docstring on why the batch alone is
                never the sample.
            warmup_steps: no promotion before this many environment steps; see
                `WARMUP_STEPS`.
        """
        if not self._applied:
            # On the first call rather than in `__init__`, which avoids depending
            # on the order the managers are constructed in -- and after a resume
            # it re-enters here via `load_state_dict`, which is why `_apply`
            # runs on every call below rather than only on promotion.
            self._apply(env, levels_xy, levels_z)
            self._applied = True

        promoted = 0.0

        # The first reset happens before any stepping; a zero-length episode
        # would read as a survived one.
        if env.common_step_counter > 0 and len(env_ids) > 0:
            steps = env.episode_length_buf[env_ids]
            valid = steps >= min_steps
            n_valid = int(valid.sum())
            if n_valid:
                # An env that timed out outlived every push of its episode; an
                # env that reset early did so through `fell_over`, the only other
                # termination this task has. `episode_length_buf` still holds the
                # ended episode's length here -- the same fact
                # `CommandRangeCurriculum`'s `min_steps` rests on.
                survived = steps[valid] >= env.max_episode_length
                self._pending += float(survived.sum())
                self._pending_n += n_valid

            if self._pending_n >= min_samples:
                ratio = self._pending / self._pending_n
                self.survival_ema = (
                    ratio
                    if self.survival_ema is None
                    else (1.0 - ema_alpha) * self.survival_ema + ema_alpha * ratio
                )
                self._pending = 0.0
                self._pending_n = 0

            if (
                self.survival_ema is not None
                and env.common_step_counter >= warmup_steps
                and env.common_step_counter - self.last_promotion_step >= dwell_steps
                and self.survival_ema >= survival_bar
                and self.level < len(levels_xy) - 1
            ):
                self.level += 1
                self.last_promotion_step = env.common_step_counter
                # The average says nothing about the rung being entered; keeping
                # it would promote again the moment the dwell expired.
                self.survival_ema = None
                promoted = 1.0

            # Idempotent; see the class docstring for why this is every call.
            self._apply(env, levels_xy, levels_z)

        return {
            "level": self.level,
            "push_xy": levels_xy[self.level],
            "push_z": levels_z[self.level],
            "survival": (
                self.survival_ema
                if self.survival_ema is not None
                else float("nan")
            ),
            "promoted": promoted,
        }

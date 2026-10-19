"""The jump command, with the backward pitch the backflip objective measures.

Everything the flight and the height books need is `jumper.ref_free_jump`'s
`RefFreeJumpCommand`; this subclass adds one quantity:

**`pitch_progress`** -- how far the body has rotated backwards about its own
lateral axis since takeoff, in radians, accumulated only while the episode's own
jump is in flight and capped at one full turn. It is the integral of the pitch
angular velocity the `pitch_rate` reward pays for, so the reward and the cap
cannot disagree about how much of the turn has been paid.

**`landing_bonus`** -- `n_feet / 6` for the most feet that were on the ground at
the same time during the `landing_window_s` after the first touchdown, zero
unless the first touchdown was feet-first. The reward manager scales by `dt`,
so the one-shot bonus is emitted as `landing_bonus / dt` on the step the window
closes: the episode then receives exactly `weight * landing_bonus`. Counting at
the exact first-contact step would almost always see one foot, because at 200 Hz
the other five arrive in later steps; the window is what makes "more feet down"
measurable.

The sign is the body-frame one. The base frame is x forward, y left, z up, so a
positive rotation about +y drops the nose and a backflip is **negative** y --
`root_link_ang_vel_b[:, 1]` is negated before it is integrated. Only backward
rotation accumulates; a forward rock does not take progress back, because the
reward is for the backflip and the cap is what stops a multi-turn farm.

**When it accumulates.** The same gate as the reward: in flight, not landed, not
the second jump, and past `min_rise` -- a hop that does not lift the robot is not
the jump and does not pay for rotation. The gate is read before
`RefFreeJumpCommand._update_command` runs, which is the state the rewards were
just computed on. On the takeoff step the flight is not yet latched, so the
first paid step is the one after it; on the touchdown step the flight is still
latched, so the last step of the rotation is paid.

**`jump_done` is the landing alone.** `ref_free_jump` also called the jump done
when the recording's span ran out, because that task ended when the robot was
back at HOME and the recording was its clock. A backflip may take longer than
the straight-jump recording, and the task has no reason to ask for HOME before
the robot has come down: here the jump is done when it has landed, and the
episode otherwise times out. `home_pose` and `back_home` read this property, so
they wait for the landing too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...ref_free_jump.mdp.commands import RefFreeJumpCommand, RefFreeJumpCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class BackflipCommandCfg(RefFreeJumpCommandCfg):
    pitch_turn_rad: float = 2.0 * math.pi
    """How far backwards the pitch reward pays for, in radians. One turn is
    360 degrees; the user's cap."""

    landing_window_s: float = 0.2
    """How long after the first touchdown the feet are counted for the landing
    bonus. 0.2 s is 40 control steps: long enough for a real landing to put its
    feet down, short enough not to become a stance reward."""

    def __post_init__(self):
        super().__post_init__()
        self.class_type = BackflipCommand

    def build(self, env: ManagerBasedRlEnv) -> BackflipCommand:
        return BackflipCommand(self, env)


class BackflipCommand(RefFreeJumpCommand):
    cfg: BackflipCommandCfg

    def __init__(self, cfg: BackflipCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        #: Radians of backward pitch paid for this episode, capped at
        #: `pitch_turn_rad`. Read by `rewards.pitch_rate` before this command's
        #: update, so it is the progress the reward's last step saw.
        self.pitch_progress = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pitch_progress"] = self.pitch_progress

        feet = self._env.scene.sensors[self.cfg.contact_sensor_name].data.force
        assert feet is not None
        self._num_feet = feet.shape[1]
        self._window_steps = max(
            1, round(self.cfg.landing_window_s / self._env.step_dt)
        )
        #: Most feet seen at once during the landing window, and whether the
        #: first touchdown was feet-first. Read by `rewards.landing_feet`.
        self.landing_max_feet = torch.zeros(self.num_envs, device=self.device)
        self._landing_ok = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        #: True for the one step after the window closes: the reward is computed
        #: before this update, so it sees this flag and pays exactly once.
        self.landing_bonus_due = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._landing_started = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._landing_left = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        self.pitch_progress[env_ids] = 0.0
        self.landing_max_feet[env_ids] = 0.0
        self._landing_ok[env_ids] = False
        self.landing_bonus_due[env_ids] = False
        self._landing_started[env_ids] = False
        self._landing_left[env_ids] = 0

    def _update_metrics(self) -> None:
        super()._update_metrics()
        # The base rebinds its own metrics; keep this tensor in the table.
        self.metrics["pitch_progress"] = self.pitch_progress

    @property
    def jump_done(self) -> torch.Tensor:
        """The episode's jump is over when it has landed. See the module
        docstring for why the recording's span is not a second clock here."""
        return self.landed

    @property
    def landing_bonus(self) -> torch.Tensor:
        """`n_feet / 6` for the most feet seen during the landing window, 0
        unless the first touchdown was feet-first."""
        return self.landing_max_feet / float(self._num_feet) * self._landing_ok.float()

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            super()._update_command(env_ids)
            return
        # Before the base latches this step's takeoff/touchdown: the flags and
        # the apex are the ones `pitch_rate` was just computed on.
        self._integrate_pitch()
        super()._update_command(env_ids)
        # After it, `landed` includes this step's touchdown: start the landing
        # window on the touchdown step itself.
        self._update_landing_window()

    def _update_landing_window(self) -> None:
        """Count feet after the first touchdown, and close the one-shot bonus.

        The maximum is kept, not the count at a single step: a six-foot landing
        arrives one foot at a time over several 5 ms steps, and the first step
        with any contact almost always has one foot. "More feet is better" is
        only measurable over a short window.
        """
        # The reward at this step already read the flag the last update set.
        self.landing_bonus_due.zero_()
        feet = self._env.scene.sensors[self.cfg.contact_sensor_name].data.force
        body = self._env.scene.sensors[self.cfg.body_contact_sensor_name].data.force
        assert feet is not None and body is not None
        n_feet = (feet.norm(dim=-1) > self.cfg.force_threshold).sum(dim=1).float()
        body_down = (body.norm(dim=-1) > self.cfg.force_threshold).any(dim=1)

        first = self.landed & ~self._landing_started
        self._landing_started |= first
        self._landing_left[first] = self._window_steps
        self._landing_ok[first] = ~body_down[first]
        self.landing_max_feet[first] = 0.0

        # Running windows, including one that started on this step. After the
        # last step `_landing_left` is 0 and `active` is false from then on, so
        # the bonus is flagged exactly once.
        active = self._landing_started & (self._landing_left > 0)
        self.landing_max_feet.copy_(
            torch.where(active, torch.maximum(self.landing_max_feet, n_feet),
                        self.landing_max_feet)
        )
        self._landing_left -= active.long()
        self.landing_bonus_due |= active & (self._landing_left == 0)

    def _integrate_pitch(self) -> None:
        """Add this step's backward rotation to `pitch_progress`, capped.

        The reward pays for the same clipped increment, so the two cannot drift
        by a step or by the cap. `omega` is the body-frame y rate with the
        backflip's sign; a forward rate contributes nothing.
        """
        omega = -self.robot.data.root_link_ang_vel_b[:, 1]
        paying = (
            self.in_flight
            & ~self.landed
            & ~self.second_jump
            & (self.apex_rise > self.cfg.min_rise)
        )
        delta = omega.clamp(min=0.0) * self._env.step_dt
        remaining = (self.cfg.pitch_turn_rad - self.pitch_progress).clamp(min=0.0)
        self.pitch_progress.add_(torch.minimum(delta, remaining) * paying.float())


__all__ = ["BackflipCommand", "BackflipCommandCfg"]

"""The two reward terms the backflip adds to the reference-free jump's objective.

Both are one-shot or capped, so neither can be farmed:

- `pitch_rate` pays the backward pitch angular velocity while the robot is in
  its own jump. The reward manager multiplies every term by `dt`, so the
  episode's total is the angle turned, and the rate is clipped to the part of
  the turn still unpaid: the total stops at one full turn
  (`BackflipCommandCfg.pitch_turn_rad`, 360 degrees) and the term is zero after
  that. This is a rate reward whose **total** is capped by the rotation, not a
  per-step cap on the rate -- a fast flip earns the same total as a slow one
  that completes the turn, but earns it sooner.
- `landing_feet` pays once, when the command's landing window closes:
  `n_feet / 6` for the most feet that were on the ground at the same time
  during the window, and only when the first touchdown was feet-first. A
  back-first landing pays nothing; a six-foot landing pays the term's full
  weight. This is the "feet first, more feet is better" term. The bonus is
  emitted as `landing_bonus / dt` so that, after the manager's `dt` scaling,
  the episode receives exactly `weight * landing_bonus`.

The landing event is the command's own touchdown, with the same descending gate
on body contact: a calf or palm touching while the base is still rising is the
push's drag, not a landing.
"""

from __future__ import annotations

import torch


def pitch_rate(env, command_name: str = "jump") -> torch.Tensor:
    """Backward pitch angular velocity, in rad/s, while the episode's jump is in
    flight; clipped so the whole episode pays for at most one turn.

    The body frame is x forward, y left, z up: a backflip (nose up) is a
    negative rotation about y, so the y rate is negated. Only backward rotation
    is paid for and only backward rotation advances the cap; a forward rock does
    not earn and does not give progress back. The reward manager scales by
    `dt`, so returning a rate makes the episode's total
    `weight * (angle turned)`, and the clip removes any part of a step that
    would take the paid angle past `pitch_turn_rad`.
    """
    cmd = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    omega = (-robot.data.root_link_ang_vel_b[:, 1]).clamp(min=0.0)
    paying = (
        cmd.in_flight
        & ~cmd.landed
        & ~cmd.second_jump
        & (cmd.apex_rise > cmd.cfg.min_rise)
    )
    remaining_rate = (cmd.cfg.pitch_turn_rad - cmd.pitch_progress).clamp(min=0.0)
    remaining_rate = remaining_rate / env.step_dt
    return torch.minimum(omega, remaining_rate) * paying.float()


def landing_feet(env, command_name: str = "jump") -> torch.Tensor:
    """The landing bonus, paid once on the step the command's landing window
    closes: `n_feet / 6` for the most feet on the ground together, or 0.

    The command starts the window at the first touchdown of the episode's jump,
    keeps the maximum number of feet seen, and disqualifies a landing whose
    first touchdown had any non-foot part down. A hop below `min_rise` does not
    count. The one-shot value is divided by `dt` because the reward manager
    scales by `dt`: the episode receives `weight * n_feet / 6`, not a rate.
    """
    cmd = env.command_manager.get_term(command_name)
    return cmd.landing_bonus / env.step_dt * cmd.landing_bonus_due.float()


__all__ = ["landing_feet", "pitch_rate"]

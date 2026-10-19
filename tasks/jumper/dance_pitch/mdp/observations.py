"""Observation terms specific to jumper.dance_pitch.

One term: the dance clock, as sin/cos pairs -- `jumper.dance_sway`'s lesson
carried over: a reward scored against a clock the policy was never shown is
not learnable. Two clocks, because the loop is 9 s and the wobble 0.5 s and
the second is not a multiple of the first.
"""

from __future__ import annotations

import math

import torch
from mjlab.envs import ManagerBasedRlEnv

from .rewards import FAST_OMEGA, LOOP_PERIOD


def pitch_phase_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[sin, cos of the 9 s loop clock; sin, cos of the 0.5 s wobble clock].

    Four values, the same shape as `sway_phase_obs` -- deliberate, so a
    `dance_sway` checkpoint loads into this task's runner unchanged (the
    standing base warm start).
    """
    t = env.episode_length_buf.to(torch.float32) * env.step_dt
    loop = 2.0 * math.pi * t / LOOP_PERIOD
    fast = FAST_OMEGA * t
    return torch.stack(
        (torch.sin(loop), torch.cos(loop), torch.sin(fast), torch.cos(fast)),
        dim=-1,
    )

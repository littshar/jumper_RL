"""Observation terms specific to jumper.dance_sway.

One term: the dance clock, as sin/cos pairs. `jumper.tripod`'s lesson -- the
gait reward is unlearnable unless the policy is shown the same clock the
reward scores against -- applied to a parametric dance: the references are
functions of episode time, and this hands the phase over.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

from .rewards import CLAW_FREQ_MULT, sway_phase


def sway_phase_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[sin(phase), cos(phase), sin(4*phase), cos(4*phase)] -- both clocks.

    The claw runs at a multiple of the sway, so its phase is derivable from
    the sway's; both are handed over anyway, because the policy's job is to
    lock onto a rhythm, not to do arithmetic on inputs it barely sees.
    """
    phase = sway_phase(env, 1.0)
    claw = sway_phase(env, CLAW_FREQ_MULT)
    return torch.stack(
        (torch.sin(phase), torch.cos(phase), torch.sin(claw), torch.cos(claw)),
        dim=-1,
    )

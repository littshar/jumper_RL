"""jumper.stand -- hold a two-leg rearing stance under pushes.

The robot stands on its LR/RR rear pair, body pitched nose-up 55 degrees so
the centre of mass returns over the rear support line, claws and middle legs
tucked clear -- a crab standing up on its hind legs. Episodes start already in
the stance; getting up *into* it from six feet is the follow-up task, trained
against a policy that can already hold what it reaches for. The command
ceiling is zero, and `push_robot` shoves it on the skeleton's 1-3 s clock
through a curriculum that widens the shove as survival at the current width
improves.

The support polygon is a line: roll has the feet's 0.318 m of separation under
it, pitch is an inverted pendulum with nothing -- and attitude itself is part
of the pose, so both the attitude reward and the fall termination are measured
against the stance's own reference orientation, not against gravity.

The honest baseline is a constant PD hold at the stance, which the strength
budget says is holdable indefinitely (worst joint at 82% of the continuous
rating) but says nothing about recovering. What an RL policy has to add is
exactly that region: pushes big enough that holding requires moving. The
reward closes every exit -- crouching, drifting, resting a tucked leg --
except the one that is wanted, which is catching with the legs.

The robot definition and the environment skeleton live in `tasks/jumper/common/`.
This directory holds only what belongs to this task: its fixed name (the register
call below), its environment config (`env_cfg.py`), its hyper-parameters
(`rl_cfg.py`), and the MDP terms only it uses (`mdp/`).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.stand",
    assets=JUMPER_ASSETS,
    description="jumper hexapod rears up on its hind legs (LR/RR) and holds it against pushes",
    tags=("jumper", "stand", "balance"),
)

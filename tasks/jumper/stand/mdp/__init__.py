"""MDP terms specific to jumper.stand.

The sit-up's reward terms and the push curriculum. Everything else this task
does is a re-aiming of terms the skeleton already has -- see the notes in
`../env_cfg.py`.
"""

from . import curriculum, rewards

__all__ = ["curriculum", "rewards"]

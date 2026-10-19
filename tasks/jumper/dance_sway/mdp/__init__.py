"""MDP terms specific to jumper.dance_sway.

The sway's clock, references and tracking terms, plus the phase observation.
Everything else this task does is a re-aiming of terms the skeleton already
has -- see the notes in `../env_cfg.py`.
"""

from . import observations, rewards

__all__ = ["observations", "rewards"]

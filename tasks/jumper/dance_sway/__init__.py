"""jumper.dance_sway -- standing sway with a waving claw.

A parametric dance, no reference clip, third revision: standing on the four
legs (claws posed in the air), the body translates side-to-side on a 2-second
rhythm shaped near-square but gentle, the right claw's wrist joint (RF_J3)
waves in the band 0.8-1.3 rad at four times the sway frequency, and roll
wobble within 5 degrees rides along uncharged.

The operator's spec, verbatim (posed in the puppet window; all angles MuJoCo
joint angles, radians): lm的j0在-0.6附近，rm的j0在0.6附近，lf的j0在-1左右，j1
在-2左右，rf的j1在-1.5左右，摆动的是rf的j3，在0.8到1.3左右摆动，机体左右摆动
需要lm，rm，lr，rr四条腿协调平移摆动，roll轴可以5度以内跟随晃动，左右平移幅
度30cm左右，可以适当缩小，lm和rm的j0为了平衡作了调整不须大幅改动，左右移动的
波形接近方波但可以平缓一点，rf的j3摆动周期0.5s左右，身体摆动周期2s，其他没说
的的角度按木偶的初始角度来。On the two open points the operator chose: the feet
stay planted (脚全程着地) and the amplitude is as large as planted legs allow
(不抠数字，尽量大).

The robot definition and the environment skeleton live in `tasks/jumper/common/`.
This directory holds only what belongs to this task: its fixed name (the register
call below), its environment config (`env_cfg.py`), its hyper-parameters
(`rl_cfg.py`), and the MDP terms only it uses (`mdp/`).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.dance_sway",
    assets=JUMPER_ASSETS,
    description="jumper hexapod sways side-to-side (2 s period, square-ish) while the right claw waves at 4x frequency",
    tags=("jumper", "dance", "periodic"),
)

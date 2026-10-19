"""jumper.dance_pitch -- the nodding-pitch dance on spread middle legs.

A parametric dance, no reference clip, first revision: standing on the four
legs with the middle hips spread (LM/RM J0 at -0.5/+0.5), both claws posed in
the air (LF/RF J2 at -1/+1, J3 at -2.1/+2.1), both shoulders wave (LF_J1
around -0.6, RF_J1 around -1.0, +-5 degrees at a 0.5 s period, lagging the
body wave's sine by 90 degrees), and the body's pitch follows a 9-second
piecewise loop, positive = nose up:

    theta_p(t) in degrees, t = episode time mod 9 s:
        20 + 5 sin(4 pi t)              0 <= t < 3
        20 - 30 (t - 3)                 3 <= t < 4
        -10 + 15 (t - 4)                4 <= t < 5
        5 + 3.75 (t - 5) + 5 sin(4 pi (t - 5))   5 <= t <= 9

The curve is continuous at every joint and closes on itself at the loop point
(theta(9) = theta(0) = 20 deg), so the loop needs no blending.

The operator's spec, verbatim (all joint angles MuJoCo radians): lm的j0为-0.5,
rm的j0为0.5,lf的j3为-2.1,rf的j3为2.1,lf的j2为-1，rf的j2为1,lf和rf的j1在-0.6和
-1附近以5度振幅0.5s周期摆动，身体的pitch角贴合[the piecewise function above]
这个分段函数，以9s为一次循环，一直循环，训练用原来的红色模型，其他和上次训练
舞蹈一样。On the open points the operator chose: pitch positive is nose up
(正 = 前部抬高), both shoulders swing in phase lagging the sine by 90 degrees
(两臂同向，最好滞后sin那个函数90度), training stays on flat ground (一直用平地),
and training runs on the original red model asset while replay keeps the
decorated one.

The robot definition and the environment skeleton live in `tasks/jumper/common/`.
This directory holds only what belongs to this task: its fixed name (the
register call below), its environment config (`env_cfg.py`), its
hyper-parameters (`rl_cfg.py`), and the MDP terms only it uses (`mdp/`).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.dance_pitch",
    assets=JUMPER_ASSETS,
    description="jumper hexapod nods through a 9 s piecewise pitch loop (25 to -10 deg) "
                "on spread middle legs while both shoulders wave at 2 Hz",
    tags=("jumper", "dance", "periodic"),
)

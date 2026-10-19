"""MDP terms specific to jumper.dance_pitch.

The pitch loop's clock, its reference signals, and the tracking terms. Like
`jumper.dance_sway`, the dance is parametric -- references are functions of
episode time, not a clip -- so the clock here is what a reference clip would
have been, and the policy sees it through `mdp/observations.py`'s phase
observation.

Terms this dance shares unchanged with `jumper.dance_sway` (`foreaft_still`,
`yaw_still`, `support_foot_load`, `hold_arms`) are imported from there rather
than duplicated: `jumper.dance_sway` itself imports `foot_pos_b` from
`jumper.posture`, so a task importing another task's MDP term is this
repository's precedent, not a new one. They move to `common/` when a third
formula dance wants them.
"""

from __future__ import annotations

import math

import torch
from mjlab.envs import ManagerBasedRlEnv

#: The loop's period, in seconds -- the operator's spec (以9s为一次循环).
LOOP_PERIOD = 9.0

#: The fast wobble's angular frequency. sin(4*pi*t) is the operator's 0.5 s
#: period, in both the pitch reference's sine terms and the shoulders' wave.
FAST_OMEGA = 4.0 * math.pi

#: The shoulders' wave amplitude: the operator's 5 degrees, in radians.
J1_AMPLITUDE = math.radians(5.0)


def dance_time(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Episode time folded into [0, LOOP_PERIOD), [num_envs].

    Episode time, not global time, for the same reason `sway_phase` gives:
    the runner randomises initial episode lengths, so each environment's own
    clock is the only one the reference can be defined from. The loop is
    closed (theta(9) = theta(0)), so folding costs nothing at the seam.
    """
    t = env.episode_length_buf.to(torch.float32) * env.step_dt
    return torch.remainder(t, LOOP_PERIOD)


def pitch_ref(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The body's pitch reference, radians, **positive = nose up**.

    The operator's piecewise curve, verbatim in degrees (see the task's
    `__init__.py`), converted here. Continuous at every segment boundary and
    at the loop seam; the *velocity* jumps at t=0 (the wobble's sine switches
    on at full slope), which is the spec's shape and only ever prices the
    position-tracking terms.
    """
    t = dance_time(env)
    seg1 = 20.0 + 5.0 * torch.sin(FAST_OMEGA * t)
    seg2 = 20.0 - 30.0 * (t - 3.0)
    seg3 = -10.0 + 15.0 * (t - 4.0)
    seg4 = 5.0 + 3.75 * (t - 5.0) + 5.0 * torch.sin(FAST_OMEGA * (t - 5.0))
    deg = torch.where(
        t < 3.0, seg1, torch.where(t < 4.0, seg2, torch.where(t < 5.0, seg3, seg4))
    )
    return torch.deg2rad(deg)


def j1_ref(env: ManagerBasedRlEnv, center: float) -> torch.Tensor:
    """A shoulder's reference, radians: `center + 5deg * sin(4 pi t - pi/2)`.

    The operator's lag: the shoulders' wave trails the pitch reference's
    `sin(4 pi t)` by 90 degrees, both shoulders in phase (两臂同向).
    `sin(x - pi/2)` is `-cos(x)`, continuous across the loop seam with the
    same 0.5 s period.
    """
    t = dance_time(env)
    return center + J1_AMPLITUDE * torch.sin(FAST_OMEGA * t - math.pi / 2.0)


def _gravity_b(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Unit gravity in the body frame, [num_envs, 3]. Normalised defensively:
    the contract of `projected_gravity_b` is a unit vector, but the sine
    below is only exact if it really is one."""
    g = env.scene["robot"].data.projected_gravity_b
    return g / torch.linalg.norm(g, dim=-1, keepdim=True)


def measured_pitch(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The body's pitch, radians, positive = nose up.

    From the body-frame gravity: level, g_b is (0, 0, -1); nose up by theta
    tips the body's x axis above the horizon and puts -sin(theta) into g_b.x.
    Verified against direct quaternion rotation, both signs, before wiring:
    +20 deg for a -20 deg rotation about +y with +x forward (the robot's
    forward is +x -- the claws sit at x ~ +0.12, the rear feet at -0.15).
    """
    return torch.asin((-_gravity_b(env)[:, 0]).clamp(-1.0, 1.0))


def measured_roll(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The body's roll, radians, from the same gravity vector. The sign
    convention is irrelevant here: every consumer targets zero, and an even
    kernel cannot tell a mirror from the original."""
    return torch.asin(_gravity_b(env)[:, 1].clamp(-1.0, 1.0))


def track_pitch(
    env: ManagerBasedRlEnv, std: float
) -> torch.Tensor:
    """`exp(-((pitch - pitch_ref) / std)^2)` -- the loop, scored each step."""
    err = pitch_ref(env) - measured_pitch(env)
    return torch.exp(-err.square() / std**2)


def pitch_pull(env: ManagerBasedRlEnv) -> torch.Tensor:
    """`|pitch_ref - pitch|` -- the constant-gradient backstop, `sway_pull`'s
    pattern: the exp kernel's gradient dies once the error leaves `std`, and
    this dance's ramp segments swing 30 deg/s, so an untracked ramp must keep
    paying. The weight carries the minus sign; this returns a magnitude."""
    return (pitch_ref(env) - measured_pitch(env)).abs()


def track_j1(
    env: ManagerBasedRlEnv,
    joint_names: tuple[str, ...],
    centers: tuple[float, ...],
    std: float,
) -> torch.Tensor:
    """`mean(exp(-((q - j1_ref) / std)^2))` over the two shoulders.

    One term for both shoulders because they are one wave (same clock, same
    amplitude) at two centres; scoring them separately would only double the
    bookkeeping. Joints by name, not index, for the reason `track_claw`
    gives: a wave trained into the wrong joint of an identical-looking
    network is silent.
    """
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    idx = [names.index(n) for n in joint_names]
    q = robot.data.joint_pos[:, idx]
    refs = torch.stack(
        [j1_ref(env, center) for center in centers], dim=-1
    )
    return torch.exp(-((q - refs) / std).square()).mean(dim=1)


def roll_level(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
    """`exp(-(roll / std)^2)` -- what `upright` was, rebuilt roll-only.

    The skeleton's `upright` scores sin(total tilt); this dance commands 25
    degrees of pitch, at which `upright` reads exp(-(0.42/std)^2) ~ 0 and
    would charge the whole performance. Roll is the one rotation nothing in
    the choreography commands, so it takes the ruler alone.
    """
    return torch.exp(-measured_roll(env).square() / std**2)

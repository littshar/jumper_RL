"""Reward terms specific to jumper.stand.

The task is a **sit-up**: from the standing HOME stance, tip the whole body
back until the butt grounds, and hold it -- a dog sitting up. The operator
measured that the HOME joint pose *is* a stable sit once the body is leaned
back onto its rear, so the joints never need to leave HOME; what the policy
learns is a whole-body rotation of the floating base and then stillness.

Everything here is built around one measurement channel the locomotion tasks
don't have: **the butt's contact with the ground**. `butt_ground_contact`
covers the chassis geoms that can ground (base link and upper shell), and the
terms below read it by slot.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

#: Columns of the `feet_ground_contact` sensor for the stance pair, in `LEGS`
#: order (LF, RF, LM, RM, LR, RR).
_STANCE_COLS = (4, 5)


def _foot_force(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Per-foot contact-force magnitudes, [num_envs, 6] in `LEGS` order."""
    sensor = env.scene.sensors[sensor_name]
    force = sensor.data.force
    assert force is not None, f"sensor {sensor_name!r} has no force field"
    return torch.linalg.norm(force, dim=-1)


def _butt_force(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Total chassis-ground contact force, [num_envs]."""
    sensor = env.scene.sensors[sensor_name]
    force = sensor.data.force
    assert force is not None, f"sensor {sensor_name!r} has no force field"
    return torch.linalg.norm(force, dim=-1).sum(dim=-1)


def butt_load(
    env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float = 8.0
) -> torch.Tensor:
    """Reward the butt for carrying load on the ground.

        reward = min(force / threshold, 1)

    The objective of the whole task, in one channel. The threshold is a third
    of the robot's weight: a settled sit carries nearly all 24.95 N through
    the chassis (the rest through the rear feet), so a third is comfortably
    above "brushing the ground" and the ramp saturates well before the full
    sit -- the term says *down*, not *how hard*.
    """
    return (_butt_force(env, sensor_name) / force_threshold).clamp(0.0, 1.0)


def biped_stance_load(
    env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float = 6.0
) -> torch.Tensor:
    """Reward the two rear feet for staying planted.

        reward = mean over LR, RR of min(force / threshold, 1)

    The rear legs are the other half of the operator's spec: rear feet AND
    butt on the ground. Saturates at half a fair share, for the same reason
    the six-foot term's 3 N does.
    """
    force = _foot_force(env, sensor_name)
    loaded = (force[:, _STANCE_COLS] / force_threshold).clamp(0.0, 1.0)
    return loaded.mean(dim=1)


def sit_pitch(
    env: ManagerBasedRlEnv,
    sit_from_vertical: float,
    std: float,
) -> torch.Tensor:
    """`exp(-((tilt-from-vertical - target) / std)^2)`, yaw-invariant.

    The sit is a *specific* lean-back angle, and unlike the rearing stance it
    is measured against gravity directly: head-up means the body's tilt from
    vertical lands near the sit angle. The skeleton's `upright` scores the
    same angle against zero and would pay the policy to stay standing instead;
    this term pays for arriving at the lean. Yaw never enters -- the tilt is
    read off the projected gravity vector's z component, which no yaw can
    change.
    """
    robot = env.scene["robot"]
    pg = robot.data.projected_gravity_b
    tilt = torch.acos((-pg[:, 2]).clamp(-1.0, 1.0))
    err = tilt - sit_from_vertical
    return torch.exp(-err.square() / std**2)


def seated_still(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 8.0,
) -> torch.Tensor:
    """How much the body moves **while seated**, squared and gated.

        cost = (|v|^2 + |w|^2) * min(butt force / threshold, 1)

    `standing_sway`'s inversion: that term prices motion while standing, this
    one while seated, and the gate is the butt contact rather than the command
    (there is no command). Its job is to stop the policy scooting around on
    its butt once the sit pays -- the transition itself is never charged,
    because while the butt is up the gate is closed.
    """
    asset = env.scene["robot"]
    gate = (_butt_force(env, sensor_name) / force_threshold).clamp(0.0, 1.0)
    lin = asset.data.root_link_lin_vel_b
    ang = asset.data.root_link_ang_vel_b
    return (lin.square().sum(dim=1) + ang.square().sum(dim=1)) * gate


def hold_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
    target: float,
) -> torch.Tensor:
    """`exp(-((height - target) / std)^2)`, the base above the rear feet.

    The height is `posture.mdp.state.body_height` -- the base above the mean
    of the foot sites named by `asset_cfg`, plus `FOOT_SITE_Z`. Given the sit
    target as a pitch and a butt contact, the height is what separates
    *sitting up* from *slumping back onto the shell*: both ground the butt,
    only one keeps the body lifted on the rear legs. The target is the
    measured height of the settled sit, quoted in `env_cfg.py`.
    """
    from ...posture.mdp.state import body_height

    err = target - body_height(env, asset_cfg)
    return torch.exp(-err.square() / std**2)

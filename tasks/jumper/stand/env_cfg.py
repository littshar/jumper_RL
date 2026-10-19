"""Environment config for jumper.stand.

**A sit-up**: episodes start standing at HOME on all limbs, and the policy
tips the whole body back until the butt grounds -- rear feet and chassis on
the ground, torso head-up, front claws in the air -- and holds it. The
operator measured the crucial fact this task is built on: **the HOME joint
pose is already a stable sit** once the body is leaned onto its rear. The
joints therefore never need to leave HOME (`pose`, at the standing width,
holds them there and is an objective rather than a compromise); what is
learned is a whole-body rotation of the floating base, and stillness after.

## What that removes

Everything the rearing design needed and the sit does not:

    init override      episodes spawn at the skeleton's own HOME stance --
                       "from four legs" is where the robot already is
    toppled            termination against a reference quaternion; the sit
                       is upright-ish, so the skeleton's `fell_over` works
                       again, widened to fire only past horizontal
    hold_tilt          the sit's attitude target is gravity-referenced and
                       yaw-free, so a plain projected-gravity kernel does it
    tucked_feet_clear  the front four limbs are explicitly FREE -- the
                       operator wants them available for dance/gesture work;
                       nothing scores their pose
    load_balance       the butt + two rear feet make a real support polygon;
                       roll stability is geometry now, not a reward

## What measures the sit

`butt_ground_contact`, a new sensor over the chassis geoms that can ground
(base link, upper shell), is the task's spine, and three terms read it:

    butt_load     the objective: chassis carrying weight, ramp at a third of
                  body weight
    seated_still  no scooting once seated -- (|v|^2 + |w|^2) gated by the
                  butt contact, so the transition is never charged
    hold_height   what separates sitting UP from slumping onto the shell:
                  the base stays lifted on the rear legs at the measured sit
                  height

plus `biped_stance_load` (rear feet planted, the operator's "后两条腿") and
`sit_pitch` (the lean-back angle, gravity-referenced, yaw-free).

## What is inherited

The command ceiling is zero, so the tracking terms -- stds pinned below, at
the zero ceiling the skeleton's ratio rule would give 0 -- are drift rulers:
2.0 each for genuinely standing still, decaying smoothly into the drift
penalty. `standing_sway` is switched OFF here (weight 0): it charges exactly
the motion the transition needs. The gait terms (`foot_clearance`,
`foot_swing_height`, `foot_slip`, `soft_landing`) go to weight 0 -- the
transition drags feet, and nothing should price that. `energy` and
`torque_headroom` carry over from the posture task's measured pair; `pose`
holds the joints at HOME, which is the sit.

## The push ladder

`mdp/curriculum.py` widens `push_robot` as survival improves, at
+/-0.02 / +/-0.04 / +/-0.07 m/s. The command curriculum is removed rather
than retuned -- its `_apply` would write `lin_std(0) = 0` into the live
tracking stds every step, silently replacing the rulers set below.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ..common.actuator import CONTINUOUS_TORQUE
from ..common.constants import EFFORT_LIMIT, GAIT_JOINTS
from ..common.mdp.curriculum import STD_ANG_RATIO, STD_LIN_RATIO
from ..common.mdp.rewards import actuator_headroom, actuator_power
from ..common.velocity_env import velocity_env_cfg
from .mdp.curriculum import (
    PUSH_LEVELS_XY,
    PUSH_LEVELS_Z,
    SURVIVAL_BAR,
    WARMUP_STEPS,
    PushStrengthCurriculum,
)
from .mdp.rewards import (
    biped_stance_load,
    butt_load,
    hold_height,
    seated_still,
    sit_pitch,
)

#: The lean-back the sit settles at, measured by settling the operator-posed
#: capture in this env with HOME actions -- radians from vertical, patched
#: below once the probe has run. `sit_pitch`'s target and `fell_over`'s
#: headroom both read from this.
SIT_FROM_VERTICAL = math.radians(60.0)

#: The base height of the settled sit, above the rear feet. `hold_height`'s
#: target; patched alongside the pitch after the probe.
SIT_HEIGHT = 0.107

#: `sit_pitch`'s std, in radians. The settled sit is measured at one
#: configuration; a real sit has posture in it. 0.15 rad (~8.6 deg) accepts
#: the slump-free band around the measured angle.
PITCH_STD = 0.15

#: `hold_height`'s std, in metres. 15 mm -- the sit's height is a posture,
#: not a set point, and the rear legs hold it with real authority once the
#: butt is grounded.
HEIGHT_STD = 0.015

#: The chassis ramp's saturation, in newtons: a third of body weight.
#: `butt_load` has the reasoning.
BUTT_FORCE_THRESHOLD = 8.0

#: The rear-feet ramp's saturation, in newtons: half a fair share.
STANCE_FORCE_THRESHOLD = 6.0

#: `fell_over`'s limit, from vertical. The sit reaches ~60-70 deg from
#: vertical by design -- the skeleton's 50 deg would fire mid-maneuver and on
#: every seated env. 100 deg is past horizontal: only a genuine tip-over ends
#: the episode, and everything short of it is charged by the reward instead
#: of deleted from it.
FELL_LIMIT = math.radians(100.0)

#: The drift ruler, in m/s.
DRIFT_STD = 0.05

#: The yaw-drift ruler, in rad/s.
YAW_DRIFT_STD = 0.10


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config.

    Args:
        asset: model XML path, from `--model`. None uses the default jumper.xml.
        play: replay mode -- observation noise and external disturbances off,
            longer episodes.
    """
    cfg = velocity_env_cfg(
        asset=asset,
        play=play,
        # No one drives this task -- there is nothing to drive -- so no
        # controls.yaml, no pad, no keyboard. `jumper.swing` is the precedent;
        # the skeleton raises only when a command is kept for an operator.
        operator_command=False,
        teleop=False,
        # Required by the skeleton even though every term that reads it is set
        # to weight 0 below; tripod's 0.025 written so the raise has an answer,
        # inert -- it reaches two terms with weight 0.
        foot_target_height=0.025,
        # The joints stay at HOME through the whole task, so the barrier the
        # locomotion tasks fight -- pose + action_rate paying for stillness --
        # is this task's friend and is kept at the flat reference values.
        action_rate_weight=-0.1,
        pose_weight=1.0,
        # The ceilings are the task: the command samples as identically zero.
        # The stds the skeleton derives become 0; the overrides below replace
        # them. The std ratios satisfy the raise and are then dead.
        command_lin_ceiling=0.0,
        command_ang_ceiling=0.0,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
        # The transition is a whole-body motion; the sway term would charge
        # exactly that. Off; `seated_still` takes the seated half of its job.
        standing_sway_weight=0.0,
    )

    # ── The rulers ────────────────────────────────────────────────────────
    cfg.rewards["track_linear_velocity"].params["std"] = DRIFT_STD
    cfg.rewards["track_angular_velocity"].params["std"] = YAW_DRIFT_STD

    # ── Attitude: the skeleton's upright fights the sit ───────────────────
    # `upright` scores alignment with gravity, and the sit leans 60-70 deg
    # away from it -- the skeleton's own term would pay the policy to stay
    # standing. Replaced by `sit_pitch`, whose target IS the lean.
    cfg.rewards.pop("upright", None)
    cfg.rewards["sit_pitch"] = RewardTermCfg(
        func=sit_pitch,
        weight=1.0,
        params={"sit_from_vertical": SIT_FROM_VERTICAL, "std": PITCH_STD},
    )

    # ── The gait terms, off ───────────────────────────────────────────────
    # The transition drags feet; nothing should price a moving foot.
    for _name in (
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
    ):
        cfg.rewards[_name].weight = 0.0

    # ── The sit's own sensor ──────────────────────────────────────────────
    # Chassis geoms that can ground, against the terrain. The feet sensor's
    # wiring, aimed at the shell.
    butt_sensor = ContactSensorCfg(
        name="butt_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=("base_link_meshcol", "upper_shell_link_meshcol"),
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (butt_sensor,)
    sensor_params = {"sensor_name": "butt_ground_contact"}

    # ── What it is paid for ───────────────────────────────────────────────
    cfg.rewards["butt_load"] = RewardTermCfg(
        func=butt_load,
        weight=1.0,
        params=sensor_params,
    )
    cfg.rewards["seated_still"] = RewardTermCfg(
        func=seated_still,
        weight=-0.1,
        params=sensor_params,
    )
    cfg.rewards["biped_stance_load"] = RewardTermCfg(
        func=biped_stance_load,
        weight=0.5,
        params={
            "sensor_name": "feet_ground_contact",
            "force_threshold": STANCE_FORCE_THRESHOLD,
        },
    )
    cfg.rewards["hold_height"] = RewardTermCfg(
        func=hold_height,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=("LR", "RR"), preserve_order=True
            ),
            "std": HEIGHT_STD,
            "target": SIT_HEIGHT,
        },
    )

    # ── And what the actuators spend ──────────────────────────────────────
    # The posture task's measured pair, carried over unchanged.
    actuator_cfg = SceneEntityCfg("robot", actuator_names=list(GAIT_JOINTS))
    cfg.rewards["energy"] = RewardTermCfg(
        func=actuator_power,
        weight=-0.02,
        params={"asset_cfg": actuator_cfg},
    )
    cfg.rewards["torque_headroom"] = RewardTermCfg(
        func=actuator_headroom,
        weight=-1.0,
        params={
            "deadband": CONTINUOUS_TORQUE / EFFORT_LIMIT,
            "asset_cfg": actuator_cfg,
        },
    )

    # ── Termination: past horizontal, not mid-maneuver ────────────────────
    # `fell_over` at its skeleton 50 deg would fire on every seated env; the
    # module docstring has the numbers. Widened in place.
    cfg.terminations["fell_over"].params["limit_angle"] = FELL_LIMIT

    # ── The curricula ─────────────────────────────────────────────────────
    # The command ladder is removed, not retuned: its `_apply` would write
    # `lin_std(0) = 0` into the live tracking stds every step.
    if "command" in cfg.curriculum:
        del cfg.curriculum["command"]

    if not play:
        cfg.curriculum["push"] = CurriculumTermCfg(
            func=PushStrengthCurriculum,
            params={
                "levels_xy": PUSH_LEVELS_XY,
                "levels_z": PUSH_LEVELS_Z,
                "survival_bar": SURVIVAL_BAR,
                "warmup_steps": WARMUP_STEPS,
                # The flat baseline's rollout is 24, so 24 * 100 is 100
                # iterations; written here so the pacing is a decision.
                "dwell_steps": 24 * 100,
            },
        )

    return cfg

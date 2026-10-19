"""Environment config for jumper.dance_pitch.

The nodding-pitch dance, first revision, as the operator specified it:
standing on the four legs with the middle hips spread, both claws posed in
the air, both shoulders waving at 2 Hz, and the body pitching through a
9 s piecewise loop (see the package's `__init__.py` for the curve).

Built on `jumper.dance_sway`'s skeleton, and the load-bearing choices are
that task's: the references are deterministic functions of episode time,
scored by exp-kernel tracking terms plus a constant-gradient pull, and handed
to the policy as a phase observation; the skeleton's whole-velocity drift
rulers are replaced by axis-only ones; the command curriculum is removed
rather than retuned. What differs is listed here so a reader of both files
sees the delta and nothing else:

- **`upright` is off and `roll_level` is on.** The dance commands 25 degrees
  of pitch; `upright` scores sin(total tilt) and would charge the entire
  performance. Roll is the one rotation nothing commands, so it takes the
  ruler alone.
- **`edge_height` is gone.** Pitching past 20 degrees moves the root height
  by geometry; a height ruler at the dwell would bid against the choreography.
- **The arms' pose stds are split.** The shoulders (J1) are the wave -- loose
  in the pose term, priced by `track_j1` -- while every other arm joint is
  tight and held by `hold_arms`.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..common.actuator import CONTINUOUS_TORQUE
from ..common.constants import EFFORT_LIMIT, GAIT_JOINTS, HOME
from ..common.mdp.curriculum import STD_ANG_RATIO, STD_LIN_RATIO
from ..common.mdp.rewards import actuator_headroom, actuator_power
from ..common.velocity_env import velocity_env_cfg
from ..dance_sway.mdp.rewards import (
    foreaft_still,
    hold_arms,
    support_foot_load,
    yaw_still,
)
from .mdp.observations import pitch_phase_obs
from .mdp.rewards import (
    pitch_pull,
    roll_level,
    track_j1,
    track_pitch,
)

#: The standing pose: the operator's holds written over HOME. The shoulder
#: centres are the wave's centres, so the spawn, the action centre and the
#: pose term's target all agree with where the wave oscillates around.
DANCE_HOME: dict[str, float] = {
    **HOME,
    "LM_J0_joint": -0.5,
    "RM_J0_joint": 0.5,
    "LF_J1_joint": -0.6,
    "RF_J1_joint": -1.0,
    "LF_J2_joint": -1.0,
    "RF_J2_joint": 1.0,
    "LF_J3_joint": -2.1,
    "RF_J3_joint": 2.1,
}

#: The waving shoulders and their centres (operator's: LF around -0.6, RF
#: around -1.0; amplitude 5 deg at 0.5 s period lives in `mdp/rewards.py`).
J1_JOINTS = ("LF_J1_joint", "RF_J1_joint")
J1_CENTERS = (-0.6, -1.0)

#: Tracking stds, radians. The exp kernel's gradient dies once the error
#: leaves `std`; each ruler sits near half the motion it prices -- the pitch
#: loop spans 35 degrees and swings 30 deg/s on its ramps, the shoulders
#: wave 10 degrees peak-to-peak. `pitch_pull` is the constant-gradient
#: backstop that makes the exact sigma less load-bearing.
PITCH_STD = 0.10
J1_STD = 0.08

#: The roll ruler, radians -- `dance_sway` carried `upright`'s 0.10 over as
#: its "roll may wobble within ~5.7 degrees" sanction; same number, now
#: scoring roll alone.
ROLL_STD = 0.10


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
        # No one drives this task -- the dance is the program.
        operator_command=False,
        teleop=False,
        # Required by the skeleton even though every term that reads it is set
        # to weight 0 below; inert here.
        foot_target_height=0.025,
        # The dance is periodic; the flat reference keeps the smoothing
        # pressure without fighting the wave.
        action_rate_weight=-0.1,
        # Pose is per-joint below; halved so what it says is "the holds hold"
        # rather than "everything freezes".
        pose_weight=0.5,
        # The ceilings are the task: the command samples as identically zero.
        command_lin_ceiling=0.0,
        command_ang_ceiling=0.0,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
        # A standing dance has no sway; the skeleton's term is off for the
        # same reason `dance_sway` turned it off.
        standing_sway_weight=0.0,
    )

    # ── The drift rulers, replaced ────────────────────────────────────────
    # Same reasoning as `dance_sway`: the skeleton's tracking terms charge the
    # base's whole velocity against a zero command, and a dance is motion.
    # The pitch loop is rotation, not translation, so both linear axes and
    # the yaw axis can carry stillness rulers; lateral has no term at all --
    # nothing commands it and nothing charges it, the position `dance_sway`
    # left it in.
    cfg.rewards["track_linear_velocity"].weight = 0.0
    cfg.rewards["track_angular_velocity"].weight = 0.0
    cfg.rewards["foreaft_still"] = RewardTermCfg(
        func=foreaft_still, weight=1.0, params={"std": 0.05}
    )
    cfg.rewards["yaw_still"] = RewardTermCfg(
        func=yaw_still, weight=0.5, params={"std": 0.10}
    )

    # ── The standing pose, as posed ───────────────────────────────────────
    robot_cfg = cfg.scene.entities["robot"]
    robot_cfg.init_state.joint_pos = DANCE_HOME

    # ── Attitude: roll only ───────────────────────────────────────────────
    # `upright` is the skeleton's and scores total tilt; 25 degrees of
    # commanded pitch would be charged by it (see the module docstring).
    cfg.rewards["upright"].weight = 0.0
    cfg.rewards["roll_level"] = RewardTermCfg(
        func=roll_level, weight=1.0, params={"std": ROLL_STD}
    )

    # ── The pose term, per joint ──────────────────────────────────────────
    # Explicit keys, no wildcard-before-specific ordering to wonder about:
    # the holds tight, the shoulders loose (they are the wave), everything
    # the pitch loop pumps -- the legs' pitch joints and the rear hips --
    # loose.
    pose = cfg.rewards["pose"]
    loose_legs = 0.40
    pose.params["std_standing"] = {
        "LF_J0_joint": 0.10,
        "LF_J1_joint": 0.30,
        "LF_J2_joint": 0.10,
        "LF_J3_joint": 0.10,
        "RF_J0_joint": 0.10,
        "RF_J1_joint": 0.30,
        "RF_J2_joint": 0.10,
        "RF_J3_joint": 0.10,
        "LM_J0_joint": 0.10,
        "RM_J0_joint": 0.10,
        "LM_J1_joint": loose_legs,
        "LM_J2_joint": loose_legs,
        "RM_J1_joint": loose_legs,
        "RM_J2_joint": loose_legs,
        "LR_.*": loose_legs,
        "RR_.*": loose_legs,
    }
    # No command ever leaves zero, so only the standing width is live; the
    # other two are written to keep the dict complete for the term's asserts.
    pose.params["std_walking"] = dict(pose.params["std_standing"])
    pose.params["std_running"] = dict(pose.params["std_standing"])

    # ── The gait terms, off ───────────────────────────────────────────────
    # Feet are planted by spec; nothing should price stepping.
    for _name in (
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
    ):
        cfg.rewards[_name].weight = 0.0

    # ── What it is paid for ───────────────────────────────────────────────
    cfg.rewards["track_pitch"] = RewardTermCfg(
        func=track_pitch, weight=4.0, params={"std": PITCH_STD}
    )
    cfg.rewards["pitch_pull"] = RewardTermCfg(func=pitch_pull, weight=-8.0)
    cfg.rewards["track_j1"] = RewardTermCfg(
        func=track_j1,
        weight=2.5,
        params={"joint_names": J1_JOINTS, "centers": J1_CENTERS, "std": J1_STD},
    )
    # The arms hold their pose outside the averaged `pose` term, linear not
    # an exp kernel, for the reason `hold_arms`'s docstring gives: a kernel
    # has no gradient where a 90-degree-off arm actually sits. The shoulders
    # (J1) are deliberately absent -- they are the wave.
    cfg.rewards["hold_arms"] = RewardTermCfg(
        func=hold_arms,
        weight=-6.0,
        params={
            "joint_names": (
                "LF_J0_joint",
                "LF_J2_joint",
                "LF_J3_joint",
                "RF_J0_joint",
                "RF_J2_joint",
                "RF_J3_joint",
            ),
        },
    )
    # All four legs planted and loaded, claws excluded -- `dance_sway`'s
    # term, same sensor slice.
    cfg.rewards["support_foot_load"] = RewardTermCfg(
        func=support_foot_load,
        weight=1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "force_threshold": 3.0,
        },
    )

    # ── And what the actuators spend ──────────────────────────────────────
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

    # ── The dance clock, to both groups ───────────────────────────────────
    for _group in ("actor", "critic"):
        cfg.observations[_group].terms["pitch_phase"] = ObservationTermCfg(
            func=pitch_phase_obs,
            history_length=0,
        )

    # ── Disturbance budget of a dance ─────────────────────────────────────
    # A standing dance trains through no random shoves; zeroed, the event
    # left in place so the config shape stays the skeleton's.
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
        }

    # The command curriculum is removed, not retuned: its `_apply` would
    # write `lin_std(0) = 0` into the live tracking stds every step.
    if "command" in cfg.curriculum:
        del cfg.curriculum["command"]

    return cfg

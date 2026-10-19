"""Environment config for jumper.dance_sway.

The parametric dance, third revision, as the operator posed it in the
puppet window: standing on the four legs with both claws posed in the air,
the body sways left-right on a 2 s rhythm shaped toward (but gentler than)
a square wave, the right claw's wrist joint (RF_J3) waves in the band
0.8-1.3 rad at 4 x the sway frequency, and roll wobble within 5 degrees
rides along uncharged. The full spec is in this package's `__init__.py`.

## How a dance without a clip is scored

The other dances imitate recordings; this one has a formula instead. The
references are deterministic functions of episode time
(`mdp/rewards.py::lateral_ref` / `claw_ref`, on the tanh-shaped clock),
scored by exp-kernel tracking terms, and -- the load-bearing half -- handed
to the policy as a phase observation. `jumper.tripod` paid for this lesson:
a reward scored against a clock the policy was never shown looks exactly
like noise from the other side.

## The drift rulers had to go

The skeleton's velocity-tracking terms score the base's whole velocity
against a zero command -- and a lateral sway **is** velocity. At this
task's amplitude the body's peak lateral speed is ~0.3 m/s against a 0.05
ruler: the moving half of every cycle scored zero on a 2.0-weight term,
which is the quiet reason the second release's policy declined to move.
They are replaced by `foreaft_still` and `yaw_still`: the same rulers
rebuilt on the axes the dance never commands. Keeping the robot standing
is `support_foot_load` and `upright`, not a velocity straightjacket.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..common.actuator import CONTINUOUS_TORQUE
from ..common.constants import EFFORT_LIMIT, GAIT_JOINTS, HOME, LEGS, STAND_Z
from ..common.mdp.curriculum import STD_ANG_RATIO, STD_LIN_RATIO
from ..common.mdp.rewards import actuator_headroom, actuator_power
from ..common.velocity_env import velocity_env_cfg
from .mdp.observations import sway_phase_obs
from .mdp.rewards import (
    edge_height,
    foreaft_still,
    hold_arms,
    support_foot_load,
    sway_pull,
    track_claw,
    track_sway,
    yaw_still,
)

#: The body's lateral sway amplitude, in metres. The operator asked for
#: 尽量大 with the feet planted (脚全程着地) -- planted legs shear the body
#: sideways only so far before the joints run out, which is of the order of
#: +-10 cm at this stance. The reference is set there, at the kinematic
#: edge, and `sway_pull`'s constant gradient is what keeps pushing the
#: policy to whatever edge it can actually hold: an unreachable reference
#: costs nothing under a linear pull, it only stops paying beyond reach.
LATERAL_AMPLITUDE = 0.10

#: The sites `lateral_offset` measures against. `foot_pos_b`'s contract is
#: all six sites in `LEGS` order (claws first, then the four legs); the
#: claw slots are dropped inside `lateral_offset`, which centres the four
#: legs only -- the claws are posing in the air and the waving one would
#: drag the centroid with it.
SUPPORT_LEGS = LEGS

#: The standing pose, as the operator posed it in the puppet (MuJoCo joint
#: angles, radians). Against HOME: LM/RM J0 spread to -0.6/+0.6 for balance
#: (the operator's instruction: they are the balance base and are not to
#: move much -- the pose term holds them tight), the left claw is posed at
#: J0 -1.0 / J1 -2.0, the right arm's J1 to -1.8 -- the operator's -1.5
#: lifted 往上一点点 after the replay showed the claw dragging; FK puts the
#: palm pad at ~10 cm clear there against ~6 cm at -1.5 -- and everything
#: else is the puppet's spawn pose unchanged. Writing them into the default
#: pose makes the spawn, the action centre and the pose term's target the
#: same place, so zero action holds this stance for free.
DANCE_HOME: dict[str, float] = {
    **HOME,
    "LM_J0_joint": -0.6,
    "RM_J0_joint": 0.6,
    "LF_J0_joint": -1.0,
    "LF_J1_joint": -2.0,
    "RF_J1_joint": -1.8,
}

#: The joint the operator's wave lives in: the right claw's **wrist** (J3),
#: moved there from the elbow (J2) by the puppet session. The band below is
#: the operator's, absolute rather than HOME-centred -- HOME (+1.417) lies
#: outside it.
CLAW_JOINT = "RF_J3_joint"
CLAW_CENTER = 1.05
CLAW_AMPLITUDE = 0.25

#: Tracking stds: the sway's ruler (metres), the claw's (rad). The exp
#: kernel's gradient dies once the error leaves `std` -- the first release
#: learned nothing at sigma far inside the miss -- so each ruler is set at
#: about half its wave's amplitude. `sway_pull` is the constant-gradient
#: backstop that makes the exact sigma less load-bearing.
SWAY_STD = 0.06
CLAW_STD = 0.15


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
        # No one drives this task -- the dance is the program. `jumper.swing`
        # is the precedent for operator_command=False.
        operator_command=False,
        teleop=False,
        # Required by the skeleton even though every term that reads it is set
        # to weight 0 below; inert here.
        foot_target_height=0.025,
        # The dance is periodic; the flat reference keeps the smoothing
        # pressure without fighting the wave.
        action_rate_weight=-0.1,
        # Pose is per-joint below; the weight is halved so what it says is
        # "the claws and the balance hips hold" rather than "everything
        # freezes".
        pose_weight=0.5,
        # The ceilings are the task: the command samples as identically zero,
        # the derived stds become 0, and the overrides below replace them.
        command_lin_ceiling=0.0,
        command_ang_ceiling=0.0,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
        # The sway IS body motion at a zero command; this term would charge
        # the whole dance. Off.
        standing_sway_weight=0.0,
    )

    # ── The drift rulers, replaced ────────────────────────────────────────
    # The skeleton's tracking terms charge the base's whole velocity against
    # a zero command -- the sway itself. Zeroed, and two axes-only rulers
    # put in their place (see the module docstring).
    cfg.rewards["track_linear_velocity"].weight = 0.0
    cfg.rewards["track_angular_velocity"].weight = 0.0
    cfg.rewards["foreaft_still"] = RewardTermCfg(
        func=foreaft_still, weight=1.0, params={"std": 0.05}
    )
    cfg.rewards["yaw_still"] = RewardTermCfg(
        func=yaw_still, weight=0.5, params={"std": 0.10}
    )

    # ── The standing pose, as posed ───────────────────────────────────────
    # One assignment, three consumers: the spawn, the pose term's target
    # (`variable_posture` reads `default_joint_pos`), and the action centre.
    robot_cfg = cfg.scene.entities["robot"]
    robot_cfg.init_state.joint_pos = DANCE_HOME

    # Attitude: upright scores sin(total tilt), so a std of 0.10 makes tilt
    # within ~5.7 degrees nearly free and charges beyond -- which is the
    # operator's "roll may wobble within 5 degrees" made a ruler.
    cfg.rewards["upright"].params["std"] = 0.10

    # ── The pose term, per joint ──────────────────────────────────────────
    # Claws tight: both are posed in the air and hold still (the left one's
    # pose is the spec; the right one's every joint but the waving wrist).
    # The waving wrist free: it is the wave. LM/RM J0 tight: the balance
    # base, the operator's 不须大幅改动. Everything the sway pumps -- the
    # legs' pitch joints and the rear hips -- loose.
    pose = cfg.rewards["pose"]
    loose_legs = 0.40
    pose.params["std_standing"] = {
        "LF_.*": 0.10,
        "RF_J0_joint": 0.10,
        "RF_J1_joint": 0.10,
        "RF_J2_joint": 0.10,
        CLAW_JOINT: 1.20,
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
    # Feet are planted by spec (脚全程着地); nothing should price stepping.
    for _name in (
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
    ):
        cfg.rewards[_name].weight = 0.0

    # ── What it is paid for ───────────────────────────────────────────────
    foot_cfg = SceneEntityCfg(
        "robot", site_names=SUPPORT_LEGS, preserve_order=True
    )
    cfg.rewards["track_sway"] = RewardTermCfg(
        func=track_sway,
        weight=4.0,
        params={
            "asset_cfg": foot_cfg,
            "amplitude": LATERAL_AMPLITUDE,
            "std": SWAY_STD,
        },
    )
    cfg.rewards["sway_pull"] = RewardTermCfg(
        func=sway_pull,
        weight=-8.0,
        params={"asset_cfg": foot_cfg, "amplitude": LATERAL_AMPLITUDE},
    )
    cfg.rewards["track_claw"] = RewardTermCfg(
        func=track_claw,
        weight=2.5,
        params={
            "center": CLAW_CENTER,
            "amplitude": CLAW_AMPLITUDE,
            "std": CLAW_STD,
            "joint_name": CLAW_JOINT,
        },
    )
    # The arms hold their pose, outside the averaged `pose` term. **Linear,
    # not an exp kernel**: the kernel version sat at zero gradient once the
    # arms were 90 degrees off (see `hold_arms`'s docstring) and the replay
    # showed exactly that failure. -6.0 against radians: a 0.1 rad sag on
    # all seven joints costs 0.6 per step, more than any term on the other
    # side of the ledger pays for moving an arm. RF_J3 is deliberately
    # absent -- it is the wave.
    cfg.rewards["hold_arms"] = RewardTermCfg(
        func=hold_arms,
        weight=-6.0,
        params={
            "joint_names": (
                "LF_J0_joint",
                "LF_J1_joint",
                "LF_J2_joint",
                "LF_J3_joint",
                "RF_J0_joint",
                "RF_J1_joint",
                "RF_J2_joint",
            ),
        },
    )
    # Level at the dwells, free through the crossing: the gate is the wave's
    # own magnitude, so the kernel prices only the two ends of each swing.
    cfg.rewards["edge_height"] = RewardTermCfg(
        func=edge_height,
        weight=1.5,
        params={"target_z": STAND_Z, "std": 0.015},
    )
    # All four legs planted and loaded -- the claws are excluded because they
    # are posed in the air and a six-foot version would bid for their return.
    cfg.rewards["support_foot_load"] = RewardTermCfg(
        func=support_foot_load,
        weight=1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "force_threshold": 3.0,
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

    # ── The dance clock, to both groups ───────────────────────────────────
    # `history_length=0`: the clock is a deterministic function of episode
    # time, so frames of it carry nothing the current frame does not.
    for _group in ("actor", "critic"):
        cfg.observations[_group].terms["sway_phase"] = ObservationTermCfg(
            func=sway_phase_obs,
            history_length=0,
        )

    # ── Disturbance budget of a dance ─────────────────────────────────────
    # The skeleton pushes every robot at +/-0.2 m/s because the four
    # locomotion tasks share the event; a standing dance has no use for it
    # and every reason not to train through random shoves. Zeroed, the event
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

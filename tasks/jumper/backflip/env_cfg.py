"""jumper.backflip -- a backflip whose policy starts from the free jump.

The observation and action layouts are `jumper.ref_free_jump`'s, deliberately
unchanged: the actor sees the same 406 numbers, the critic the same 435, and the
20 joint targets are absolute around HOME. That makes the published
`ref_free_jump/out/example/model_5999.pt` a valid initialization:

    python scripts/train.py --task jumper.backflip --model jumper_red \
        --checkpoint tasks/jumper/ref_free_jump/out/example/model_5999.pt

## What it is trained to do

A backflip in one jump, then a landing and the way back to HOME. The objective
has four parts.

**Maximum height.** `apex_height` pays the episode's own apex every step after
the landing, and `flight_height` pays the same quantity densely while the robot
is in flight. The apex is the least of the six feet and the base, each against
HOME, so feet flicked up over a base that stayed put are not height. Both terms
are linear and uncapped: higher is always worth more. They are
`ref_free_jump`'s two terms; the weights are lower here (1.0 and 0.5 against
4.0 and 2.0) so they do not drown the pitch term.

**Backward pitch.** `pitch_rate` pays the body's backward pitch angular velocity
while the episode's own jump is in flight. The reward manager scales every term
by `dt`, so the episode's total is the angle turned times the weight; the rate
is clipped to the part of the turn still unpaid, so the total stops at one full
turn, `PITCH_TURN_RAD = 2*pi`, and the term is zero after that. This is the
requested "rotate once, 360 degrees, angular-velocity reward capped": a fast flip
earns the same total as a slow one that completes the turn, but earns it sooner.
Only backward rotation (negative body-frame y) pays; a forward rock neither
earns nor gives progress back.

**Feet-first landing.** `landing_feet` pays once, `n_feet / 6`, for the most
feet that were on the ground at the same time during the `LANDING_WINDOW_S`
after the episode's first touchdown, and only when no non-foot part was down at
that touchdown. Six feet pay the full weight, one foot a sixth, a back-first
landing nothing. The window is necessary: at 200 Hz the feet of a real landing
arrive a few steps apart, and the very first contact is almost always one foot.
A non-foot contact while the base is still rising is the push's drag, not a
landing, so it does not disqualify.

**The fall penalty is gone or small.** The 50-degree `fell_over` termination is
removed: a backflip goes through it by design. `illegal_contact` is cut from -5
to -0.5, so a body-first landing is not worth much but is not worth less than
the whole episode either. `upright` is removed -- a continuous reward for staying
upright is a reward for not flipping. The self-collision, joint-limit,
action-rate and horizontal-drift terms are the jump task's; the drift term keeps
the flip in place.

An episode starts at HOME with the go at the first step, as `ref_free_jump`'s
does, and ends when the robot is back at HOME after a landing, or at `EPISODE_S`
either way. There is no `fell_over` and no reference divergence.

## What the recording does here

The reference is loaded by the command but is not a motion prior and is not fed
forward. `RSI_FRACTION` is 0 (every episode starts at HOME), there are no
`prior_joint_pos`/`prior_joint_vel` rewards, and the action term's
`feedforward_enabled` is false -- a flag rather than a step schedule, so a
replayed checkpoint cannot re-introduce the recording's straight jump. The
recording remains in the command's go/flight bookkeeping (`time_since_go`,
`jump_phase`) and in the critic's four existing input terms (`jump_phase`,
`jump_state`, `prior_target`, `feedforward`); none of them reaches the actor.

## Why the checkpoint loads

The actor is 406 wide (3 base angular velocity + 3 projected gravity + four
history terms x 20 driven joints x 5 frames) and the critic 435; the action is
20. Every observation term and its order is `ref_free_jump`'s. The checkpoint's
iteration counter, environment step counter, actor, critic and optimizer are
restored by `--checkpoint`, which is the repository's full-state resume. The
feedforward is off because of the flag, not because the restored step counter is
past its schedule.

## Physics

200 Hz and kp=20, as the recording and the jump task: the push is about 50 ms.
The red model (`assets/jumper/jumper_red.xml`) is physics-identical to
`jumper.xml` -- same bodies, joints, ranges, masses and inertias in the same
order -- so the checkpoint's joint layout is unchanged.

The reward weights are first choices, not measurements: they were set so the
pitch term can compete with the height term rather than be drowned by it, and
are to be tuned from the training logs.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from ..common.constants import (
    ACTUATOR_FORCE_NOISE,
    EFFORT_LIMIT,
    FOOT_GEOMS,
    GAIT_JOINTS,
    IMU_ANG_VEL_NOISE,
    IMU_GRAVITY_NOISE,
    JUMPER_FOOT_FRICTION_RANGE,
    PUSH_VELOCITY_RANGE,
    get_jumper_robot_cfg,
)
from ..common.mdp.observations import actuator_force
from ..common.tof import tof_sensor
from ..jump import mdp as jump_mdp
from ..posture.mdp.history import StridedHistory
from ..ref_free_jump import mdp as ref_free_mdp
from . import mdp as backflip_mdp

# ── Physics: the jump task's ───────────────────────────────────────────────────
#: 200 Hz control. The recording's push lasts about 50 ms: ten control steps
#: here, two or three at 50 Hz.
_SIM_TIMESTEP = 0.0025
_DECIMATION = 2

#: The gain the recording was made at. See the module docstring: fed forward, its
#: commands are *this* jump only at the gain that produced them. The damping
#: stays the hardware's; the torque-speed curve is the servo's.
_KP = 20.0

# ── The action ─────────────────────────────────────────────────────────────────
#: Absolute targets around HOME. The recording goes up to 1.8 rad from HOME -- the
#: middle and rear knees at the push -- which the policy must command itself by
#: the end: at the walking tasks' 0.25 an output of 7.2, at 0.5 one of 3.6. `rl_cfg.py` halves the initial std to match,
#: so exploration stays 0.15 rad, as in jumper.jump.
ACTION_SCALE = 0.5

# ── The observation's history ──────────────────────────────────────────────────
#: Frames of history, and the control steps between them: five frames 20 ms
#: apart, reaching 80 ms back. That is the window `OBS_HISTORY = 5` gives the
#: walking tasks at 50 Hz, kept at 200 Hz by striding rather than by widening, as
#: jumper.posture does -- `posture/mdp/history.py` has the argument. Five
#: consecutive frames here would reach back 20 ms.
OBS_HISTORY = 5
OBS_HISTORY_STRIDE = 4

#: The terms that carry it, in every group: the walking tasks' per-joint set
#: (`HISTORY_TERMS` in `common/velocity_env.py`, where `actuator_force`'s history
#: measured as the largest contribution). The actor has no linear velocity, no
#: contact and no preview of the recording, so the takeoff and the touchdown are
#: things it has to infer, and transients are what history is for. Chosen by that
#: analogy; not measured on this task.
HISTORY_TERMS = frozenset({"joint_pos", "joint_vel", "actions", "actuator_force"})

# ── The go ─────────────────────────────────────────────────────────────────────
#: No delay: the policy jumps when the episode starts, as in `ref_free_jump`.
GO_DELAY_RANGE = (0.0, 0.0)

#: Every episode starts at HOME. There is no RSI here: the flip is a motion the
#: recording does not contain, and a spawn partway through the straight jump
#: would hand the policy a state its own flip never reaches.
RSI_FRACTION = 0.0
RSI_PHASE_RANGE = (0.0, 0.0)

# ── The objective ──────────────────────────────────────────────────────────────
#: Metres the robot's lowest point rises per unit of height reward: 0.10 m above
#: HOME pays 1. Both height terms are linear and uncapped.
HEIGHT_SCALE = 0.10

#: The latched apex, paid every step after the landing, and the height in
#: flight. Lower than `ref_free_jump`'s 4.0/2.0 so the pitch term is not drowned:
#: the flip has to be worth more than a straight jump that lands on six feet.
#: First choices, to be tuned from the training curve.
APEX_WEIGHT = 1.0
FLIGHT_WEIGHT = 0.5

#: The backward pitch reward, a rate in rad/s. The reward manager scales by dt,
#: so the episode's total is `weight * angle`; `PITCH_TURN_RAD` is the user's
#: 360-degree cap, enforced by clipping the rate to the unpaid remainder. At
#: weight 10 the most the term can pay in an episode is 10 x 2*pi ~ 63,
#: comparable to the height terms over a short post-landing stand.
PITCH_TURN_RAD = 2.0 * math.pi
PITCH_RATE_WEIGHT = 10.0

#: The feet-first landing: `n_feet / 6` once, over the most feet seen during the
#: window. Ten points for six feet, 1.67 for one, zero for a back-first landing.
LANDING_FEET_WEIGHT = 10.0
#: How long after touchdown the feet are counted. 40 control steps: a real
#: landing puts its feet down over a few 5 ms steps, not one.
LANDING_WINDOW_S = 0.2

#: Standing at HOME once the jump is done. The kernel width is the jump task's;
#: 0.01 was tried there and found to have no gradient far from HOME.
HOME_WEIGHT = 1.0
HOME_SIGMA_SQ = 0.05

#: Horizontal distance from the spawn, charged as (d / 0.10 m)^2. A backflip is
#: in place; this is the jump task's term, not a fall penalty.
DRIFT_WEIGHT = -1.0
DRIFT_SCALE = 0.10

#: Every step anything but a foot is on the ground, **reduced from the jump
#: task's -5 to -0.5**: the user asked for the fall penalty to be reduced or
#: removed, and a back-landing is already paid nothing by `landing_feet`. It is
#: not zero so that a belly landing is still worse than a clean one, but it is
#: small enough not to outweigh the height and rotation the flip earns.
ILLEGAL_CONTACT_WEIGHT = -0.5

#: Every step two legs push on each other. The jump task's, unchanged: this is
#: a hardware protection, not a fall penalty.
LEG_COLLISION_WEIGHT = -5.0

#: `action_rate_l2`, per squared radian of joint target per step. The jump
#: task's realised -0.0144 at scale 0.25, i.e. -0.2304 per squared radian at
#: this scale 0.5 (times scale^2 in the term below). A flip moves the targets
#: faster than the straight jump; this is the term to reduce if the flip cannot
#: move.
ACTION_RATE_PER_RAD_SQ = -0.2304

#: Back at HOME, which ends the episode after the landing: the 20 driven joints
#: within `HOME_TOLERANCE` rad RMS of HOME, moving at under `HOME_SPEED_SQ`
#: (rad/s)^2, for `HOME_HOLD_STEPS` steps in a row. The jump task's values.
HOME_TOLERANCE = 0.05
HOME_SPEED_SQ = 0.5
HOME_HOLD_STEPS = 20

#: The longest an episode runs. Longer than the jump task's 2.5 s: the flip
#: itself is about the jump's duration, but the policy has to come down and the
#: episode no longer ends on the recording's span.
EPISODE_S = 3.0

#: Every collision geom, by name: exactly the `*_meshcol` ones on this model. The
#: illegal-contact sensor takes these minus the feet -- by exclusion, because
#: listing the links covered the same 35 parts today and would have missed one
#: added tomorrow without a word.
_COLLISION_GEOMS = r".*_meshcol"


def _stride_history(cfg: ManagerBasedRlEnvCfg) -> None:
    """Give every group's `HISTORY_TERMS` `OBS_HISTORY` frames, `OBS_HISTORY_STRIDE`
    control steps apart, with jumper.posture's `StridedHistory`.

    jumper.posture's handover, and the three parts of it that each fail silently:

    - the manager's `history_length` stays **0**, or the frames are stacked twice;
    - the term's noise moves **into** the wrapper, which applies it once, as a
      frame is written. Left to the manager, every stored frame is re-noised each
      step and the policy can average away an error no robot can. It moves only
      where the group is corrupted, since clearing a term's noise is how the
      manager honours `enable_corruption = False`;
    - `common/mdp/symmetry.py` and `scripts/export.py` read the count and the
      stride off the wrapper (`history_frames`, `history_stride`), because the
      config's `history_length` says 0.

    The critic's terms too: the critic is mirrored as well, and a value function
    fitted on another window than the actor's is fitted on another observation.

    **Each group's term is copied before it is wrapped.** mjlab's skeleton builds
    the critic as `{**actor_terms, ...}`, so `joint_vel` and `actions` are one
    object in both groups. Unwrapped that is harmless, the manager clearing the
    noise per group; wrapped, the first group's noise travels in the shared params
    and the critic -- uncorrupted by design -- reads the actor's +/-1.5 rad/s,
    as jumper.posture's did until MR !70 gave it the same copy.
    """
    p = StridedHistory.PREFIX
    for group_name, group in cfg.observations.items():
        if group.history_length:
            raise ValueError(
                f"{group_name}: a group-level history_length overwrites every "
                f"term's, and would stack these frames a second time"
            )
        missing = HISTORY_TERMS - set(group.terms)
        if missing:
            raise ValueError(f"{group_name} has no {sorted(missing)} to give history to")
        for name in HISTORY_TERMS:
            term = group.terms[name] = copy.copy(group.terms[name])
            assert not term.history_length, f"{group_name}/{name} is already stacked"
            assert not any(k.startswith(p) for k in term.params or {}), (
                f"{group_name}/{name} has a param colliding with {p!r}"
            )
            term.params = {
                **(term.params or {}),
                p + "func": term.func,
                p + "frames": OBS_HISTORY,
                p + "stride": OBS_HISTORY_STRIDE,
                p + "noise": term.noise if group.enable_corruption else None,
            }
            term.func = StridedHistory
            term.noise = None


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Reference-free high jump.

    Args:
        asset: path to the model XML, from `--model`. None uses the default jumper.xml.
        play: replay mode -- noise and pushes off, and every episode starts at HOME
            and jumps on the go, as the deployment will.
    """
    cfg = make_velocity_env_cfg()

    # ── Sim ───────────────────────────────────────────────────────────────────
    cfg.sim.mujoco.timestep = _SIM_TIMESTEP
    cfg.decimation = _DECIMATION
    # The velocity skeleton's contact budget for this robot, as measured there.
    cfg.sim.njmax = 512
    cfg.sim.nconmax = 128
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 128

    # ── Scene ─────────────────────────────────────────────────────────────────
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    robot = get_jumper_robot_cfg(effort_limit=EFFORT_LIMIT, asset=asset)
    for actuator in robot.articulation.actuators:
        actuator.stiffness = _KP
    cfg.scene.entities = {"robot": robot}

    # Neither of the skeleton's scans is read by anything here; the two contact
    # sets are what the flight bookkeeping and the penalties read.
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ())
        if s.name not in ("terrain_scan", "foot_height_scan")
    ) + (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            # Read by nothing here. It is what marks a contact sensor as the
            # feet's for `play.py --measure` (`mjrl/viewer/monitor.py`), which
            # without it replays with no foot-force panel and says nothing.
            track_air_time=True,
        ),
        # Leg against leg: the walking tasks' sensor. The whole robot on both
        # sides, and the collision masks leave only different legs able to touch.
        # One entry per physics substep, so a knock between control steps counts.
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=_DECIMATION,
        ),
        ContactSensorCfg(
            name="body_ground_contact",
            primary=ContactMatch(mode="geom", pattern=_COLLISION_GEOMS, entity="robot",
                                 exclude=FOOT_GEOMS),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
            num_slots=1,
        ),
    )

    # ── Observations ──────────────────────────────────────────────────────────
    for group in cfg.observations.values():
        for name in ("command", "height_scan", "foot_height", "foot_air_time",
                     "foot_contact", "foot_contact_forces"):
            group.terms.pop(name, None)
    # No state estimator on the robot, so no linear velocity for the actor; the
    # critic keeps it.
    del cfg.observations["actor"].terms["base_lin_vel"]

    # The 20 driven joints, as everywhere else; the grippers are held at HOME.
    gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))
    for group in cfg.observations.values():
        for name in ("joint_pos", "joint_vel"):
            term = group.terms.get(name)
            if term is not None:
                term.params = dict(term.params or {}) | {"asset_cfg": gait_cfg}

    # This robot's IMU, from `constants.py`, not the Go1's the skeleton carries.
    for group in cfg.observations.values():
        if "base_ang_vel" in group.terms:
            group.terms["base_ang_vel"].noise = Unoise(
                n_min=-IMU_ANG_VEL_NOISE, n_max=IMU_ANG_VEL_NOISE
            )
        if "projected_gravity" in group.terms:
            group.terms["projected_gravity"].noise = Unoise(
                n_min=-IMU_GRAVITY_NOISE, n_max=IMU_GRAVITY_NOISE
            )

    # The current sensor, as the walking tasks read it: the 20 driven actuators,
    # with this robot's noise. On the ground the legs carry the body and in the
    # air only themselves -- the difference the takeoff and the touchdown make.
    force_cfg = SceneEntityCfg("robot", actuator_names=list(GAIT_JOINTS))
    for group in cfg.observations.values():
        group.terms["actuator_force"] = ObservationTermCfg(
            func=actuator_force,
            params={"asset_cfg": force_cfg},
            noise=Unoise(n_min=-ACTUATOR_FORCE_NOISE, n_max=ACTUATOR_FORCE_NOISE),
        )

    # The go clock, for the critic alone: the recording's progress since the go,
    # which the prior, the feedforward and the height bookkeeping are read
    # against. The actor jumps when it starts to run and needs no clock -- and a
    # clock is what the robot could not supply without the recording.
    cfg.observations["critic"].terms["jump_phase"] = ObservationTermCfg(
        func=jump_mdp.observations.jump_phase, params={"command_name": "jump"}
    )
    # And for the critic alone, what the rewards are computed from.
    cfg.observations["critic"].terms["jump_state"] = ObservationTermCfg(
        func=ref_free_mdp.observations.jump_state,
        params={"command_name": "jump", "scale": HEIGHT_SCALE, "mirror_kind": "scalar"},
    )
    cfg.observations["critic"].terms["prior_target"] = ObservationTermCfg(
        func=ref_free_mdp.observations.prior_target,
        params={"command_name": "jump", "mirror_kind": "joint"},
    )
    cfg.observations["critic"].terms["feedforward"] = ObservationTermCfg(
        func=ref_free_mdp.observations.feedforward, params={"mirror_kind": "scalar"},
    )

    # ── Action: absolute targets around HOME, no recording fed forward ────────
    # The class and its two critic observations are kept so the actor stays 406
    # wide and the critic 435; `feedforward_enabled=False` makes the term a plain
    # absolute position target, and the critic's `feedforward` input a constant
    # 0. `hold_step`/`end_step` are then unused but must still be ordered.
    cfg.actions["joint_pos"] = ref_free_mdp.actions.PriorFeedforwardJointPositionActionCfg(
        entity_name="robot",
        actuator_names=list(GAIT_JOINTS),
        scale={j: ACTION_SCALE for j in GAIT_JOINTS},
        use_default_offset=True,
        command_name="jump",
        hold_step=0,
        end_step=1,
        feedforward_enabled=False,
    )

    # ── Command: the go ───────────────────────────────────────────────────────
    cfg.commands = {
        "jump": backflip_mdp.commands.BackflipCommandCfg(
            entity_name="robot",
            motion_file=str(jump_mdp.reference.JUMP_REF_NPZ),
            command_delay_range=GO_DELAY_RANGE,
            rsi_fraction=RSI_FRACTION,
            rsi_phase_range=RSI_PHASE_RANGE,
            home_tolerance=HOME_TOLERANCE,
            home_speed_sq=HOME_SPEED_SQ,
            pitch_turn_rad=PITCH_TURN_RAD,
            landing_window_s=LANDING_WINDOW_S,
            # A back-first landing ends the flight only while descending: during
            # the push the calves and palms touch the ground on 57% of the
            # recording's push steps, and that is not a landing.
            body_contact_ends_flight=True,
            body_contact_sensor_name="body_ground_contact",
        )
    }

    # ── Rewards ───────────────────────────────────────────────────────────────
    # Maximum height and backward rotation are the objective; the landing is a
    # one-shot bonus. There is no reference prior, no upright term and no
    # fall-over penalty -- a backflip is upside down by design.
    cfg.rewards = {
        "apex_height": RewardTermCfg(
            func=ref_free_mdp.rewards.apex_height, weight=APEX_WEIGHT,
            params={"command_name": "jump", "scale": HEIGHT_SCALE},
        ),
        "flight_height": RewardTermCfg(
            func=ref_free_mdp.rewards.flight_height, weight=FLIGHT_WEIGHT,
            params={"command_name": "jump", "scale": HEIGHT_SCALE},
        ),
        # The requested backward pitch: the body's backward y rate integrated
        # over the step, paid only in this episode's flight, capped at one turn.
        "pitch_rate": RewardTermCfg(
            func=backflip_mdp.rewards.pitch_rate, weight=PITCH_RATE_WEIGHT,
            params={"command_name": "jump"},
        ),
        # `n_feet / 6` once, at the jump's first touchdown, zero if a non-foot
        # part is down at that step: more feet, more reward.
        "landing_feet": RewardTermCfg(
            func=backflip_mdp.rewards.landing_feet, weight=LANDING_FEET_WEIGHT,
            params={"command_name": "jump"},
        ),
        "home_pose": RewardTermCfg(
            func=ref_free_mdp.rewards.home_pose, weight=HOME_WEIGHT,
            params={"command_name": "jump", "sigma_sq": HOME_SIGMA_SQ,
                    "asset_cfg": gait_cfg},
        ),
        "horizontal_drift": RewardTermCfg(
            func=ref_free_mdp.rewards.horizontal_drift, weight=DRIFT_WEIGHT,
            params={"scale": DRIFT_SCALE},
        ),
        "illegal_contact": RewardTermCfg(
            func=ref_free_mdp.rewards.illegal_contact, weight=ILLEGAL_CONTACT_WEIGHT,
            params={"sensor_name": "body_ground_contact", "force_threshold": 1.0},
        ),
        "leg_collision": RewardTermCfg(
            func=ref_free_mdp.rewards.leg_collision, weight=LEG_COLLISION_WEIGHT,
            params={"sensor_name": "self_collision", "force_threshold": 1.0},
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        # Per radian of target, so times scale^2: see ACTION_RATE_PER_RAD_SQ.
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2, weight=ACTION_RATE_PER_RAD_SQ * ACTION_SCALE**2
        ),
    }

    # No curriculum: every weight above is the task, from the first iteration.
    cfg.curriculum = {}

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        # A time-out, not a failure: see `mdp/terminations.py`.
        "back_home": TerminationTermCfg(
            func=ref_free_mdp.terminations.back_home, time_out=True,
            params={"command_name": "jump", "hold_steps": HOME_HOLD_STEPS},
        ),
    }

    # ── Events ────────────────────────────────────────────────────────────────
    # The robot's own ranges, from `constants.py`, as `common/velocity_env.py`
    # sets them for the walking tasks.
    cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
        "robot", geom_names=FOOT_GEOMS
    )
    cfg.events["foot_friction"].params["ranges"] = JUMPER_FOOT_FRICTION_RANGE
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
    cfg.events["push_robot"].params["velocity_range"].update(PUSH_VELOCITY_RANGE)
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    cfg.events = {
        # HOME first, then RSI overwrites the share it spawns on the recording.
        "reset_to_default": EventTermCfg(
            func=envs_mdp.reset_scene_to_default, mode="reset"
        ),
        "reset_from_reference": EventTermCfg(
            func=jump_mdp.events.reset_from_reference_phase,
            mode="reset",
            params={"command_name": "jump", "rsi_fraction": RSI_FRACTION,
                    "phase_range": RSI_PHASE_RANGE},
        ),
        **cfg.events,
    }

    # ── Episode and viewer ────────────────────────────────────────────────────
    cfg.episode_length_s = EPISODE_S
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 0.9
    cfg.viewer.elevation = -15.0

    if play:
        # Every episode as the deployment will run it: from HOME, the jump at
        # once, and home again. Each ends there, so replay loops whole attempts.
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.events["reset_from_reference"].params["rsi_fraction"] = 0.0
        cfg.commands["jump"].rsi_fraction = 0.0
        cfg.curriculum = {}

    # The dToF, which the model carries (`common/tof.py`). Replay only, where the
    # live viewer shows it in the window's corner; nothing in training reads it,
    # and there it would be 2268 rays per environment per step spent on nothing.
    if play:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

    # Last, after `play`: the history takes each term's noise with it, and whether
    # there is any to take is the group's `enable_corruption`.
    _stride_history(cfg)
    return cfg


__all__ = ["env_cfg"]

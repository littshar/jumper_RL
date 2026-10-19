"""MDP terms specific to jumper.dance_sway.

The sway's clock, its reference signals, and the tracking terms. The
dance is parametric -- references are functions of episode time, not a
clip -- so the clock here is what a reference clip would have been, and
the policy sees it through `mdp/observations.py`'s phase observation (the same
hand-holding `jumper.tripod`'s gait clock does: a reward scored against a
clock the policy was never shown is not learnable).
"""

from __future__ import annotations

import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...posture.mdp.state import foot_pos_b

#: The sway's period, in seconds -- the operator's spec (周期 2s).
SWAY_PERIOD = 2.0

#: The claw's frequency, as a multiple of the sway's -- the operator's spec
#: (0.5 s claw period against the 2 s body period).
CLAW_FREQ_MULT = 4.0

#: How square the body's sway is. The reference is `tanh(k sin)` normalised,
#: which is a sine at k=0 and stiffens toward a square wave as k grows. The
#: operator asked for 近似方波但可以平缓一点 -- near-square, but gentler than
#: the k=2 the second release used, whose reversals the action-rate term
#: charged hard. k=1.2 still dwells noticeably at the sides (a plain sine
#: does not) without the snap.
SQUARE_K = 1.2

#: The claw wave's shape. The operator gave a band (0.8 to 1.3 rad) and a
#: period, not a shape; at 2 Hz the servo is kindest served a plain sine
#: (k=1.0), so only the body's wave pays for squareness.
CLAW_K = 1.0


def sway_phase(env: ManagerBasedRlEnv, mult: float = 1.0) -> torch.Tensor:
    """The dance clock at `mult` x the sway frequency, radians, [num_envs].

    Episode time, not global time: the runner randomises initial episode
    lengths to decorrelate environments, so the only clock every environment
    shares is its own -- and the reference is defined from the episode's
    start, which the phase observation below hands to the policy.
    """
    t = env.episode_length_buf.to(torch.float32) * env.step_dt
    return 2.0 * math.pi * mult * t / SWAY_PERIOD


def square_sine(phase: torch.Tensor, k: float = SQUARE_K) -> torch.Tensor:
    """`tanh(k sin)` normalised to unit peak -- the sine shaped toward square."""
    return torch.tanh(k * torch.sin(phase)) / math.tanh(k)


def lateral_ref(env: ManagerBasedRlEnv, amplitude: float) -> torch.Tensor:
    """The body's lateral reference, metres. Positive is +y in the body frame
    -- to the robot's left -- the same axis `lateral_offset` measures on, so
    the two cannot disagree in sign."""
    return amplitude * square_sine(sway_phase(env, 1.0))


def claw_ref(env: ManagerBasedRlEnv, center: float, amplitude: float) -> torch.Tensor:
    """The right claw's joint reference, radians, as an **absolute** band.

    The operator posed the wave in the puppet: RF_J3 swings between 0.8 and
    1.3 rad, which is `center=1.05 +- 0.25` -- and HOME (+1.417) lies
    *outside* that band, so centring on HOME (as the second release did,
    for the J2 elbow) is not an option here: the wave is where the operator
    put it, not wherever the rest pose happens to be.
    """
    return center + amplitude * square_sine(sway_phase(env, CLAW_FREQ_MULT), CLAW_K)


def lateral_offset(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """The base's lateral position relative to the support-feet centroid, metres.

    **Translation, not rotation** -- the operator's spec is the body moving
    side to side (左右平移). The measure: the feet in the body frame
    (`posture.mdp.state.foot_pos_b`), the centroid's y, negated -- the
    body's y-offset from its own planted base, in body axes, so spawn yaw
    cannot leak into it. `asset_cfg` carries all six sites in `LEGS` order
    (that is `foot_pos_b`'s contract) and the **last four -- the legs -- are
    the ones centred here**: the claws are posing in the air and the waving
    one would drag the centroid with it.
    """
    feet = foot_pos_b(env, asset_cfg)
    return -feet[:, 2:6, 1].mean(dim=1)


def track_sway(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    amplitude: float,
    std: float,
) -> torch.Tensor:
    """`exp(-((offset - lateral_ref) / std)^2)` -- the sway, scored each step."""
    err = lateral_ref(env, amplitude) - lateral_offset(env, asset_cfg)
    return torch.exp(-err.square() / std**2)


def sway_pull(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    amplitude: float,
) -> torch.Tensor:
    """`|lateral_ref - offset|` -- a constant-gradient pull toward the wave.

    The exp kernel's gradient dies as the error grows, and the first
    widened-sigma run showed what that leaves: a policy swaying +-1 cm
    scoring 1.08 of 2.0 with no pressure to close the last 5. This term is
    the pressure -- linear in the error, so every centimetre of shortfall
    costs the same wherever the policy starts. With the amplitude set past
    what planted legs can reach (the operator's 尽量大), it is also what
    pushes the policy to the kinematic edge: the pull never stops paying
    for another centimetre, even where the kernel has given up.
    The weight carries the minus sign; this returns a magnitude.
    """
    return (lateral_ref(env, amplitude) - lateral_offset(env, asset_cfg)).abs()


def track_claw(
    env: ManagerBasedRlEnv,
    center: float,
    amplitude: float,
    std: float,
    joint_name: str = "RF_J3_joint",
) -> torch.Tensor:
    """`exp(-((q_claw - claw_ref) / std)^2)` -- the waving claw.

    The joint is `joint_name` by name rather than index: the action layout
    has no stable ordering across robots, and a wave trained into the wrong
    joint of an identical-looking network is silent. The second release
    waved J2 (the elbow); the operator's puppet session moved that to J3
    (the wrist), whose band 0.8-1.3 is this term's `center`/`amplitude`.
    """
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    idx = names.index(joint_name)
    q = robot.data.joint_pos[:, idx]
    err = claw_ref(env, center, amplitude) - q
    return torch.exp(-err.square() / std**2)


def hold_arms(
    env: ManagerBasedRlEnv,
    joint_names: tuple[str, ...],
) -> torch.Tensor:
    """`mean(|q - q_home|)` over **only** the named arm joints, radians.

    The operator watched the replay and named the failure: the left arm's
    J0/J1 swung with the dance (超过90度), and the right arm's J1 sagged
    until the claw dragged. The first fix was an exp kernel at std 0.08,
    and it changed nothing -- **because an exp kernel has no gradient where
    the policy actually was**: at 1.5 rad off, `exp(-(1.5/0.08)^2)` is 0 to
    forty digits, so a 90-degree excursion scored exactly as badly as a
    170-degree one and nothing pulled the arm home. Same disease the sway's
    kernel had, same cure: a **linear** penalty, whose gradient is the same
    constant at every error, so the way back is priced from wherever the
    policy stands. The weight carries the minus sign.
    """
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    idx = [names.index(n) for n in joint_names]
    q = robot.data.joint_pos[:, idx]
    target = robot.data.default_joint_pos[:, idx]
    return (q - target).abs().mean(dim=1)


def edge_height(
    env: ManagerBasedRlEnv,
    target_z: float,
    std: float,
) -> torch.Tensor:
    """`exp(-((z - target_z) * |wave|)^2 / std^2)` -- level at the dwell, free mid-cross.

    The operator's reading of the replay: the body's height swings too much
    through the lateral sway (高度变化有点太大), and what matters is the
    **two ends** -- the height at the left dwell and at the right dwell
    should agree (移动到头的时候高度最好一致); the crossing between them is
    explicitly not priced (中间可以不用管). The gate is the wave's own
    magnitude `|tanh(k sin)|`: ~1 while dwelling at either side, ~0 while
    crossing, so the kernel switches itself off through the middle without
    a threshold or a phase branch.
    """
    robot = env.scene["robot"]
    z = robot.data.root_link_pos_w[:, 2]
    gate = square_sine(sway_phase(env, 1.0)).abs()
    err = (z - target_z) * gate
    return torch.exp(-err.square() / std**2)


def support_foot_load(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 3.0,
) -> torch.Tensor:
    """`standing_foot_load` over the **four legs**, claws excluded.

    The shared `feet_ground_contact` sensor carries six slots -- the four
    leg tips and both claws' palm pads -- because the locomotion tasks walk
    on all six. This dance poses both claws in the air, so the six-foot
    version caps at 4/6 and, worse, offers a fifth of its weight back for
    lowering a claw to the floor: a standing reward quietly bidding against
    the claw pose. This term reads the same sensor and keeps the leg slots
    (`FEET` order: LF palm, RF palm, then LM/RM/LR/RR -- hence 2:6), with
    the same ramp and the standing gate dropped: this task's command is
    identically zero, so the gate would be the constant 1.
    """
    sensor = env.scene.sensors[sensor_name]
    force = sensor.data.force
    assert force is not None, f"sensor {sensor_name!r} has no force field"
    leg_force = force[:, 2:6]
    loaded = (torch.linalg.norm(leg_force, dim=-1) / force_threshold).clamp(0.0, 1.0)
    return loaded.mean(dim=1)


def foreaft_still(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
    """`exp(-(vx_b / std)^2)` -- no wandering along the sway's silent axis.

    The skeleton's drift rulers had to go (see `env_cfg.py`): they score the
    base's whole velocity against a zero command, and a lateral sway **is**
    velocity -- at the amplitude this task asks for, the rulers charged the
    entire dance. But nothing else in the budget prices x, and a dance that
    strolls off while swaying is not what was asked. This is the linear
    ruler rebuilt on the one axis the wave never moves: body-frame vx.
    """
    vx = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    return torch.exp(-vx.square() / std**2)


def yaw_still(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
    """`exp(-(wz_b / std)^2)` -- the angular ruler rebuilt, yaw only.

    Same reasoning as `foreaft_still`: the skeleton's angular ruler charged
    the whole tilt wobble the operator sanctioned (roll within 5 degrees),
    while what actually needs pricing is the robot yawing off its heading
    -- the one rotation nothing else in the budget sees.
    """
    wz = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return torch.exp(-wz.square() / std**2)

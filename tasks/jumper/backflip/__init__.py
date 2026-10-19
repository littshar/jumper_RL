"""jumper.backflip -- a backflip from the reference-free jump's policy.

The observation and action layout is `jumper.ref_free_jump`'s, unchanged, so its
published `model_5999` is a valid pretrained initialization. The objective is a
free backflip: maximum height, backward pitch capped at one full turn, and a
feet-first landing worth more the more feet are down. The recording is not used
as a motion prior or a feedforward; it is loaded only for the flight/apex
bookkeeping and the critic's existing input terms.
"""

from __future__ import annotations

from ...paths import ASSETS_DIR
from ...registry import AssetSpec, register
from ..common.assets import JUMPER_ASSETS

#: The red shell the task is trained on. It is physically identical to
#: `jumper.xml` (same bodies, joints, ranges, masses and inertias, in the same
#: order; only appearance differs), so the `ref_free_jump` checkpoint loads.
RED_XML = ASSETS_DIR / "jumper" / "jumper_red.xml"

BACKFLIP_ASSETS = (
    AssetSpec(
        name="jumper_red",
        path=RED_XML,
        description="22-DoF heterogeneous hexapod, red shell (physics-identical "
                    "to jumper.xml)",
    ),
) + JUMPER_ASSETS

register(
    id="jumper.backflip",
    assets=BACKFLIP_ASSETS,
    description=(
        "jumper hexapod backflip: maximum height, backward pitch capped at one "
        "turn, feet-first landing; initialized from the ref-free jump checkpoint"
    ),
    tags=("locomotion", "jumper", "jump", "backflip"),
)

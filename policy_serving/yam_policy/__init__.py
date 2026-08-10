"""yam_policy — a lightweight, openpi-compatible policy serving layer.

Two sides, both speaking websocket + msgpack-numpy:

* server (heavy env): wrap a policy implementing :class:`BasePolicy` in
  :class:`WebsocketPolicyServer` and run it (see :mod:`yam_policy.serve`).
* client (robot/workstation env): :class:`WebsocketClientPolicy` + optional
  :class:`ActionChunkBroker` to execute action chunks open-loop.

The wire protocol matches openpi exactly, so real openpi checkpoints served with
``openpi`` work against this client, and policies written against
:class:`BasePolicy` can be served by openpi.
"""

from .action_chunk_broker import ActionChunkBroker
from .base_policy import BasePolicy
from .websocket_client import WebsocketClientPolicy
from .websocket_server import WebsocketPolicyServer

#: Visualisation (FK, wrist-camera projection, drawing, recorded sample logs). Imported
#: lazily -- it needs mujoco/mink/pillow, which a policy server has no reason to load:
#:
#:     from yam_policy.viz import WristCameraGeometry, overlay_samples
__all__ = [
    "ActionChunkBroker",
    "BasePolicy",
    "WebsocketClientPolicy",
    "WebsocketPolicyServer",
]

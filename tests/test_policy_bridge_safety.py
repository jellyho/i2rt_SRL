"""Policy bridge rollout-state and intervention safety tests."""

from __future__ import annotations

from workstation.policy_bridge.bridge import PolicyBridge, RolloutState


class _ResetCounter:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


def _bridge() -> PolicyBridge:
    bridge = PolicyBridge.__new__(PolicyBridge)
    bridge.policy = _ResetCounter()
    bridge.state = RolloutState.RUNNING
    bridge._intervening = False
    return bridge


def test_intervention_start_and_end_each_discard_policy_chunks():
    bridge = _bridge()

    assert bridge._handle_intervention(True) is True
    assert bridge.state == RolloutState.INTERVENING
    assert bridge.policy.resets == 1

    assert bridge._handle_intervention(True) is True
    assert bridge.policy.resets == 1

    assert bridge._handle_intervention(False) is False
    assert bridge.state == RolloutState.ARMED
    assert bridge.policy.resets == 2

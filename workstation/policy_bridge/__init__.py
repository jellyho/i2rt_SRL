"""Policy client bits shared by the deployment tools.

:class:`~workstation.policy_bridge.deploy_runner.DeploymentPolicyRunner` runs the
openpi-style observation→action loop, translating between the robot server's portal RPC
and a websocket policy server. Both `yam-data deploy` and its `--headless` mode drive
this one implementation — there is deliberately no second copy, because this code decides
what the policy actually receives.
"""

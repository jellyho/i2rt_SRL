"""Test package marker.

Without this file `tests` is only a namespace package, and a *regular* `tests` package
shipped by an installed dependency (draccus) wins the import — which is why the sim smoke
tests (test_serving, test_replay, test_policy_serving) failed to collect with
`No module named 'tests._util'` and had not been running at all.
"""

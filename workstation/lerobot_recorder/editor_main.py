"""Entry point: launch the dataset viewer/editor GUI.

    python -m workstation.lerobot_recorder.editor_main --root ~/lerobot_data
    python -m workstation.lerobot_recorder.editor_main --repo-id user/yam_pick
    python -m workstation.lerobot_recorder.editor_main --mock   # synthetic dataset, no lerobot

Browse the datasets under ``root``, scrub every episode across all camera views, and
edit: relabel outcomes, set the task, or delete episodes (re-indexed + backed up).
Reads the dataset off disk only — no robot or cameras needed.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from i2rt.serving.rig_config import Resolver, load_rig
from workstation.lerobot_recorder.config import RecorderConfig


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="YAM ↔ LeRobot dataset viewer/editor")
    p.add_argument("--config", default=None, help="config.yaml (repo/root defaults)")
    p.add_argument("--repo-id", default="user/yam_bimanual")
    p.add_argument("--root", default="~/lerobot_data")
    p.add_argument("--mock", action="store_true", help="synthetic dataset (no lerobot)")
    args = p.parse_args(argv)

    rig = load_rig(args.config)
    rec = Resolver(args, p, rig.get("recorder", {}))
    cfg = RecorderConfig(repo_id=rec.get("repo_id"), root=rec.get("root"), mock=args.mock)

    from PyQt5 import QtWidgets

    from workstation.lerobot_recorder import theme
    from workstation.lerobot_recorder.editor_gui import EditorGUI

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(theme.QSS)  # same modern theme as the recorder/replay GUIs
    gui = EditorGUI(cfg)
    gui.resize(1500, 900)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

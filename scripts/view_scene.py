"""Interactive MuJoCo viewer for the YAM Peg-in-Hole scene."""

import mujoco
import mujoco.viewer

from yam_peg_insert import load_peg_insert_model


def main() -> None:
    model, data = load_peg_insert_model()
    print("Launching MuJoCo Viewer. Press Space to pause/resume, Esc to exit.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()

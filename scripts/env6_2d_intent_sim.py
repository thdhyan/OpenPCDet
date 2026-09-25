#!/usr/bin/env python3
"""Run env 6 (--env social_static) as the same scenario in intent-sim's 2D
benchmark: same four people, poses and task text, intent-sim's own
gpt-6-luna multi-round static policy, rendered to MP4 by its recorder.

Run with intent-sim's venv (it provides pygame/openai and the .env keys):

    SDL_VIDEODRIVER=dummy ../intent-sim/.venv/bin/python scripts/env6_2d_intent_sim.py

Frames: G1 sim is x right / y forward (robot at the origin facing +y);
intent-sim is a 10 x 7 m canvas, origin top-left, y DOWN, 90 deg = north.
The robot goes to (5, 6.5) facing north, so sim +y maps to intent north and
headings carry over unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INTENT_SIM = REPO.parent / "intent-sim"
sys.path[:0] = [str(INTENT_SIM), str(REPO)]

import experiments.policy_runner as policy_runner  # noqa: E402
from llm.llm_client import LLMClient  # noqa: E402
from playback.recorder import render_trace_to_video  # noqa: E402
from sim.entities import Human, Robot  # noqa: E402
from sim.scene import Scene  # noqa: E402

from g1_sim.social_environments import STATIC_HUMANS  # noqa: E402

ROBOT_2D = (5.0, 6.5)
TASKS = {
    "talk_dhyan": "Go talk to Dhyan.",
    "join_zach_nirshal": "Join Zach and Nirshal's conversation.",
}


def build_scene() -> Scene:
    ox, oy = ROBOT_2D
    humans = [Human(name=n, x=round(ox + x, 3), y=round(oy - y, 3), facing_deg=heading % 360)
              for n, _, (x, y), heading in STATIC_HUMANS]
    return Scene(robot=Robot(x=ox, y=oy, facing_deg=90.0), humans=humans)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--out", default=str(REPO / "logs/eval/2d"))
    a = p.parse_args()

    policy_runner.RUNS_DIR = Path(a.out)  # keep these out of intent-sim's scored runs/
    client = LLMClient(provider="openai", model=a.model)
    for key, task in TASKS.items():
        run_dir = policy_runner.run_policy(build_scene(), task, client, experiment=f"env6_{key}")
        video = render_trace_to_video(run_dir, output_path=run_dir / f"env6_2d_{key}.mp4")
        steps = json.loads((run_dir / "trace.json").read_text())["steps"]
        final = [s for s in steps if s.get("result") is not None]
        print(f"{key}: {video}")
        for s in final:
            print(f"   {s['function']} {s['args']} -> {s['result'].get('ok', s['result'])}"
                  f"  robot=({s['robot_x']:.2f}, {s['robot_y']:.2f})")


if __name__ == "__main__":
    main()

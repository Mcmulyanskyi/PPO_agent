from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from ppo_pendulum.config import load_config
from ppo_pendulum.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pendulum_ppo.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/ppo_pendulum.pt")
    parser.add_argument("--vecnormalize", default="checkpoints/vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(
        config=config,
        checkpoint_path=args.checkpoint,
        vecnormalize_path=args.vecnormalize,
        episodes=args.episodes,
        render=args.render,
    )


if __name__ == "__main__":
    main()

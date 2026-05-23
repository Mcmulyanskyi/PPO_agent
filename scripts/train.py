from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from ppo_pendulum.config import load_config
from ppo_pendulum.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pendulum_ppo.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = train(config)
    print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

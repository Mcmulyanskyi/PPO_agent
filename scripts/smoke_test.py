from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from ppo_pendulum.config import load_config


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pendulum_ppo.yaml")
    print(config)


if __name__ == "__main__":
    main()

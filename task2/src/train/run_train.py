from __future__ import annotations

from src.common.config import parse_common_args, prepare_runtime
from src.train.trainer import train


def main() -> None:
    parser = parse_common_args("Train the Task 2 VLM through caption, grounding, and natural VQA stages.")
    args = parser.parse_args()
    _root, cfg, dirs = prepare_runtime(args)
    train(cfg, dirs, fresh=bool(args.fresh))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python -m src.run_live` from task3 root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import parse_common_args, prepare_runtime
from src.common.logging_utils import append_status, write_json
from src.common.seed import set_seed
from src.data.load_supplied import load_all_supplied
from src.live.generate import run_live_experiment
from src.live.math_subset import load_live_math_questions


def main() -> int:
    parser = parse_common_args("ATML PA3 Task 3 — live Step 4 generation")
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()
    root, cfg, dirs, data_root = prepare_runtime(args)
    set_seed(int(cfg["seed"]))

    status_path = dirs["root"] / "STATUS.md"
    append_status(status_path, f"# Task 3 live run\n\n- seed: {cfg['seed']}")
    append_status(status_path, f"- data_root: {data_root} exists={data_root.exists()}")
    append_status(status_path, f"- smoke: {cfg.get('smoke_active')}")

    supplied = load_all_supplied(data_root, cfg, dirs)
    supplied_live_ids = None
    supplied_problems = None
    if supplied.get("available") and "math" in supplied.get("datasets", {}):
        math_payload = supplied["datasets"]["math"]
        splits = math_payload["splits"]
        supplied_live_ids = list(splits["live_ids"])
        df = math_payload["frame"]
        # Recover problem text if present in generations.
        problems = {}
        for sid in supplied_live_ids:
            sub = df[df["sample_id"].astype(str) == str(sid)]
            if len(sub) == 0:
                continue
            row = sub.iloc[0]
            problem = None
            # Official ATML-PA3-Data uses question_text (see data README).
            for col in ("question_text", "problem", "question", "prompt", "input_text"):
                if col in sub.columns and row.get(col) is not None and str(row.get(col)).strip():
                    problem = str(row[col])
                    break
            gold = row.get("gold_answer_used", row.get("gold_answer"))
            if problem:
                problems[str(sid)] = {"problem": problem, "gold_answer": gold}
        if len(problems) == len(supplied_live_ids):
            supplied_problems = problems
            append_status(status_path, "- live source: official ATML-PA3-Data")
        else:
            append_status(
                status_path,
                f"- official data present but problem text incomplete ({len(problems)}/{len(supplied_live_ids)}); falling back to HF MATH",
            )
            supplied_live_ids = None
            supplied_problems = None
    else:
        append_status(
            status_path,
            "- Official ATML-PA3-Data NOT found. Live IDs from HF MATH fallback; may need re-alignment later.",
        )

    live_payload = load_live_math_questions(
        cfg,
        artifacts_dirs=dirs,
        supplied_live_ids=supplied_live_ids,
        supplied_problems=supplied_problems,
    )
    questions = live_payload["questions"]
    if args.max_questions is not None:
        questions = questions[: int(args.max_questions)]
    if cfg.get("live_n_questions_override") is not None:
        questions = questions[: int(cfg["live_n_questions_override"])]

    write_json(dirs["live"] / "live_source.json", {k: v for k, v in live_payload.items() if k != "questions"})
    append_status(status_path, f"- n_live_questions: {len(questions)}")
    append_status(status_path, f"- live_source: {live_payload.get('source')}")

    out = run_live_experiment(cfg, questions, dirs)
    append_status(status_path, f"- inference_sec: {out['summary'].get('inference_sec_excluding_load')}")
    append_status(status_path, f"- single_acc: {out['summary'].get('single', {}).get('accuracy')}")
    append_status(status_path, f"- parallel_acc: {out['summary'].get('parallel', {}).get('accuracy')}")
    append_status(status_path, f"- sequential_acc: {out['summary'].get('sequential', {}).get('accuracy')}")
    append_status(status_path, "- LIVE_COMPLETE")
    print(json.dumps(out["summary"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

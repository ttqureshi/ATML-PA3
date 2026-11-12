from __future__ import annotations

from pathlib import Path

from src.data.image_io import image_path
from src.eval.scoring import pair_records_from_rows

SELECTION_RULE = (
    "Within each final-test suite, pairs are ranked by (1) both correct-image predictions "
    "exactly matching their canonical answers, (2) how many complementary predictions match "
    "the partner image's canonical answer, (3) whether the two complementary predictions differ, "
    "and (4) mean reference ΔNLL. The best pair of each subtype is kept first, then the list is "
    "filled to four pairs per suite. Ties keep the sorted pair id. This favors examples that are "
    "right on the original image and visibly sensitive to the partner image."
)


def _rank_key(record: dict) -> tuple:
    correct = record["conditions"]["correct"]
    comp = record["conditions"].get("complementary", {})
    both_correct = int(correct["exact"][0] == 1 and correct["exact"][1] == 1)
    follow = int(sum(comp.get("follow_partner", [0, 0])))
    differ = int(comp.get("pred_norm", ["", ""])[0] != comp.get("pred_norm", ["", ""])[1]) if comp else 0
    if comp:
        delta = 0.5 * (
            (comp["nll"][0] - correct["nll"][0]) + (comp["nll"][1] - correct["nll"][1])
        )
    else:
        delta = 0.0
    return (both_correct, follow, differ, delta)


def select_pairs(records: list[dict], limit: int = 4) -> list[dict]:
    ordered = sorted(records, key=lambda rec: rec["pair_id"])
    ordered.sort(key=_rank_key, reverse=True)
    picked: list[dict] = []
    seen = set()
    for record in ordered:
        subtype = str(record["subtype"])
        if subtype in seen:
            continue
        picked.append(record)
        seen.add(subtype)
        if len(picked) >= limit:
            return picked
    for record in ordered:
        if record in picked:
            continue
        picked.append(record)
        if len(picked) >= limit:
            break
    return picked


def _serialize(record: dict) -> dict:
    conditions = {}
    for name, payload in record["conditions"].items():
        conditions[name] = {
            "predictions": payload["prediction"],
            "prediction_norm": payload["pred_norm"],
            "canonical": payload["canonical"],
            "partner_canonical": payload["partner_canonical"],
            "exact": payload["exact"],
            "primary": payload["primary"],
            "nll": payload["nll"],
            "empty": payload["empty"],
            "image_id": payload["image_id"],
            "follow_partner": payload["follow_partner"],
        }
    return {
        "suite": record["suite"],
        "pair_id": record["pair_id"],
        "subtype": record["subtype"],
        "question": record["question"],
        "answers": record["conditions"].get("correct", {}).get("answers"),
        "conditions": conditions,
        "rank": _rank_key(record),
    }


def build_qualitative(rows: list[dict], images_root: Path) -> dict:
    by_suite = pair_records_from_rows(rows)
    selected = []
    for suite, records in by_suite.items():
        for record in select_pairs(records, 4):
            item = _serialize(record)
            item["image_files"] = []
            for image_id in item["conditions"].get("correct", {}).get("image_id", []):
                path = image_path(images_root, "val2014", int(image_id)) if image_id is not None else None
                item["image_files"].append(str(path) if path is not None else None)
            selected.append(item)
    return {"selection_rule": SELECTION_RULE, "examples": selected}


def save_contact_sheet(payload: dict, images_root: Path, out_path: Path) -> None:
    try:
        from PIL import Image
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    examples = payload.get("examples") or []
    if not examples:
        return
    cols = 2
    rows = len(examples)
    figure, axes = plt.subplots(rows, cols, figsize=(8, 3.1 * rows), squeeze=False)
    for row_index, example in enumerate(examples):
        files = example.get("image_files") or []
        for col in range(cols):
            axis = axes[row_index][col]
            axis.axis("off")
            path = Path(files[col]) if col < len(files) and files[col] else None
            if path is not None and path.exists():
                axis.imshow(Image.open(path).convert("RGB"))
            title = "image A" if col == 0 else "image B"
            pred = example["conditions"].get("correct", {}).get("predictions", ["", ""])[col]
            axis.set_title(f"{example['pair_id']} {title}\n{pred}", fontsize=8)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=120)
    plt.close(figure)
    del images_root

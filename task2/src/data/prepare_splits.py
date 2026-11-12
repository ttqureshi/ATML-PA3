"""Select COCO caption, grounding, and VQAv2 complementary splits.

Selection uses ``random.Random(seed)`` only. Evaluation images are val2014 and
are image-disjoint from each other. Training images are train2014.
"""

from __future__ import annotations

import argparse
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common.config import parse_common_args, prepare_runtime
from src.common.logging_utils import read_json, write_json
from src.common.seed import set_seed
from src.data.image_io import coco_split_name, download_file, download_images
from src.data.vqa_normalize import normalize_vqa_answer

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
VQA_URLS = {
    "questions_train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "questions_val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "annotations_train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "annotations_val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "pairs_train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip",
    "pairs_val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip",
}

IRREGULAR_PLURAL = {
    "person": "people",
    "sheep": "sheep",
    "skis": "skis",
    "mouse": "mice",
}


def pluralize(name: str) -> str:
    if name in IRREGULAR_PLURAL:
        return IRREGULAR_PLURAL[name]
    parts = name.split(" ")
    word = parts[-1]
    if word.endswith(("s", "x", "z", "ch", "sh")):
        word = word + "es"
    elif word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        word = word[:-1] + "ies"
    elif word.endswith("fe"):
        word = word[:-2] + "ves"
    elif word.endswith("f"):
        word = word[:-1] + "ves"
    else:
        word = word + "s"
    parts[-1] = word
    return " ".join(parts)


def _extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)


def _find_json(root: Path, tokens: list[str]) -> Path:
    matches = []
    for path in root.rglob("*.json"):
        name = path.name.lower()
        if all(token in name for token in tokens):
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"No JSON in {root} matching {tokens}")
    matches.sort(key=lambda path: (len(path.parts), path.name))
    return matches[0]


def download_annotations(ann_root: Path, retries: int = 5) -> dict[str, str]:
    coco_dir = ann_root / "coco"
    vqa_dir = ann_root / "vqa"
    coco_zip = ann_root / "annotations_trainval2014.zip"
    print("DOWNLOAD coco annotations", flush=True)
    download_file(COCO_ANN_URL, coco_zip, retries=retries, timeout=300)
    if not (coco_dir / "captions_train2014.json").exists():
        _extract_zip(coco_zip, coco_dir)
    paths = {
        "captions_train": str(_find_json(coco_dir, ["caption", "train"])),
        "captions_val": str(_find_json(coco_dir, ["caption", "val"])),
        "instances_train": str(_find_json(coco_dir, ["instance", "train"])),
        "instances_val": str(_find_json(coco_dir, ["instance", "val"])),
    }
    for key, url in VQA_URLS.items():
        dest_zip = ann_root / f"{key}.zip"
        extract_dir = vqa_dir / key
        print("DOWNLOAD", key, flush=True)
        download_file(url, dest_zip, retries=retries, timeout=300)
        if not any(extract_dir.rglob("*.json")):
            _extract_zip(dest_zip, extract_dir)
    paths["questions_train"] = str(_find_json(vqa_dir / "questions_train", ["question", "train"]))
    paths["questions_val"] = str(_find_json(vqa_dir / "questions_val", ["question", "val"]))
    paths["vqa_ann_train"] = str(_find_json(vqa_dir / "annotations_train", ["annotation", "train"]))
    paths["vqa_ann_val"] = str(_find_json(vqa_dir / "annotations_val", ["annotation", "val"]))
    paths["pairs_train"] = str(_find_json(vqa_dir / "pairs_train", ["complementary", "train"]))
    paths["pairs_val"] = str(_find_json(vqa_dir / "pairs_val", ["complementary", "val"]))
    return paths


def parse_complementary_pairs(obj: Any) -> list[tuple[int, int]]:
    if isinstance(obj, dict):
        if "pairs" in obj:
            return parse_complementary_pairs(obj["pairs"])
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for key, value in obj.items():
            if key in {"info", "license", "data_type", "data_subtype", "version"}:
                continue
            try:
                left = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                raw = value.get("complementary_question_id", value.get("complementary"))
                right = int(raw)
            else:
                right = int(value)
            pair = tuple(sorted((left, right)))
            if pair not in seen and pair[0] != pair[1]:
                seen.add(pair)
                out.append(pair)
        if out:
            return out
        raise ValueError("Could not parse complementary-pair mapping")
    if not isinstance(obj, list):
        raise ValueError(f"Unsupported complementary-pair container: {type(obj)}")
    out = []
    seen = set()
    for item in obj:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            left, right = int(item[0]), int(item[1])
        elif isinstance(item, dict):
            if "question_id" in item:
                left = int(item["question_id"])
                right = int(item.get("complementary_question_id", item.get("complementary")))
            else:
                nums = []
                for value in item.values():
                    try:
                        nums.append(int(value))
                    except (TypeError, ValueError):
                        continue
                if len(nums) < 2:
                    continue
                left, right = nums[0], nums[1]
        else:
            continue
        pair = tuple(sorted((left, right)))
        if pair[0] != pair[1] and pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def _questions_of(blob: dict) -> list[dict]:
    return list(blob.get("questions", blob))


def _annotations_of(blob: dict) -> list[dict]:
    return list(blob.get("annotations", blob))


def _consensus(raw_answers: list[str], min_agreement: int) -> tuple[str, str, int] | None:
    norms = [normalize_vqa_answer(ans) for ans in raw_answers]
    if len(norms) != 10:
        return None
    counts = Counter(norms)
    consensus, agreement = counts.most_common(1)[0]
    if agreement < min_agreement or consensus == "":
        return None
    raw_hits = [ans for ans in raw_answers if normalize_vqa_answer(ans) == consensus]
    target = sorted(Counter(raw_hits).items(), key=lambda item: (-item[1], item[0]))[0][0]
    return consensus, target, agreement


def _index_side(questions: list[dict], annotations: list[dict]) -> tuple[dict, dict]:
    qmap = {int(row["question_id"]): row for row in questions}
    amap = {int(row["question_id"]): row for row in annotations}
    return qmap, amap


def build_natural_candidates(
    questions: list[dict],
    annotations: list[dict],
    pairs: list[tuple[int, int]],
    allowed_images: set[int],
    min_agreement: int,
) -> tuple[dict[str, list[dict]], Counter]:
    qmap, amap = _index_side(questions, annotations)
    grouped: dict[str, list[dict]] = {"yes/no": [], "number": [], "other": []}
    rejected: Counter = Counter()
    for left_id, right_id in pairs:
        q1, q2 = qmap.get(left_id), qmap.get(right_id)
        a1, a2 = amap.get(left_id), amap.get(right_id)
        if not q1 or not q2 or not a1 or not a2:
            rejected["missing_record"] += 1
            continue
        image_ids = [int(q1["image_id"]), int(q2["image_id"])]
        if image_ids[0] == image_ids[1] or any(img not in allowed_images for img in image_ids):
            rejected["image"] += 1
            continue
        text1 = " ".join(str(q1["question"]).split())
        text2 = " ".join(str(q2["question"]).split())
        if text1 != text2 or not text1:
            rejected["question_mismatch"] += 1
            continue
        type1 = str(a1.get("answer_type", ""))
        type2 = str(a2.get("answer_type", ""))
        if type1 != type2 or type1 not in grouped:
            rejected["answer_type"] += 1
            continue
        raw1 = [str(item["answer"]) for item in a1["answers"]]
        raw2 = [str(item["answer"]) for item in a2["answers"]]
        side1 = _consensus(raw1, min_agreement)
        side2 = _consensus(raw2, min_agreement)
        if side1 is None or side2 is None:
            rejected["agreement"] += 1
            continue
        if side1[0] == side2[0]:
            rejected["same_consensus"] += 1
            continue
        grouped[type1].append(
            {
                "question": text1,
                "answer_type": type1,
                "source": "natural",
                "examples": [
                    {
                        "image_id": image_ids[0],
                        "question_id": left_id,
                        "target": side1[1],
                        "canonical": side1[0],
                        "answers": raw1,
                        "agreement": side1[2],
                    },
                    {
                        "image_id": image_ids[1],
                        "question_id": right_id,
                        "target": side2[1],
                        "canonical": side2[0],
                        "answers": raw2,
                        "agreement": side2[2],
                    },
                ],
            }
        )
    return grouped, rejected


def _take_natural(
    cands: list[dict],
    n: int,
    absolute_block: set[int],
    prefer_unique: set[int],
    rng,
    *,
    allow_reuse: bool,
) -> list[dict]:
    """Pick ``n`` pairs.

    ``absolute_block`` images are never used (other evaluation sets).
    ``prefer_unique`` images are avoided on the first pass and then recorded.
    A second pass may reuse ``prefer_unique`` images when ``allow_reuse`` is set.
    """
    order = list(cands)
    rng.shuffle(order)
    chosen: list[dict] = []
    used_pairs: set[tuple[int, int]] = set()

    def _scan(reuse: bool) -> None:
        for pair in order:
            if len(chosen) >= n:
                return
            images = (int(pair["examples"][0]["image_id"]), int(pair["examples"][1]["image_id"]))
            key = tuple(sorted(images))
            if key in used_pairs or images[0] == images[1]:
                continue
            if any(img in absolute_block for img in images):
                continue
            if not reuse and any(img in prefer_unique for img in images):
                continue
            chosen.append(pair)
            used_pairs.add(key)
            prefer_unique.update(images)

    _scan(False)
    if len(chosen) < n and allow_reuse:
        _scan(True)
    return chosen


def _instance_indexes(instances: dict) -> tuple[list[str], dict[int, set[str]], dict[tuple[int, str], int], set[int]]:
    names = [str(cat["name"]) for cat in instances["categories"]]
    id_to_name = {int(cat["id"]): str(cat["name"]) for cat in instances["categories"]}
    image_ids = {int(img["id"]) for img in instances["images"]}
    image_cats: dict[int, set[str]] = defaultdict(set)
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for ann in instances["annotations"]:
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        image_id = int(ann["image_id"])
        name = id_to_name[int(ann["category_id"])]
        image_cats[image_id].add(name)
        counts[(image_id, name)] += 1
    return names, image_cats, counts, image_ids


def _presence_question(template: str, category: str) -> str:
    return template.format(object=category)


def _counting_question(template: str, category: str) -> str:
    return template.format(objects=pluralize(category))


def build_presence_pairs(
    image_ids: list[int],
    image_cats: dict[int, set[str]],
    categories: list[str],
    n_pairs: int,
    reserved: set[int],
    rng,
    template: str,
    *,
    exact: bool,
) -> list[dict]:
    yes: dict[str, list[int]] = {cat: [] for cat in categories}
    no: dict[str, list[int]] = {cat: [] for cat in categories}
    for image_id in image_ids:
        if image_id in reserved:
            continue
        present = image_cats.get(image_id, set())
        for cat in categories:
            (yes if cat in present else no)[cat].append(image_id)
    for cat in categories:
        rng.shuffle(yes[cat])
        rng.shuffle(no[cat])
    order = list(categories)
    rng.shuffle(order)
    pairs: list[dict] = []
    guard = 0
    while len(pairs) < n_pairs and guard < max(1000, n_pairs * 50):
        guard += 1
        progressed = False
        for cat in order:
            yes[cat] = [img for img in yes[cat] if img not in reserved]
            no[cat] = [img for img in no[cat] if img not in reserved]
            if not yes[cat] or not no[cat]:
                continue
            positive = yes[cat].pop()
            negative = no[cat].pop()
            if positive == negative:
                continue
            reserved.add(positive)
            reserved.add(negative)
            pairs.append(
                {
                    "source": "presence",
                    "task": "presence",
                    "answer_type": "presence",
                    "category": cat,
                    "question": _presence_question(template, cat),
                    "examples": [
                        {
                            "image_id": positive,
                            "target": "yes",
                            "canonical": "yes",
                            "role": "positive",
                        },
                        {
                            "image_id": negative,
                            "target": "no",
                            "canonical": "no",
                            "role": "negative",
                        },
                    ],
                }
            )
            progressed = True
            if len(pairs) >= n_pairs:
                break
        if not progressed:
            break
    if exact and len(pairs) < n_pairs:
        raise RuntimeError(f"Need {n_pairs} presence pairs, built {len(pairs)}")
    return pairs


def build_counting_pairs(
    image_ids: list[int],
    counts: dict[tuple[int, str], int],
    categories: list[str],
    n_pairs: int,
    reserved: set[int],
    rng,
    template: str,
    *,
    exact: bool,
) -> list[dict]:
    eligible: dict[str, dict[int, list[int]]] = {cat: {k: [] for k in (1, 2, 3, 4)} for cat in categories}
    for image_id in image_ids:
        if image_id in reserved:
            continue
        for cat in categories:
            count = int(counts.get((image_id, cat), 0))
            if count in (1, 2, 3, 4):
                eligible[cat][count].append(image_id)
    for cat in categories:
        for count in (1, 2, 3, 4):
            rng.shuffle(eligible[cat][count])
    order = list(categories)
    rng.shuffle(order)
    pairs: list[dict] = []
    guard = 0
    while len(pairs) < n_pairs and guard < max(1000, n_pairs * 50):
        guard += 1
        progressed = False
        for cat in order:
            buckets = []
            for count in (1, 2, 3, 4):
                eligible[cat][count] = [img for img in eligible[cat][count] if img not in reserved]
                if eligible[cat][count]:
                    buckets.append(count)
            if len(buckets) < 2:
                continue
            first, second = rng.sample(buckets, 2)
            image_a = eligible[cat][first].pop()
            image_b = eligible[cat][second].pop()
            if image_a == image_b:
                continue
            reserved.add(image_a)
            reserved.add(image_b)
            pairs.append(
                {
                    "source": "counting",
                    "task": "counting",
                    "answer_type": "counting",
                    "category": cat,
                    "question": _counting_question(template, cat),
                    "examples": [
                        {
                            "image_id": image_a,
                            "target": str(first),
                            "canonical": str(first),
                            "count": first,
                        },
                        {
                            "image_id": image_b,
                            "target": str(second),
                            "canonical": str(second),
                            "count": second,
                        },
                    ],
                }
            )
            progressed = True
            if len(pairs) >= n_pairs:
                break
        if not progressed:
            break
    if exact and len(pairs) < n_pairs:
        raise RuntimeError(f"Need {n_pairs} counting pairs, built {len(pairs)}")
    return pairs


def _assign_pair_ids(pairs: list[dict], prefix: str) -> list[dict]:
    assigned = []
    for index, pair in enumerate(pairs):
        row = dict(pair)
        row["pair_id"] = f"{prefix}_{index:06d}"
        assigned.append(row)
    return assigned


def _select_captions(
    captions: list[dict],
    allowed: set[int],
    banned: set[int],
    n_images: int,
    per_image: int,
    rng,
    split_name: str,
    id_prefix: str,
) -> list[dict]:
    by_image: dict[int, list[dict]] = defaultdict(list)
    for row in captions:
        image_id = int(row["image_id"])
        text = str(row.get("caption", "")).strip()
        if image_id in allowed and image_id not in banned and text:
            by_image[image_id].append(row)
    candidates = sorted(img for img, rows in by_image.items() if len(rows) >= per_image)
    rng.shuffle(candidates)
    if len(candidates) < n_images:
        raise RuntimeError(f"Need {n_images} caption images on {split_name}, found {len(candidates)}")
    chosen_images = candidates[:n_images]
    examples = []
    for image_id in chosen_images:
        rows = sorted(by_image[image_id], key=lambda row: int(row.get("id", 0)))
        picked = rows if len(rows) == per_image else rng.sample(rows, per_image)
        for row in picked:
            examples.append(
                {
                    "example_id": f"{id_prefix}_{len(examples):06d}",
                    "image_id": image_id,
                    "image_split": split_name,
                    "caption_id": int(row.get("id", 0)),
                    "caption": str(row["caption"]).strip(),
                    "source": "caption",
                }
            )
    banned.update(chosen_images)
    return examples


def data_sizes(cfg: dict) -> dict[str, Any]:
    data = cfg["data"]
    return {
        "min_agreement": int(data["min_agreement"]),
        "caption_train_images": int(data["caption_train_images"]),
        "captions_per_train_image": int(data["captions_per_train_image"]),
        "caption_dev_images": int(data["caption_dev_images"]),
        "grounding_train_pool_cap": {k: int(v) for k, v in data["grounding_train_pool_cap"].items()},
        "grounding_dev_pairs": {k: int(v) for k, v in data["grounding_dev_pairs"].items()},
        "grounding_test_pairs": {k: int(v) for k, v in data["grounding_test_pairs"].items()},
        "natural_train_pairs": {k: int(v) for k, v in data["natural_train_pairs"].items()},
        "natural_dev_pairs": {k: int(v) for k, v in data["natural_dev_pairs"].items()},
        "natural_test_pairs": {k: int(v) for k, v in data["natural_test_pairs"].items()},
        "templates": dict(data["templates"]),
    }


def build_splits(loaded: dict, sizes: dict, seed: int) -> dict:
    """Pure selection. ``loaded`` holds captions, instances, VQA questions, annotations, pairs."""
    rng = __import__("random").Random(seed)
    min_agreement = int(sizes["min_agreement"])
    templates = sizes["templates"]
    train_names, train_cats, train_counts, train_images = _instance_indexes(loaded["instances"]["train2014"])
    val_names, val_cats, val_counts, val_images = _instance_indexes(loaded["instances"]["val2014"])
    categories = sorted(set(train_names) & set(val_names))

    val_natural, val_reject = build_natural_candidates(
        loaded["questions"]["val2014"],
        loaded["annotations"]["val2014"],
        loaded["pairs"]["val2014"],
        val_images,
        min_agreement,
    )
    train_natural, train_reject = build_natural_candidates(
        loaded["questions"]["train2014"],
        loaded["annotations"]["train2014"],
        loaded["pairs"]["train2014"],
        train_images,
        min_agreement,
    )

    val_used: set[int] = set()
    natural_dev: list[dict] = []
    natural_test: list[dict] = []
    for answer_type in ("yes/no", "number", "other"):
        dev_n = int(sizes["natural_dev_pairs"][answer_type])
        got = _take_natural(val_natural[answer_type], dev_n, val_used, val_used, rng, allow_reuse=False)
        if len(got) < dev_n:
            raise RuntimeError(
                f"Natural val dev {answer_type}: need {dev_n}, got {len(got)} "
                f"(candidates {len(val_natural[answer_type])}, rejected {dict(val_reject)})"
            )
        natural_dev.extend(got)
    for answer_type in ("yes/no", "number", "other"):
        test_n = int(sizes["natural_test_pairs"][answer_type])
        got = _take_natural(val_natural[answer_type], test_n, val_used, val_used, rng, allow_reuse=False)
        if len(got) < test_n:
            raise RuntimeError(
                f"Natural val test {answer_type}: need {test_n}, got {len(got)} "
                f"(candidates {len(val_natural[answer_type])})"
            )
        natural_test.extend(got)

    train_prefer: set[int] = set()
    natural_train: list[dict] = []
    for answer_type in ("yes/no", "number", "other"):
        need = int(sizes["natural_train_pairs"][answer_type])
        got = _take_natural(
            train_natural[answer_type], need, set(), train_prefer, rng, allow_reuse=True
        )
        if len(got) < need:
            raise RuntimeError(
                f"Natural train {answer_type}: need {need}, got {len(got)} "
                f"(candidates {len(train_natural[answer_type])}, rejected {dict(train_reject)})"
            )
        natural_train.extend(got)

    dev_p = int(sizes["grounding_dev_pairs"]["presence"])
    test_p = int(sizes["grounding_test_pairs"]["presence"])
    dev_c = int(sizes["grounding_dev_pairs"]["counting"])
    test_c = int(sizes["grounding_test_pairs"]["counting"])
    val_id_list = sorted(val_images)
    presence_val = build_presence_pairs(
        val_id_list, val_cats, categories, dev_p + test_p, val_used, rng, templates["presence"], exact=True
    )
    counting_val = build_counting_pairs(
        val_id_list, val_counts, categories, dev_c + test_c, val_used, rng, templates["counting"], exact=True
    )
    grounding_dev = presence_val[:dev_p] + counting_val[:dev_c]
    grounding_test = presence_val[dev_p : dev_p + test_p] + counting_val[dev_c : dev_c + test_c]

    train_id_list = sorted(train_images)
    presence_reserved: set[int] = set()
    counting_reserved: set[int] = set()
    presence_train = build_presence_pairs(
        train_id_list,
        train_cats,
        categories,
        int(sizes["grounding_train_pool_cap"]["presence"]),
        presence_reserved,
        rng,
        templates["presence"],
        exact=False,
    )
    counting_train = build_counting_pairs(
        train_id_list,
        train_counts,
        categories,
        int(sizes["grounding_train_pool_cap"]["counting"]),
        counting_reserved,
        rng,
        templates["counting"],
        exact=False,
    )
    if not presence_train or not counting_train:
        raise RuntimeError(
            f"Empty grounding train pool: presence={len(presence_train)} counting={len(counting_train)}"
        )

    captions_dev = _select_captions(
        loaded["captions"]["val2014"],
        val_images,
        val_used,
        int(sizes["caption_dev_images"]),
        1,
        rng,
        "val2014",
        "cap_dev",
    )
    captions_train = _select_captions(
        loaded["captions"]["train2014"],
        train_images,
        set(),
        int(sizes["caption_train_images"]),
        int(sizes["captions_per_train_image"]),
        rng,
        "train2014",
        "cap_train",
    )

    natural_train = _assign_pair_ids(natural_train, "natural_train")
    natural_dev = _assign_pair_ids(natural_dev, "natural_dev")
    natural_test = _assign_pair_ids(natural_test, "natural_test")
    grounding_dev = _assign_pair_ids(grounding_dev, "ground_dev")
    grounding_test = _assign_pair_ids(grounding_test, "ground_test")
    presence_train = _assign_pair_ids(presence_train, "presence_train")
    counting_train = _assign_pair_ids(counting_train, "counting_train")

    splits = {
        "seed": int(seed),
        "sizes": sizes,
        "selection_order": [
            "natural_val_dev_by_type",
            "natural_val_test_by_type",
            "natural_train_by_type",
            "grounding_val_presence_then_counting",
            "grounding_train_presence_pool",
            "grounding_train_counting_pool",
            "caption_dev",
            "caption_train",
        ],
        "rejected": {"val": dict(val_reject), "train": dict(train_reject)},
        "captions_train": captions_train,
        "captions_dev": captions_dev,
        "grounding_train_presence": presence_train,
        "grounding_train_counting": counting_train,
        "grounding_dev": grounding_dev,
        "grounding_test": grounding_test,
        "natural_train": natural_train,
        "natural_dev": natural_dev,
        "natural_test": natural_test,
    }
    verify_splits(splits, sizes)
    splits["selected_ids"] = selected_ids(splits)
    return splits


def _images_of_pairs(pairs: list[dict]) -> set[int]:
    out = set()
    for pair in pairs:
        for side in pair["examples"]:
            out.add(int(side["image_id"]))
    return out


def _images_of_captions(rows: list[dict]) -> set[int]:
    return {int(row["image_id"]) for row in rows}


def verify_splits(splits: dict, sizes: dict) -> None:
    if len(splits["captions_train"]) != int(sizes["caption_train_images"]) * int(sizes["captions_per_train_image"]):
        raise RuntimeError("Unexpected caption train count")
    if len({row["image_id"] for row in splits["captions_dev"]}) != int(sizes["caption_dev_images"]):
        raise RuntimeError("Unexpected caption dev image count")
    if len(splits["grounding_dev"]) != sum(sizes["grounding_dev_pairs"].values()):
        raise RuntimeError("Unexpected grounding dev pair count")
    if len(splits["grounding_test"]) != sum(sizes["grounding_test_pairs"].values()):
        raise RuntimeError("Unexpected grounding test pair count")
    for split_name, spec in (
        ("natural_train", sizes["natural_train_pairs"]),
        ("natural_dev", sizes["natural_dev_pairs"]),
        ("natural_test", sizes["natural_test_pairs"]),
    ):
        counts = Counter(pair["answer_type"] for pair in splits[split_name])
        for key, need in spec.items():
            if counts[key] != int(need):
                raise RuntimeError(f"{split_name} {key}: expected {need}, got {counts[key]}")
    dev_sets = {
        "caption_dev": _images_of_captions(splits["captions_dev"]),
        "ground_dev": _images_of_pairs(splits["grounding_dev"]),
        "ground_test": _images_of_pairs(splits["grounding_test"]),
        "natural_dev": _images_of_pairs(splits["natural_dev"]),
        "natural_test": _images_of_pairs(splits["natural_test"]),
    }
    train_images = _images_of_captions(splits["captions_train"])
    train_images |= _images_of_pairs(splits["grounding_train_presence"])
    train_images |= _images_of_pairs(splits["grounding_train_counting"])
    train_images |= _images_of_pairs(splits["natural_train"])
    names = list(dev_sets)
    for i, left in enumerate(names):
        if dev_sets[left] & train_images:
            raise RuntimeError(f"{left} overlaps training images")
        for right in names[i + 1 :]:
            overlap = dev_sets[left] & dev_sets[right]
            if overlap:
                raise RuntimeError(f"{left} overlaps {right} on {len(overlap)} images")
    for pair in splits["grounding_dev"] + splits["grounding_test"] + splits["grounding_train_presence"]:
        if pair["task"] == "presence":
            answers = [side["canonical"] for side in pair["examples"]]
            if sorted(answers) != ["no", "yes"]:
                raise RuntimeError(f"Presence pair {pair['pair_id']} is not yes/no")
            if pair["examples"][0]["image_id"] == pair["examples"][1]["image_id"]:
                raise RuntimeError("Presence pair reused an image")
    for pair in splits["grounding_dev"] + splits["grounding_test"] + splits["grounding_train_counting"]:
        if pair["task"] != "counting":
            continue
        counts = [int(side["count"]) for side in pair["examples"]]
        if len(set(counts)) != 2 or any(c not in {1, 2, 3, 4} for c in counts):
            raise RuntimeError(f"Counting pair {pair['pair_id']} has counts {counts}")
    for pair in splits["natural_train"] + splits["natural_dev"] + splits["natural_test"]:
        left, right = pair["examples"]
        if left["canonical"] == right["canonical"]:
            raise RuntimeError(f"Natural pair {pair['pair_id']} has identical consensus")
        if int(left["agreement"]) < int(sizes["min_agreement"]) or int(right["agreement"]) < int(sizes["min_agreement"]):
            raise RuntimeError(f"Natural pair {pair['pair_id']} failed agreement")
        if len(left["answers"]) != 10 or len(right["answers"]) != 10:
            raise RuntimeError("Natural examples must keep all 10 answers")


def selected_ids(splits: dict) -> dict:
    def qids(pairs: list[dict]) -> list[int]:
        out = []
        for pair in pairs:
            for side in pair["examples"]:
                if side.get("question_id") is not None:
                    out.append(int(side["question_id"]))
        return out

    def pids(pairs: list[dict]) -> list[str]:
        return [pair["pair_id"] for pair in pairs]

    train_images = sorted(
        _images_of_captions(splits["captions_train"])
        | _images_of_pairs(splits["grounding_train_presence"])
        | _images_of_pairs(splits["grounding_train_counting"])
        | _images_of_pairs(splits["natural_train"])
    )
    dev_images = sorted(
        _images_of_captions(splits["captions_dev"])
        | _images_of_pairs(splits["grounding_dev"])
        | _images_of_pairs(splits["natural_dev"])
    )
    test_images = sorted(_images_of_pairs(splits["grounding_test"]) | _images_of_pairs(splits["natural_test"]))
    return {
        "seed": splits["seed"],
        "images": {"train2014": train_images, "val2014_dev": dev_images, "val2014_final": test_images},
        "questions": {
            "natural_train": qids(splits["natural_train"]),
            "natural_dev": qids(splits["natural_dev"]),
            "natural_test": qids(splits["natural_test"]),
        },
        "pairs": {
            "grounding_train_presence": pids(splits["grounding_train_presence"]),
            "grounding_train_counting": pids(splits["grounding_train_counting"]),
            "grounding_dev": pids(splits["grounding_dev"]),
            "grounding_test": pids(splits["grounding_test"]),
            "natural_train": pids(splits["natural_train"]),
            "natural_dev": pids(splits["natural_dev"]),
            "natural_test": pids(splits["natural_test"]),
        },
        "captions": {
            "train_example_ids": [row["example_id"] for row in splits["captions_train"]],
            "dev_example_ids": [row["example_id"] for row in splits["captions_dev"]],
        },
    }


def iter_required_images(splits: dict) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for row in splits["captions_train"]:
        items.append(("train2014", int(row["image_id"])))
    for row in splits["captions_dev"]:
        items.append(("val2014", int(row["image_id"])))
    for key, split in (
        ("grounding_train_presence", "train2014"),
        ("grounding_train_counting", "train2014"),
        ("natural_train", "train2014"),
        ("grounding_dev", "val2014"),
        ("natural_dev", "val2014"),
        ("grounding_test", "val2014"),
        ("natural_test", "val2014"),
    ):
        for pair in splits[key]:
            for side in pair["examples"]:
                items.append((split, int(side["image_id"])))
    return items


def _load_official(paths: dict[str, str]) -> dict:
    import json

    def _read(path: str):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    captions_train = _read(paths["captions_train"])
    captions_val = _read(paths["captions_val"])
    questions_train = _read(paths["questions_train"])
    questions_val = _read(paths["questions_val"])
    ann_train = _read(paths["vqa_ann_train"])
    ann_val = _read(paths["vqa_ann_val"])
    pairs_train = _read(paths["pairs_train"])
    pairs_val = _read(paths["pairs_val"])
    return {
        "captions": {
            "train2014": captions_train["annotations"],
            "val2014": captions_val["annotations"],
        },
        "instances": {
            "train2014": _read(paths["instances_train"]),
            "val2014": _read(paths["instances_val"]),
        },
        "questions": {
            "train2014": _questions_of(questions_train),
            "val2014": _questions_of(questions_val),
        },
        "annotations": {
            "train2014": _annotations_of(ann_train),
            "val2014": _annotations_of(ann_val),
        },
        "pairs": {
            "train2014": parse_complementary_pairs(pairs_train),
            "val2014": parse_complementary_pairs(pairs_val),
        },
    }


def _signature(cfg: dict) -> dict:
    return {
        "seed": int(cfg["seed"]),
        "smoke": bool(cfg.get("smoke_active")),
        "sizes": data_sizes(cfg),
        "num_visual_tokens": int(cfg["model"]["num_visual_tokens"]),
    }


def main() -> None:
    parser = parse_common_args("Download COCO/VQAv2 annotations and select Task 2 splits.")
    args = parser.parse_args()
    _root, cfg, dirs = prepare_runtime(args)
    set_seed(int(cfg["seed"]))
    signature = _signature(cfg)
    splits_path = dirs["data"] / "splits.json"
    ids_path = dirs["data"] / "selected_ids.json"
    if splits_path.exists() and not args.fresh:
        existing = read_json(splits_path)
        if existing.get("signature") == signature:
            print("SPLITS cached", splits_path, flush=True)
            items = iter_required_images(existing)
            download_images(
                items,
                dirs["images"],
                workers=int(cfg["data"]["download_workers"]),
                retries=int(cfg["data"]["download_retries"]),
            )
            return
    paths = download_annotations(dirs["annotations"], retries=int(cfg["data"]["download_retries"]))
    write_json(dirs["data"] / "annotation_paths.json", paths)
    print("SELECT splits", flush=True)
    loaded = _load_official(paths)
    splits = build_splits(loaded, data_sizes(cfg), int(cfg["seed"]))
    splits["signature"] = signature
    write_json(splits_path, splits)
    write_json(ids_path, splits["selected_ids"])
    print(
        "SPLIT_COUNTS",
        {
            "captions_train": len(splits["captions_train"]),
            "captions_dev": len(splits["captions_dev"]),
            "presence_train": len(splits["grounding_train_presence"]),
            "counting_train": len(splits["grounding_train_counting"]),
            "ground_dev": len(splits["grounding_dev"]),
            "ground_test": len(splits["grounding_test"]),
            "natural_train": len(splits["natural_train"]),
            "natural_dev": len(splits["natural_dev"]),
            "natural_test": len(splits["natural_test"]),
        },
        flush=True,
    )
    download_images(
        iter_required_images(splits),
        dirs["images"],
        workers=int(cfg["data"]["download_workers"]),
        retries=int(cfg["data"]["download_retries"]),
    )


if __name__ == "__main__":
    main()

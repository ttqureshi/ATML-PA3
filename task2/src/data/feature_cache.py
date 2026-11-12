from __future__ import annotations

import gc
import time
from collections import OrderedDict
from pathlib import Path

import torch

from src.common.config import parse_common_args, prepare_runtime
from src.common.logging_utils import read_json, write_json
from src.common.seed import set_seed
from src.data.image_io import coco_split_name, image_path, open_rgb
from src.data.prepare_splits import iter_required_images
from src.models.siglip import extract_patch_features, load_siglip, preprocess_images


def _peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def feature_file(features_root: Path, split: str, image_id: int) -> Path:
    split_name = coco_split_name(split)
    return features_root / split_name / f"{int(image_id):012d}.pt"


def feature_key(split: str, image_id: int) -> str:
    return f"{coco_split_name(split)}/{int(image_id)}"


class FeatureStore:
    """LRU loader for cached FP16 patch features shaped ``[196, 768]``."""

    def __init__(self, features_root: Path, index: dict, capacity: int = 512):
        self.root = features_root
        self.index = index
        self.capacity = capacity
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, split: str, image_id: int) -> torch.Tensor:
        key = feature_key(split, image_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        meta = self.index.get("images", {}).get(key)
        path = self.root / meta["file"] if meta else feature_file(self.root, split, image_id)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached features for {key} at {path}")
        try:
            tensor = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            tensor = torch.load(path, map_location="cpu")
        tensor = tensor.to(dtype=torch.float16).contiguous()
        if tuple(tensor.shape) != (196, 768):
            raise RuntimeError(f"Feature {key} has shape {tuple(tensor.shape)}, expected (196, 768)")
        self._cache[key] = tensor
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return tensor


def _required_keys(splits: dict) -> list[tuple[str, int]]:
    unique = []
    seen = set()
    for split, image_id in iter_required_images(splits):
        key = (coco_split_name(split), int(image_id))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _usable(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 200_000


def cache_features(cfg: dict, dirs: dict[str, Path], device: torch.device) -> dict:
    splits = read_json(dirs["data"] / "splits.json")
    features_root = dirs["features"]
    index_path = features_root / "features_index.json"
    required = _required_keys(splits)
    pending = [(split, image_id) for split, image_id in required if not _usable(feature_file(features_root, split, image_id))]
    index = {
        "model": cfg["model"]["vision_encoder"],
        "dtype": "float16",
        "num_patches": 196,
        "hidden": int(cfg["model"]["vision_dim"]),
        "spatial_order": "14x14 row-major",
        "layernorm": "SiglipVisionModel.last_hidden_state applies post_layernorm exactly once",
        "preprocessing": "AutoImageProcessor deterministic resize/rescale/normalize; RGB; no augmentation",
        "images": {},
        "num_images": len(required),
    }
    if index_path.exists() and not pending:
        existing = read_json(index_path)
        if int(existing.get("num_images", -1)) == len(required) and existing.get("model") == cfg["model"]["vision_encoder"]:
            print("FEATURE_CACHE complete", len(required), flush=True)
            return existing
    wall = 0.0
    peak = 0.0
    if pending:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        print(f"FEATURE_CACHE extracting {len(pending)} / {len(required)}", flush=True)
        started = time.perf_counter()
        model, processor = load_siglip(cfg["model"]["vision_encoder"], device)
        batch_size = int(cfg["data"].get("feature_batch_size", 32))
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            images = [open_rgb(image_path(dirs["images"], split, image_id)) for split, image_id in chunk]
            pixels = preprocess_images(processor, images).to(device=device, dtype=next(model.parameters()).dtype)
            tokens = extract_patch_features(model, pixels)
            for (split, image_id), row in zip(chunk, tokens):
                dest = feature_file(features_root, split, image_id)
                dest.parent.mkdir(parents=True, exist_ok=True)
                cpu = row.detach().to(dtype=torch.float16).cpu().contiguous()
                torch.save(cpu, dest)
            if start % (batch_size * 10) == 0:
                print(f"FEATURE_CACHE {min(start + batch_size, len(pending))}/{len(pending)}", flush=True)
        peak = _peak_mb()
        wall = time.perf_counter() - started
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"FEATURE_CACHE freed vision encoder peak_mb={peak:.1f} sec={wall:.1f}", flush=True)
    elif index_path.exists():
        previous = read_json(index_path)
        peak = float(previous.get("peak_gpu_mem_mb") or 0.0)
        wall = float(previous.get("wall_time_sec") or 0.0)

    images_meta = {}
    for split, image_id in required:
        rel = feature_file(features_root, split, image_id).relative_to(features_root).as_posix()
        images_meta[feature_key(split, image_id)] = {"file": rel, "split": split, "image_id": int(image_id), "shape": [196, 768]}
    index["images"] = images_meta
    index["peak_gpu_mem_mb"] = peak
    index["wall_time_sec"] = wall
    index["num_extracted_this_run"] = len(pending)
    write_json(index_path, index)
    write_json(
        dirs["resources"] / "feature_cache.json",
        {"peak_gpu_mem_mb": peak, "wall_time_sec": wall, "num_images": len(required), "extracted_this_run": len(pending)},
    )
    return index


def load_feature_store(dirs: dict[str, Path]) -> FeatureStore:
    index_path = dirs["features"] / "features_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing feature index: {index_path}")
    return FeatureStore(dirs["features"], read_json(index_path))


def main() -> None:
    parser = parse_common_args("Cache frozen SigLIP patch features, then free the vision encoder.")
    args = parser.parse_args()
    _root, cfg, dirs = prepare_runtime(args)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_features(cfg, dirs, device)


if __name__ == "__main__":
    main()

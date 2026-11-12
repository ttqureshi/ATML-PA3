from __future__ import annotations

import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from src.common.logging_utils import write_json


def coco_split_name(split: str) -> str:
    key = split.strip().lower()
    if key in {"train", "train2014"}:
        return "train2014"
    if key in {"val", "val2014"}:
        return "val2014"
    raise ValueError(f"Unknown COCO split: {split}")


def coco_filename(split: str, image_id: int) -> str:
    split_name = coco_split_name(split)
    return f"COCO_{split_name}_{int(image_id):012d}.jpg"


def coco_image_url(split: str, image_id: int) -> str:
    split_name = coco_split_name(split)
    return f"http://images.cocodataset.org/{split_name}/{coco_filename(split_name, image_id)}"


def image_path(images_root: Path, split: str, image_id: int) -> Path:
    split_name = coco_split_name(split)
    return images_root / split_name / coco_filename(split_name, image_id)


def download_file(url: str, dest: Path, retries: int = 5, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "atml-pa3-task2"})
            with urllib.request.urlopen(req, timeout=timeout) as response, open(tmp, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            if tmp.stat().st_size <= 0:
                raise IOError(f"empty download: {url}")
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Failed to download {url} -> {dest}: {last_error}")


def _valid_jpeg(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with open(path, "rb") as handle:
            return handle.read(3) == b"\xff\xd8\xff"
    except OSError:
        return False


def _download_one(split: str, image_id: int, images_root: Path, retries: int) -> str:
    dest = image_path(images_root, split, image_id)
    if _valid_jpeg(dest):
        return "cached"
    url = coco_image_url(split, image_id)
    download_file(url, dest, retries=retries, timeout=60)
    if not _valid_jpeg(dest):
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Downloaded file is not a JPEG: {dest}")
    return "downloaded"


def download_images(
    items: list[tuple[str, int]],
    images_root: Path,
    *,
    workers: int = 16,
    retries: int = 5,
) -> dict[str, int]:
    """Download only the requested COCO image ids. ``items`` are (split, image_id)."""
    unique: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for split, image_id in items:
        key = (coco_split_name(split), int(image_id))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    stats = {"cached": 0, "downloaded": 0, "total": len(unique)}
    if not unique:
        return stats
    workers = max(1, min(int(workers), len(unique)))
    print(f"IMAGE_DOWNLOAD total={len(unique)} workers={workers}", flush=True)
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, split, image_id, images_root, retries): (split, image_id)
            for split, image_id in unique
        }
        done = 0
        for future in as_completed(futures):
            split, image_id = futures[future]
            done += 1
            try:
                status = future.result()
                stats[status] = stats.get(status, 0) + 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{split}:{image_id}: {exc}")
            if done % 50 == 0 or done == len(unique):
                print(f"IMAGE_DOWNLOAD progress {done}/{len(unique)}", flush=True)
    if errors:
        sample = "\n".join(errors[:8])
        raise RuntimeError(f"{len(errors)} image downloads failed.\n{sample}")
    write_json(images_root / "download_stats.json", stats)
    return stats


def open_rgb(path: Path):
    if Image is None:
        raise ImportError("Pillow is required to read images")
    with Image.open(path) as img:
        return img.convert("RGB")

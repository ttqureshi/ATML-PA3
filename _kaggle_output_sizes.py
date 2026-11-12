"""List Kaggle kernel output sizes for safe local pulls."""
from __future__ import annotations

import base64
import json
import urllib.parse
import urllib.request
from pathlib import Path

cfg = json.loads(Path.home().joinpath(".kaggle/kaggle.json").read_text())
auth = base64.b64encode(f"{cfg['username']}:{cfg['key']}".encode()).decode()
HEADERS = {"Authorization": f"Basic {auth}"}


def list_all(user: str, slug: str) -> list[tuple[str, int]]:
    files: list[tuple[str, int]] = []
    token = None
    for _ in range(100):
        q = f"userName={user}&kernelSlug={slug}"
        if token:
            q += f"&pageToken={urllib.parse.quote(token)}"
        url = f"https://www.kaggle.com/api/v1/kernels/output/list?{q}"
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001
            print(user, slug, "LIST_FAIL", type(exc).__name__, exc)
            return files
        if isinstance(raw, dict):
            batch = raw.get("files") or raw.get("data") or raw.get("items") or []
            token = raw.get("nextPageToken") or raw.get("nextItemToken") or raw.get("nextPage")
        else:
            batch = raw
            token = None
        for f in batch:
            if isinstance(f, str):
                files.append((f.replace("\\", "/"), 0))
            else:
                name = (f.get("path") or f.get("name") or f.get("fileName") or "").replace("\\", "/")
                size = int(f.get("totalBytes") or f.get("size") or f.get("fileSize") or 0)
                if name:
                    files.append((name, size))
        if not token:
            break
    return files


def summarize(user: str, slug: str) -> int:
    rows = list_all(user, slug)
    total = sum(s for _, s in rows)
    keep = []
    skip = []
    for n, s in rows:
        low = n.lower()
        if any(
            x in low
            for x in (
                "/raw/data/",
                "annotations",
                "instances_",
                "captions_",
                "person_keypoints",
                ".jpg",
                ".jpeg",
            )
        ):
            skip.append((n, s))
        else:
            keep.append((n, s))
    keep_sum = sum(s for _, s in keep)
    skip_sum = sum(s for _, s in skip)
    print("====", slug, "====")
    print("files", len(rows), "total_bytes", total, f"({total / 1e9:.2f} GB)")
    print("keep_files", len(keep), "keep_bytes", keep_sum, f"({keep_sum / 1e6:.1f} MB)")
    print("skip_rawish_bytes", skip_sum, f"({skip_sum / 1e6:.1f} MB)")
    for n, s in sorted(keep, key=lambda x: -x[1])[:30]:
        print(f"  {s / 1e6:8.2f} MB  {n}")
    return keep_sum


if __name__ == "__main__":
    for slug in ("atml-pa3-t2-pipeline", "atml-pa3-t3-live", "atml-pa3-t3-offline"):
        summarize("mtayyabtahir", slug)

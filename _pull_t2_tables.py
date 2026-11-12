"""Download Task2 kernel output files, skipping huge annotation blobs."""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER = "mtayyabtahir"
SLUG = "atml-pa3-t2-pipeline"
DEST = Path(r"g:\My Drive\MS AI\Semester_3\ATML\PAs\ATML-PA3\task2\results_pipeline_kaggle")

SKIP_SUBSTR = (
    "annotations_train",
    "annotations_trainval",
    "annotations_val.zip",
    "person_keypoints",
    "instances_train",
    "instances_val",
    "captions_train",
    "captions_val",
    "questions_train.zip",
    "questions_val.zip",
    "pairs_train.zip",
    "pairs_val.zip",
    ".jpg",
    ".jpeg",
    ".png",  # keep figures/ ; only skip raw images if any
)

KEEP_ALWAYS = (
    "STATUS.md",
    "tables/",
    "figures/",
    "logs/",
    "qualitative/",
    "configs_used/",
    "code_fingerprint",
    "stage3_end",
    "final_results",
    "failure.txt",
    "eval",
)


def auth_header() -> dict[str, str]:
    cfg = json.loads(Path.home().joinpath(".kaggle/kaggle.json").read_text())
    token = base64.b64encode(f"{cfg['username']}:{cfg['key']}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def get_json(url: str):
    req = urllib.request.Request(url, headers=auth_header())
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def list_files() -> list[dict]:
    files: list[dict] = []
    token = None
    for _ in range(50):
        q = f"userName={USER}&kernelSlug={SLUG}"
        if token:
            q += f"&pageToken={urllib.parse.quote(token)}"
        url = f"https://www.kaggle.com/api/v1/kernels/output/list?{q}"
        raw = get_json(url)
        if isinstance(raw, dict):
            batch = raw.get("files") or raw.get("data") or raw.get("items") or []
            token = raw.get("nextPageToken") or raw.get("nextItemToken") or raw.get("nextPage")
        elif isinstance(raw, list):
            batch = raw
            token = None
        else:
            raise RuntimeError(f"Unexpected list payload: {type(raw)} {repr(raw)[:200]}")
        files.extend(batch)
        print(f"listed +{len(batch)} total={len(files)} next={bool(token)}", flush=True)
        if not token:
            break
    return files


def should_download(name: str) -> bool:
    n = name.replace("\\", "/")
    if any(k in n for k in KEEP_ALWAYS):
        # still skip huge coco annotation jsons unless under tables/figures
        if "/raw/" in n and n.endswith(".json") and "coco" in n:
            return False
        if "/raw/" in n and n.endswith(".zip"):
            return False
        return True
    if "/raw/" in n:
        return False
    if any(s in n for s in SKIP_SUBSTR) and "figures/" not in n:
        return False
    # default: download non-raw
    return "/raw/" not in n


def download_file(name: str, dest: Path, retries: int = 5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"SKIP exists {name}", flush=True)
        return
    q = f"userName={USER}&kernelSlug={SLUG}&filePath={urllib.parse.quote(name)}"
    # try known endpoints
    urls = [
        f"https://www.kaggle.com/api/v1/kernels/output/download?{q}",
        f"https://www.kaggle.com/api/v1/kernels/output?{q}",
    ]
    last_err = None
    for attempt in range(retries):
        for url in urls:
            try:
                req = urllib.request.Request(url, headers=auth_header())
                with urllib.request.urlopen(req, timeout=600) as resp:
                    data = resp.read()
                dest.write_bytes(data)
                print(f"OK {name} ({len(data)} bytes)", flush=True)
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"try {attempt+1} {url.split('?')[0]} -> {type(e).__name__}: {e}", flush=True)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed {name}: {last_err}")


def main() -> None:
    files = list_files()
    # normalize names
    names = []
    for f in files:
        if isinstance(f, str):
            names.append(f)
        else:
            names.append(f.get("path") or f.get("name") or f.get("fileName") or "")
    names = [n for n in names if n]
    print("sample names:", names[:10], flush=True)
    wanted = [n for n in names if should_download(n)]
    print(f"want {len(wanted)} / {len(names)}", flush=True)
    # prioritize tables
    wanted.sort(key=lambda n: (0 if "tables/" in n or "final_results" in n else 1, n))
    for name in wanted:
        rel = name
        # strip leading output/ or similar
        for prefix in ("output/", "/kaggle/working/", "results/"):
            if rel.startswith(prefix) and prefix != "results/":
                rel = rel[len(prefix) :]
        out = DEST / rel
        # if name already starts with results/, keep; else put under results/
        if not str(rel).replace("\\", "/").startswith("results/") and "results/" not in str(DEST):
            # DEST already ends at results_pipeline_kaggle
            pass
        try:
            download_file(name, out)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {e}", flush=True)
    # report
    tables = list((DEST / "results" / "tables").glob("*")) if (DEST / "results" / "tables").exists() else []
    # also search
    found = list(DEST.rglob("*final_results*"))
    print("final_results found:", found, flush=True)
    print("tables dir:", tables, flush=True)


if __name__ == "__main__":
    main()

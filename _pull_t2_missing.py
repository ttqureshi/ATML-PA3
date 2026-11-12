"""Pull missing Task2 tables/eval/resources from Kaggle kernel output."""
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

WANT_PREFIXES = (
    "results/tables/",
    "results/raw/eval/",
    "results/resources/",
    "tables/",
    "raw/eval/",
    "resources/",
)
WANT_NAMES = (
    "final_results.md",
    "final_results.csv",
    "final_advantages.csv",
    "dev_metrics.md",
    "dev_metrics.csv",
    "resources.md",
    "resources.csv",
    "summary.json",
    "summary_recomputed.json",
    "per_example.jsonl",
    "generations.jsonl",
)


def auth_header() -> dict[str, str]:
    cfg = json.loads(Path.home().joinpath(".kaggle/kaggle.json").read_text())
    token = base64.b64encode(f"{cfg['username']}:{cfg['key']}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def get_json(url: str):
    req = urllib.request.Request(url, headers=auth_header())
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def list_files() -> list[str]:
    files: list[str] = []
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
            raise RuntimeError(f"Unexpected list payload: {type(raw)}")
        for f in batch:
            if isinstance(f, str):
                files.append(f)
            else:
                name = f.get("path") or f.get("name") or f.get("fileName") or ""
                if name:
                    files.append(name)
        print(f"listed +{len(batch)} total={len(files)} next={bool(token)}", flush=True)
        if not token:
            break
    return files


def wanted(name: str) -> bool:
    n = name.replace("\\", "/")
    if any(n.startswith(p) or f"/{p}" in f"/{n}" for p in WANT_PREFIXES):
        return True
    base = n.rsplit("/", 1)[-1]
    return base in WANT_NAMES


def download_file(name: str, dest: Path, retries: int = 6) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        # Always refresh corrupted/wrong final_results.md if tiny STATUS dump
        if dest.name == "final_results.md" and dest.stat().st_size < 800:
            pass
        else:
            print(f"SKIP exists {name} ({dest.stat().st_size} bytes)", flush=True)
            return
    q = f"userName={USER}&kernelSlug={SLUG}&filePath={urllib.parse.quote(name)}"
    urls = [
        f"https://www.kaggle.com/api/v1/kernels/output/download?{q}",
        f"https://www.kaggle.com/api/v1/kernels/output?{q}",
    ]
    last_err = None
    for attempt in range(retries):
        for url in urls:
            try:
                req = urllib.request.Request(url, headers=auth_header())
                with urllib.request.urlopen(req, timeout=900) as resp:
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
    names = list_files()
    picks = [n for n in names if wanted(n)]
    # Prefer smaller summaries before huge jsonl
    picks.sort(
        key=lambda n: (
            0 if n.endswith((".md", ".csv", ".json")) else 1,
            0 if "summary" in n else 1,
            0 if "tables/" in n.replace("\\", "/") else 1,
            n,
        )
    )
    print(f"want {len(picks)} files:", flush=True)
    for n in picks:
        print(" ", n, flush=True)
    for name in picks:
        rel = name.replace("\\", "/")
        for prefix in ("output/", "/kaggle/working/"):
            if rel.startswith(prefix):
                rel = rel[len(prefix) :]
        if not rel.startswith("results/"):
            rel = "results/" + rel.lstrip("/")
        out = DEST / rel
        try:
            download_file(name, out)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {e}", flush=True)
    tables = sorted((DEST / "results" / "tables").glob("*")) if (DEST / "results" / "tables").exists() else []
    eval_dir = DEST / "results" / "raw" / "eval" / "final"
    eval_files = sorted(eval_dir.glob("*")) if eval_dir.exists() else []
    print("tables:", [p.name for p in tables], flush=True)
    print("eval:", [p.name for p in eval_files], flush=True)


if __name__ == "__main__":
    main()

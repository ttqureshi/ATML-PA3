"""Stream Task2 kernel output zip; extract only non-raw / small result files."""
from __future__ import annotations

import base64
import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

USER = "mtayyabtahir"
SLUG = "atml-pa3-t2-pipeline"
DEST = Path(r"g:\My Drive\MS AI\Semester_3\ATML\PAs\ATML-PA3\task2\results_pipeline_kaggle")

SKIP_PREFIXES = (
    "results/raw/",
    "raw/",
)
# also skip giant trainer states if desired? keep them — useful.
MAX_MEMBER_BYTES = 80 * 1024 * 1024  # 80MB safety


def auth_header() -> dict[str, str]:
    cfg = json.loads(Path.home().joinpath(".kaggle/kaggle.json").read_text())
    token = base64.b64encode(f"{cfg['username']}:{cfg['key']}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def main() -> None:
    urls = [
        f"https://www.kaggle.com/api/v1/kernels/output/download/{USER}/{SLUG}",
        f"https://www.kaggle.com/api/v1/kernels/{USER}/{SLUG}/output/download",
    ]
    data = None
    last = None
    for url in urls:
        try:
            print("GET", url, flush=True)
            req = Request(url, headers=auth_header())
            with urlopen(req, timeout=1800) as resp:
                # stream to temp file to avoid RAM blowup
                tmp = DEST / "_output_full.zip"
                DEST.mkdir(parents=True, exist_ok=True)
                n = 0
                with tmp.open("wb") as f:
                    while True:
                        chunk = resp.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        n += len(chunk)
                        if n % (64 * 1024 * 1024) < 8 * 1024 * 1024:
                            print(f"  downloaded {n/1e6:.1f} MB", flush=True)
                print(f"zip size {n/1e6:.1f} MB -> {tmp}", flush=True)
                data = tmp
                break
        except Exception as e:  # noqa: BLE001
            last = e
            print("fail", type(e).__name__, e, flush=True)
    if data is None:
        raise SystemExit(f"download failed: {last}")

    extracted = []
    skipped = []
    with zipfile.ZipFile(data, "r") as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if name.endswith("/"):
                continue
            if any(name.startswith(p) or f"/{p}" in name for p in SKIP_PREFIXES):
                skipped.append(name)
                continue
            if info.file_size > MAX_MEMBER_BYTES and "checkpoints/" in name and "trainer_state" in name:
                skipped.append(name)
                continue
            # normalize into DEST
            out_name = name
            if not out_name.startswith("results/") and "/results/" in out_name:
                out_name = out_name[out_name.index("results/") :]
            target = DEST / out_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())
            extracted.append(out_name)
            print("extracted", out_name, info.file_size, flush=True)

    print("extracted", len(extracted), "skipped", len(skipped), flush=True)
    found = list(DEST.rglob("*final_results*"))
    print("final_results", found, flush=True)
    tables = DEST / "results" / "tables"
    if tables.exists():
        print("tables", list(tables.iterdir()), flush=True)


if __name__ == "__main__":
    main()

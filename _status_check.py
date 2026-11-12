import json
import base64
import urllib.request
from pathlib import Path

cfg = json.loads(Path.home().joinpath(".kaggle/kaggle.json").read_text())
auth = base64.b64encode(f"{cfg['username']}:{cfg['key']}".encode()).decode()


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"Authorization": "Basic " + auth})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


for slug in ["atml-pa3-t2-pipeline", "atml-pa3-t3-live"]:
    url = f"https://www.kaggle.com/api/v1/kernels/status?userName=mtayyabtahir&kernelSlug={slug}"
    try:
        raw = get(url)
        print("===", slug, "===")
        print(raw[:3000].decode("utf-8", errors="replace"))
    except Exception as e:
        print("===", slug, "ERR", e)

# try session status alternate
for slug in ["atml-pa3-t2-pipeline", "atml-pa3-t3-live"]:
    url = f"https://www.kaggle.com/api/v1/kernels/push/status?userName=mtayyabtahir&kernelSlug={slug}"
    try:
        raw = get(url)
        print("=== push-status", slug, "===")
        print(raw[:2000].decode("utf-8", errors="replace"))
    except Exception as e:
        print("=== push-status", slug, "ERR", type(e).__name__, e)

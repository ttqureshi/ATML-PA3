from __future__ import annotations

import base64
import io
import json
import os
import uuid
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KAGGLE_USER = "mtayyabtahir"


def _cid() -> str:
    return uuid.uuid4().hex[:12]


def build_code_zip_b64() -> str:
    buf = io.BytesIO()
    include = ["src", "configs", "requirements.txt", "README.md", "STATUS.md"]
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in include:
            path = ROOT / name
            if path.is_dir():
                for file_path in path.rglob("*"):
                    if file_path.is_file() and "__pycache__" not in file_path.parts and file_path.suffix != ".pyc":
                        archive.write(file_path, file_path.relative_to(ROOT).as_posix())
            elif path.exists():
                archive.write(path, path.name)
        archive.writestr("results/.gitkeep", "")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_notebook(stage: str, title: str, code_b64: str) -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "id": _cid(),
            "metadata": {},
            "source": [
                f"# {title}\n",
                f"\nStage: `{stage}`\n",
                "\nSelf-contained ATML PA3 Task 3 runner. Source is embedded as a zip. "
                "Artifacts go to `/kaggle/working/results`. Internet is required for "
                "Qwen2.5-3B-Instruct and (if needed) MATH problems from HuggingFace.\n",
            ],
        },
        {
            "cell_type": "code",
            "id": _cid(),
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import os, sys, shutil, pathlib, zipfile, io, base64\n",
                f"STAGE = '{stage}'\n",
                "dst = pathlib.Path('/kaggle/working/task3')\n",
                "if dst.exists():\n",
                "    shutil.rmtree(dst)\n",
                "dst.mkdir(parents=True)\n",
                "print('Extracting embedded code zip')\n",
                f"CODE_B64 = '''{code_b64}'''\n",
                "raw = base64.b64decode(CODE_B64)\n",
                "with zipfile.ZipFile(io.BytesIO(raw)) as zf:\n",
                "    zf.extractall(dst)\n",
                "assert (dst / 'src').is_dir() and (dst / 'configs').is_dir(), list(dst.iterdir())\n",
                "os.chdir(dst)\n",
                "sys.path.insert(0, str(dst))\n",
                "print('CODE_ROOT', dst)\n",
                "print('STAGE', STAGE)\n",
            ],
        },
        {
            "cell_type": "code",
            "id": _cid(),
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import torch, subprocess, sys\n",
                "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print(torch.cuda.get_device_name(0), 'VRAM_GiB', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))\n",
                "pkgs = [\n",
                "    'transformers>=4.45.0', 'accelerate', 'sentencepiece', 'protobuf',\n",
                "    'pyyaml', 'safetensors', 'tqdm', 'pandas', 'pyarrow', 'matplotlib',\n",
                "    'huggingface_hub', 'datasets', 'math-verify', 'latex2sympy2_extended', 'scipy',\n",
                "]\n",
                "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade'] + pkgs)\n",
            ],
        },
        {
            "cell_type": "code",
            "id": _cid(),
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import os, sys, subprocess, pathlib, json, time, traceback, shutil\n",
                "os.environ['ATML_ARTIFACTS_ROOT'] = '/kaggle/working/results'\n",
                "os.environ['HF_HOME'] = '/tmp/hf_home'\n",
                "os.environ['HF_HUB_CACHE'] = '/tmp/hf_home'\n",
                "os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_home'\n",
                "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n",
                "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')\n",
                "import zipfile\n",
                "def _parquet_ok(root):\n",
                "    root = pathlib.Path(root)\n",
                "    needed = [\n",
                "        root / 'qwen25-math' / 'generations.parquet',\n",
                "        root / 'qwen25-math' / 'probability_metrics.parquet',\n",
                "        root / 'qwen25vl-mathvista' / 'generations.parquet',\n",
                "        root / 'qwen25vl-mathvista' / 'probability_metrics.parquet',\n",
                "        root / 'qwen25vl-mathvista' / 'attention_metrics.parquet',\n",
                "    ]\n",
                "    ok = True\n",
                "    for p in needed:\n",
                "        sz = p.stat().st_size if p.exists() else -1\n",
                "        print('check', p, 'bytes', sz, flush=True)\n",
                "        if sz < 1_000_000:\n",
                "            ok = False\n",
                "    return ok\n",
                "def _extract_zips(zips, out):\n",
                "    if out.exists():\n",
                "        shutil.rmtree(out)\n",
                "    out.mkdir(parents=True)\n",
                "    for zpath in zips:\n",
                "        print('Extracting', zpath, 'size', zpath.stat().st_size, flush=True)\n",
                "        with zipfile.ZipFile(zpath) as zf:\n",
                "            zf.extractall(out)\n",
                "    # Nested ATML-PA3-Data/ or per-folder zips from kaggle -r zip.\n",
                "    nested = out / 'ATML-PA3-Data'\n",
                "    if nested.exists() and (nested / 'qwen25-math').exists():\n",
                "        return nested\n",
                "    for sub in list(out.glob('*.zip')):\n",
                "        target = out / sub.stem\n",
                "        target.mkdir(parents=True, exist_ok=True)\n",
                "        with zipfile.ZipFile(sub) as zf:\n",
                "            zf.extractall(target)\n",
                "    if (out / 'qwen25-math').exists():\n",
                "        return out\n",
                "    return out\n",
                "data_candidates = [\n",
                "    pathlib.Path('/kaggle/input/atml-pa3-data'),\n",
                "    pathlib.Path('/kaggle/input/datasets/mtayyabtahir/atml-pa3-data'),\n",
                "]\n",
                "resolved = None\n",
                "for cand in data_candidates:\n",
                "    if not cand.exists():\n",
                "        continue\n",
                "    print('candidate', cand, 'children', sorted(p.name for p in cand.iterdir())[:30], flush=True)\n",
                "    zips = sorted({*cand.glob('*.zip'), *cand.rglob('ATML-PA3-Data.zip'), *cand.rglob('qwen25*.zip')})\n",
                "    # Prefer extracting archives; Kaggle sometimes mounts broken 0-byte parquet stubs beside the zip.\n",
                "    if zips:\n",
                "        resolved = _extract_zips(zips, pathlib.Path('/kaggle/working/ATML-PA3-Data'))\n",
                "    elif _parquet_ok(cand):\n",
                "        resolved = cand\n",
                "    elif (cand / 'ATML-PA3-Data').exists() and _parquet_ok(cand / 'ATML-PA3-Data'):\n",
                "        resolved = cand / 'ATML-PA3-Data'\n",
                "    if resolved is not None:\n",
                "        if not _parquet_ok(resolved):\n",
                "            print('INVALID data root (parquet too small/missing):', resolved, flush=True)\n",
                "            resolved = None\n",
                "            continue\n",
                "        break\n",
                "if resolved is not None:\n",
                "    os.environ['ATML_PA3_DATA'] = str(resolved)\n",
                "    print('ATML_PA3_DATA', resolved, flush=True)\n",
                "else:\n",
                "    raise SystemExit('ATML-PA3-Data not found or parquet files invalid under /kaggle/input')\n",
                "if STAGE == 'smoke':\n",
                "    os.environ['ATML_SMOKE'] = '1'\n",
                "pathlib.Path('/kaggle/working/results').mkdir(parents=True, exist_ok=True)\n",
                "t0 = time.time()\n",
                "module = 'src.run_offline' if STAGE == 'offline' else 'src.run_live'\n",
                "cmd = [sys.executable, '-m', module, '--artifacts-root', '/kaggle/working/results']\n",
                "if STAGE == 'smoke':\n",
                "    cmd.append('--smoke')\n",
                "print('Running', cmd, flush=True)\n",
                "try:\n",
                "    rc = subprocess.call(cmd)\n",
                "except Exception:\n",
                "    pathlib.Path('/kaggle/working/results/failure.txt').write_text(traceback.format_exc())\n",
                "    raise\n",
                "print('RC', rc, 'elapsed_min', round((time.time() - t0) / 60, 2))\n",
                "status = {'stage': STAGE, 'returncode': rc, 'elapsed_sec': time.time() - t0}\n",
                "pathlib.Path('/kaggle/working/results/stage_status.json').write_text(json.dumps(status, indent=2))\n",
                "shutil.rmtree('/kaggle/working/task3', ignore_errors=True)\n",
                "shutil.rmtree('/tmp/hf_home', ignore_errors=True)\n",
                "if rc != 0:\n",
                "    raise SystemExit(rc)\n",
            ],
        },
        {
            "cell_type": "code",
            "id": _cid(),
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import pathlib\n",
                "root = pathlib.Path('/kaggle/working/results')\n",
                "files = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())\n",
                "print('artifact_files', len(files))\n",
                "for path in files[:200]:\n",
                "    print(path)\n",
                "status_md = root / 'STATUS.md'\n",
                "print(status_md.read_text()[:4000] if status_md.exists() else 'no STATUS.md')\n",
                "summary = root / 'live' / 'live_summary.json'\n",
                "print(summary.read_text()[:2000] if summary.exists() else 'no live_summary')\n",
                "status = root / 'stage_status.json'\n",
                "print(status.read_text() if status.exists() else 'no status')\n",
            ],
        },
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    user = os.environ.get("ATML_KAGGLE_USER", DEFAULT_KAGGLE_USER).strip() or DEFAULT_KAGGLE_USER
    code_b64 = build_code_zip_b64()
    print("kaggle_user", user)
    print("embedded_code_bytes", len(base64.b64decode(code_b64)))
    stages = {
        "atml-pa3-t3-smoke": ("smoke", "atml-pa3-t3-smoke"),
        "atml-pa3-t3-live": ("live", "atml-pa3-t3-live"),
        "atml-pa3-t3-offline": ("offline", "atml-pa3-t3-offline"),
    }
    out_root = ROOT / "kaggle" / "kernels"
    out_root.mkdir(parents=True, exist_ok=True)
    dataset_sources = []
    # Attach official data dataset if/when uploaded.
    data_ds = os.environ.get("ATML_KAGGLE_DATA_DATASET", f"{user}/atml-pa3-data").strip()
    if data_ds:
        dataset_sources = [data_ds]
    for folder_name, (stage, slug) in stages.items():
        folder = out_root / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        notebook = build_notebook(stage, slug, code_b64)
        code_file = f"{folder_name}.ipynb"
        (folder / code_file).write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")
        # Offline needs the parquet dataset; live/smoke keep internet for HF.
        attach_data = stage == "offline" or bool(os.environ.get("ATML_KAGGLE_DATA_DATASET"))
        meta = {
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": code_file,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "keywords": [],
            "dataset_sources": dataset_sources if attach_data else [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
            "machine_shape": "NvidiaTeslaT4",
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("wrote", folder, "datasets=", meta["dataset_sources"])


if __name__ == "__main__":
    main()

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
                "\nSelf-contained ATML PA3 Task 2 runner. Source is embedded in the notebook. "
                "Artifacts go to `/kaggle/working/results`. Internet is required to download "
                "COCO/VQAv2 files and the SigLIP and Qwen weights.\n",
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
                "dst = pathlib.Path('/kaggle/working/task2')\n",
                "if dst.exists():\n",
                "    shutil.rmtree(dst)\n",
                "dst.mkdir(parents=True)\n",
                "print('Extracting embedded code zip')\n",
                f"CODE_B64 = '''{code_b64}'''\n",
                "raw = base64.b64decode(CODE_B64)\n",
                "with zipfile.ZipFile(io.BytesIO(raw)) as zf:\n",
                "    zf.extractall(dst)\n",
                "assert (dst / 'src').is_dir() and (dst / 'configs').is_dir(), list(dst.iterdir())\n",
                "vlm_src = (dst / 'src' / 'models' / 'vlm.py').read_text(encoding='utf-8')\n",
                "siglip_src = (dst / 'src' / 'models' / 'siglip.py').read_text(encoding='utf-8')\n",
                "print('HAS_RUN_LM_FORWARD', 'def run_lm_forward' in vlm_src)\n",
                "print('HAS_THINKING_OFF', 'enable_thinking' in vlm_src and 'False' in vlm_src)\n",
                "print('LAYERNORM_ONCE', 'exactly once' in siglip_src)\n",
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
                "subprocess.call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'])\n",
                "pkgs = [\n",
                "    'transformers>=4.51.0', 'peft==0.14.0', 'accelerate', 'sentencepiece', 'protobuf',\n",
                "    'pyyaml', 'safetensors', 'pillow', 'tqdm', 'pandas', 'matplotlib', 'huggingface_hub',\n",
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
                "import os, sys, pathlib, json, time, traceback, shutil\n",
                "os.environ['ATML_ARTIFACTS_ROOT'] = '/kaggle/working/results'\n",
                "os.environ['HF_HOME'] = '/tmp/hf_home'\n",
                "os.environ['HF_HUB_CACHE'] = '/tmp/hf_home'\n",
                "os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_home'\n",
                "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n",
                "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')\n",
                "if STAGE == 'smoke':\n",
                "    os.environ['ATML_SMOKE'] = '1'\n",
                "pathlib.Path('/kaggle/working/results').mkdir(parents=True, exist_ok=True)\n",
                "t0 = time.time()\n",
                "cmd = [sys.executable, '-m', 'src.run_stage', '--stage', STAGE, '--artifacts-root', '/kaggle/working/results']\n",
                "print('Running', cmd, flush=True)\n",
                "try:\n",
                "    rc = subprocess.call(cmd)\n",
                "except Exception:\n",
                "    pathlib.Path('/kaggle/working/results/failure.txt').write_text(traceback.format_exc())\n",
                "    raise\n",
                "print('RC', rc, 'elapsed_min', round((time.time() - t0) / 60, 2))\n",
                "status = {'stage': STAGE, 'returncode': rc, 'elapsed_sec': time.time() - t0}\n",
                "pathlib.Path('/kaggle/working/results/stage_status.json').write_text(json.dumps(status, indent=2))\n",
                "shutil.rmtree('/kaggle/working/task2', ignore_errors=True)\n",
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
                "for path in files[:160]:\n",
                "    print(path)\n",
                "status = root / 'stage_status.json'\n",
                "print(status.read_text() if status.exists() else 'no status')\n",
            ],
        },
    ]
    # The run cell references STAGE and subprocess. Import subprocess there.
    cells[3]["source"][0] = "import os, sys, subprocess, pathlib, json, time, traceback, shutil\n"
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
        "atml-pa3-t2-smoke": ("smoke", "atml-pa3-t2-smoke"),
        "atml-pa3-t2-pipeline": ("pipeline", "atml-pa3-t2-pipeline"),
    }
    out_root = ROOT / "kaggle" / "kernels"
    out_root.mkdir(parents=True, exist_ok=True)
    for folder_name, (stage, slug) in stages.items():
        folder = out_root / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        notebook = build_notebook(stage, slug, code_b64)
        code_file = f"{folder_name}.ipynb"
        (folder / code_file).write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")
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
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
            "machine_shape": "NvidiaTeslaT4",
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("wrote", folder)


if __name__ == "__main__":
    main()

from __future__ import annotations

import base64
import io
import json
import uuid
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _cid() -> str:
    return uuid.uuid4().hex[:12]


def build_code_zip_b64() -> str:
    buf = io.BytesIO()
    include = ["src", "configs", "assets", "requirements.txt", "README.md"]
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in include:
            path = ROOT / name
            if path.is_dir():
                for f in path.rglob("*"):
                    if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                        zf.write(f, f.relative_to(ROOT).as_posix())
            elif path.exists():
                zf.write(path, path.name)
        zf.writestr("results/.gitkeep", "")
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
                "\nSelf-contained ATML PA3 Task 1 runner. Source is embedded in the notebook (and also attached as a dataset when available). Artifacts go to `/kaggle/working/results`.\n",
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
                "SMOKE_STEPS = int(os.environ.get('ATML_SMOKE_STEPS', '4'))\n",
                "dst = pathlib.Path('/kaggle/working/task1')\n",
                "if dst.exists():\n",
                "    shutil.rmtree(dst)\n",
                "dst.mkdir(parents=True)\n",
                "\n",
                "# ALWAYS use the notebook-embedded code zip. An attached code dataset may be stale\n",
                "# and previously overrode critical bugfixes (FP16 / checkpoint size / smoke scope).\n",
                "print('Extracting embedded code zip (ignoring attached dataset code)')\n",
                f"CODE_B64 = '''{code_b64}'''\n",
                "raw = base64.b64decode(CODE_B64)\n",
                "with zipfile.ZipFile(io.BytesIO(raw)) as zf:\n",
                "    zf.extractall(dst)\n",
                "assert (dst/'src').is_dir() and (dst/'configs').is_dir(), list(dst.iterdir())\n",
                "# Prove which smoke path is embedded\n",
                "smoke_src = (dst/'src'/'run_stage.py').read_text(encoding='utf-8')\n",
                "print('SMOKE_HAS_LORA_ONLY', 'LoRA-only smoke' in smoke_src or 'lora_r4' in smoke_src and 'full_ft' in smoke_src)\n",
                "print('TRAINER_HAS_NO_FP32_CAST', 'p.data = p.data.float()' not in (dst/'src'/'train'/'trainer.py').read_text(encoding='utf-8'))\n",
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
                "    print(torch.cuda.get_device_name(0), 'VRAM_GiB', round(torch.cuda.get_device_properties(0).total_memory/2**30, 2))\n",
                "# Kaggle's torchao (0.10) is incompatible with recent peft, which raises\n",
                "# ImportError during LoRA inject instead of skipping torchao dispatch.\n",
                "subprocess.call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'])\n",
                "pkgs = [\n",
                "    'datasets', 'peft==0.14.0', 'accelerate', 'sentencepiece', 'pyyaml',\n",
                "    'lm-eval', 'einops', 'safetensors', 'matplotlib', 'seaborn', 'pandas', 'tabulate',\n",
                "]\n",
                "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'transformers>=4.51.0'] + pkgs)\n",
            ],
        },
        {
            "cell_type": "code",
            "id": _cid(),
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import os, subprocess, sys, pathlib, json, time, traceback, shutil\n",
                "os.environ['ATML_ARTIFACTS_ROOT'] = '/kaggle/working/results'\n",
                "os.environ['HF_HOME'] = '/tmp/hf_home'\n",
                "os.environ['HF_HUB_CACHE'] = '/tmp/hf_home'\n",
                "os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_home'\n",
                "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n",
                "pathlib.Path('/kaggle/working/results').mkdir(parents=True, exist_ok=True)\n",
                "t0 = time.time()\n",
                "cmd = [sys.executable, '-m', 'src.run_stage', '--stage', STAGE, '--artifacts-root', '/kaggle/working/results']\n",
                "if STAGE == 'smoke':\n",
                "    cmd += ['--smoke-steps', str(SMOKE_STEPS)]\n",
                "print('Running', cmd, flush=True)\n",
                "try:\n",
                "    rc = subprocess.call(cmd)\n",
                "except Exception:\n",
                "    pathlib.Path('/kaggle/working/results/failure.txt').write_text(traceback.format_exc())\n",
                "    raise\n",
                "print('RC', rc, 'elapsed_min', round((time.time()-t0)/60, 2))\n",
                "status = {'stage': STAGE, 'returncode': rc, 'elapsed_sec': time.time()-t0}\n",
                "pathlib.Path('/kaggle/working/results/stage_status.json').write_text(json.dumps(status, indent=2))\n",
                "# Keep working directory small for output downloads\n",
                "shutil.rmtree('/kaggle/working/task1', ignore_errors=True)\n",
                "shutil.rmtree('/kaggle/working/hf_home', ignore_errors=True)\n",
                "if rc != 0:\n",
                "    # Capture tail of process by re-running is impossible; rely on logs + stage_status\n",
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
                "for p in files[:120]:\n",
                "    print(p)\n",
                "status = (root/'stage_status.json')\n",
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
    code_b64 = build_code_zip_b64()
    print("embedded_code_bytes", len(base64.b64decode(code_b64)))
    # folder_name -> (stage, title/slug used by Kaggle)
    # Live slugs for already-created kernels must be reused or pushes create duplicates.
    stages = {
        "atml-pa3-t1-smoke": ("smoke", "atml-pa3-t1-smoke"),
        "atml-pa3-t1-pipeline-prep": ("pipeline_prep", "atml-pa3-t1-prep-baseline"),
        "atml-pa3-t1-train-full-ft": ("train:full_ft", "atml-pa3-t1-train-fullft"),
        "atml-pa3-t1-train-full-ft-l2sp": ("train:full_ft_l2sp", "atml-pa3-t1-train-full-ft-l2sp"),
        "atml-pa3-t1-train-lora-r4": ("train:lora_r4", "atml-pa3-t1-train-lora-r4"),
        "atml-pa3-t1-train-lora-r16": ("train:lora_r16", "atml-pa3-t1-train-lora-r16"),
        "atml-pa3-t1-train-oplora": ("train:oplora", "atml-pa3-t1-train-oplora"),
        "atml-pa3-t1-pipeline-post": ("pipeline_post", "atml-pa3-t1-pipeline-post"),
    }
    out_root = ROOT / "kaggle" / "kernels"
    out_root.mkdir(parents=True, exist_ok=True)
    for folder_name, (stage, slug) in stages.items():
        title = slug
        folder = out_root / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        nb = build_notebook(stage, title, code_b64)
        code_file = f"{folder_name}.ipynb"
        (folder / code_file).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        meta = {
            "id": f"mtayyabtahir/{slug}",
            "title": title,
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

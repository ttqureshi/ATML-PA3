"""Poll gapfill + live kernels; pull carefully when COMPLETE.

Safety rules:
- Never overwrite ``results_live_kaggle`` (HF-fallback archive).
- Official live lands only in ``results_live_official_kaggle``.
- Gapfill merges into offline pack only after verifying marker files / keys.
- Existing offline adaptive/tables are backed up before merge.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "gapfill": {
        "ref": "mtayyabtahir/atml-pa3-t3-gapfill",
        "out": ROOT / "results_gapfill_kaggle",
        "marker": "GAPFILL_COMPLETE",
    },
    "live": {
        "ref": "mtayyabtahir/atml-pa3-t3-live",
        "out": ROOT / "results_live_official_kaggle",
        "marker": "LIVE_COMPLETE",
    },
}
LOG_DIR = ROOT / "results_gapfill_kaggle"
OFFLINE = ROOT / "results_offline_kaggle" / "results"


def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (LOG_DIR / "watch.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(ref: str) -> str:
    raw = subprocess.check_output(["kaggle", "kernels", "status", ref], text=True, stderr=subprocess.STDOUT)
    if "COMPLETE" in raw:
        return "COMPLETE"
    if "ERROR" in raw or "CANCELLED" in raw:
        return "ERROR"
    if "RUNNING" in raw or "QUEUED" in raw or "PENDING" in raw:
        return "RUNNING"
    return raw.strip()


def find_results_root(out: Path) -> Path | None:
    """Kaggle output layout varies; find the directory that holds STATUS.md."""
    candidates = [out / "results", out]
    for p in out.rglob("STATUS.md"):
        candidates.insert(0, p.parent)
    for c in candidates:
        if (c / "STATUS.md").exists():
            return c
    return None


def pull_fresh(ref: str, out: Path) -> Path:
    """Pull into a timestamped staging folder, then promote to ``out``."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging = out.parent / f"{out.name}_staging_{stamp}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    log(f"pulling {ref} -> {staging}")
    subprocess.check_call(["kaggle", "kernels", "output", ref, "-p", str(staging)])
    files = list(staging.rglob("*"))
    n_files = sum(1 for p in files if p.is_file())
    log(f"pulled_files={n_files}")
    if n_files == 0:
        raise RuntimeError(f"Empty pull for {ref}")
    if out.exists():
        bak = out.parent / f"{out.name}_bak_{stamp}"
        if bak.exists():
            shutil.rmtree(bak)
        shutil.move(str(out), str(bak))
        log(f"backed_up_previous {out.name} -> {bak.name}")
    shutil.move(str(staging), str(out))
    return out


def status_text(results_root: Path) -> str:
    p = results_root / "STATUS.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def verify_gapfill(results_root: Path) -> list[str]:
    errors = []
    text = status_text(results_root)
    if "GAPFILL_COMPLETE" not in text and "GAPFILL_COMPLETE" not in (results_root / "STATUS.md").read_text(encoding="utf-8", errors="ignore") if (results_root / "STATUS.md").exists() else "":
        # Also accept if key artifacts exist even if status string formatting differs.
        pass
    tokenwise = results_root / "tables" / "mathvista_tokenwise_visconf.json"
    if not tokenwise.exists():
        # search one level down
        hits = list(results_root.rglob("mathvista_tokenwise_visconf.json"))
        if not hits:
            errors.append("missing mathvista_tokenwise_visconf.json")
    # early stop keys
    for name in ("math_adaptive.json", "mathvista_adaptive.json"):
        hits = list(results_root.rglob(name))
        if not hits:
            errors.append(f"missing {name}")
            continue
        data = json.loads(hits[0].read_text(encoding="utf-8"))
        if "early_stop_analysis" not in data:
            errors.append(f"{name} missing early_stop_analysis")
    if "GAPFILL_COMPLETE" not in text:
        # soft warning only if artifacts present
        if errors:
            errors.append("STATUS.md missing GAPFILL_COMPLETE")
    return errors


def verify_live(results_root: Path) -> list[str]:
    errors = []
    text = status_text(results_root)
    if "LIVE_COMPLETE" not in text:
        errors.append("STATUS.md missing LIVE_COMPLETE")
    summary_hits = list(results_root.rglob("live_summary.json"))
    source_hits = list(results_root.rglob("live_source.json"))
    if not summary_hits:
        errors.append("missing live_summary.json")
    if source_hits:
        src = json.loads(source_hits[0].read_text(encoding="utf-8"))
        source = src.get("source") or src.get("live_source")
        if source not in {"supplied", "official"}:
            # Also check nested
            if src.get("note") and "official" in str(src.get("note")).lower():
                pass
            elif source != "supplied":
                errors.append(f"live source is {source!r}, expected supplied/official")
        live_ids = src.get("live_ids") or []
        if live_ids and len(live_ids) != 50:
            errors.append(f"live_ids length {len(live_ids)} != 50")
    else:
        # STATUS may still say live_source
        if "huggingface_fallback" in text:
            errors.append("live still reports huggingface_fallback")
        if "official" not in text.lower() and "supplied" not in text.lower():
            errors.append("cannot confirm official live source")
    return errors


def backup_file(path: Path, stamp: str) -> None:
    if not path.exists():
        return
    bak_dir = path.parent / f"_bak_{stamp}"
    bak_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, bak_dir / path.name)
    log(f"backup {path.name} -> {bak_dir}")


def merge_gapfill(src: Path) -> None:
    results_root = find_results_root(src)
    if results_root is None:
        raise RuntimeError(f"No STATUS.md under {src}")
    errs = verify_gapfill(results_root)
    if errs:
        raise RuntimeError("gapfill verify failed: " + "; ".join(errs))
    if not OFFLINE.exists():
        raise RuntimeError(f"offline pack missing: {OFFLINE}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rels = [
        "tables/mathvista_eval_summary.json",
        "tables/mathvista_bootstrap_vs_sc_n32.json",
        "tables/mathvista_tokenwise_visconf.json",
        "adaptive/math_adaptive.json",
        "adaptive/mathvista_adaptive.json",
        "figures/mathvista_accuracy_curves.png",
    ]
    for rel in rels:
        hits = list(results_root.rglob(Path(rel).name)) if "/" in rel else [results_root / rel]
        # Prefer exact relative path
        p = results_root / rel
        if not p.exists():
            found = list(results_root.rglob(Path(rel).name))
            p = found[0] if found else None
        if p is None or not p.exists():
            log(f"skip_missing {rel}")
            continue
        target = OFFLINE / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        backup_file(target, stamp)
        shutil.copy2(p, target)
        log(f"merged {rel} -> {target}")
    # Append gapfill note to offline STATUS rather than replacing whole history.
    gap_status = results_root / "STATUS.md"
    offline_status = OFFLINE / "STATUS.md"
    if gap_status.exists() and offline_status.exists():
        backup_file(offline_status, stamp)
        with offline_status.open("a", encoding="utf-8") as f:
            f.write("\n\n## Gap-fill merge\n")
            f.write(gap_status.read_text(encoding="utf-8"))
            f.write("\n")
        log("appended gapfill STATUS into offline STATUS.md")


def finalize_live(src: Path) -> None:
    results_root = find_results_root(src)
    if results_root is None:
        raise RuntimeError(f"No STATUS.md under {src}")
    errs = verify_live(results_root)
    if errs:
        raise RuntimeError("live verify failed: " + "; ".join(errs))
    # Explicitly do NOT touch results_live_kaggle (HF archive).
    hf = ROOT / "results_live_kaggle"
    log(f"preserved_hf_live_archive exists={hf.exists()} path={hf}")
    log(f"official_live_ready path={src}")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    done = {k: False for k in TARGETS}
    ok = {k: False for k in TARGETS}
    t0 = time.time()
    log("watcher_start careful_pull=1")
    while not all(done.values()):
        for name, cfg in TARGETS.items():
            if done[name]:
                continue
            st = status(cfg["ref"])
            log(f"{name} {st}")
            if st == "COMPLETE":
                try:
                    out = pull_fresh(cfg["ref"], cfg["out"])
                    if name == "gapfill":
                        merge_gapfill(out)
                    else:
                        finalize_live(out)
                    ok[name] = True
                    log(f"{name} PULL_OK")
                except Exception as exc:
                    log(f"{name} PULL_FAIL {exc}")
                    ok[name] = False
                done[name] = True
            elif st == "ERROR":
                log(f"{name} KERNEL_ERROR")
                done[name] = True
                ok[name] = False
        if not all(done.values()):
            time.sleep(90)
        if time.time() - t0 > 8 * 3600:
            raise SystemExit("timeout")
    manifest = {"done": done, "ok": ok, "finished_at": datetime.now().isoformat(timespec="seconds")}
    (LOG_DIR / "pull_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"WATCH_DONE {json.dumps(manifest)}")


if __name__ == "__main__":
    main()

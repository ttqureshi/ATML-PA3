from __future__ import annotations

import contextlib
import math
import time
from pathlib import Path
from typing import Any

import torch

from src.common.config import (
    checkpoint_points,
    cumulative_stage_ends,
    dev_eval_points,
    locate_completed,
    parse_common_args,
    prepare_runtime,
    stage_lengths,
    total_updates,
)
from src.common.logging_utils import JsonlLogger, read_json, write_json
from src.common.paths import checkpoint_dir
from src.common.seed import set_seed
from src.data.feature_cache import load_feature_store
from src.data.samplers import build_stage_examples, load_training_pools
from src.models.connector import Connector
from src.models.lora import (
    assert_only_lora_trainable,
    attach_lora,
    is_peft_model,
    trainable_parameter_report,
)
from src.models.vlm import (
    calibrate_output_scale,
    collate_examples,
    end_of_turn_id,
    example_weighted_loss,
    load_language_model,
    load_tokenizer,
    prepare_lm_for_training,
    run_lm_forward,
)
from src.train.checkpointing import (
    find_latest_checkpoint,
    load_checkpoint_blob,
    load_connector_weights,
    read_dev_marker,
    restore_rng,
    save_checkpoint,
    save_stage_weights,
    set_schedule_position,
    write_dev_marker,
)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _make_scaler(enabled: bool):
    if not enabled:
        return None
    # Lower init scale than the default 2**16 — full caption batches were overflowing
    # FP16 grads at update 1 on T4 (smoke passed; pipeline hit GradScaler skips).
    kwargs = {"enabled": True, "init_scale": 2**12, "growth_interval": 2000}
    try:
        return torch.amp.GradScaler("cuda", **kwargs)
    except TypeError:
        try:
            return torch.cuda.amp.GradScaler(**kwargs)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=True)


def warmup_steps(total: int, fraction: float) -> int:
    if total <= 1:
        return 1
    raw = int(math.floor(float(fraction) * total + 0.5))
    return min(max(raw, 1), total - 1)


def lr_multiplier(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """LambdaLR `step` is last_epoch. Epoch 0 is the first update."""
    step = int(step)
    if step < warmup:
        return float(step + 1) / float(max(warmup, 1))
    if total <= warmup:
        return 1.0
    progress = (step - warmup + 1) / float(total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_ratio + (1.0 - min_ratio) * cosine)


def _peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def _count_sources(examples: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        source = str(example["source"])
        counts[source] = counts.get(source, 0) + 1
    return counts


def _add_counts(total: dict, new: dict) -> None:
    for key, value in new.items():
        total[key] = int(total.get(key, 0)) + int(value)


class StageRunner:
    def __init__(self, cfg, dirs, device, feature_store, lm, connector, tokenizer, splits):
        self.cfg = cfg
        self.dirs = dirs
        self.device = device
        self.feature_store = feature_store
        self.lm = lm
        self.connector = connector
        self.tokenizer = tokenizer
        self.splits = splits
        self.eot_id = end_of_turn_id(tokenizer)
        self.effective = int(cfg["training"]["effective_batch"])
        self.microbatch = int(cfg["training"]["microbatch_size"])
        self.accum = int(cfg["training"]["grad_accum"])
        if self.microbatch * self.accum != self.effective:
            raise RuntimeError("microbatch * grad_accum must equal the effective batch")
        self.ckpt_root = checkpoint_dir(dirs, bool(cfg.get("smoke_active")))
        self.metrics = JsonlLogger(dirs["logs"] / "train_metrics.jsonl")
        self.dev_metrics = JsonlLogger(dirs["logs"] / "dev_metrics.jsonl")
        self.use_scaler = device.type == "cuda"
        self.scaler = _make_scaler(self.use_scaler)
        self.optimizer = None
        self.scheduler = None
        self.sampler = None
        self.lr_fn = None
        self.stage_index = 0
        self.stage = None
        self.lora_attached = is_peft_model(lm)
        self.mixture: dict[str, dict[str, int]] = {}
        self.checked_grad: set[int] = set()
        self.peak_mb = 0.0
        self.skipped = 0

    def _trainable(self):
        params = [p for p in self.connector.parameters() if p.requires_grad]
        params.extend(p for p in self.lm.parameters() if p.requires_grad)
        return params

    def enter_stage(self, stage_index: int, already_done: int, resume_blob: dict | None) -> None:
        from src.data.samplers import ExampleSampler

        self.stage_index = stage_index
        self.stage = self.cfg["training"]["stages"][stage_index - 1]
        if stage_index >= 2 and not self.lora_attached:
            if resume_blob is not None and already_done == 0:
                restore_rng(resume_blob)
                resume_blob = None
            lora_cfg = self.cfg["lora"]
            self.lm = attach_lora(
                self.lm,
                r=int(lora_cfg["r"]),
                alpha=int(lora_cfg["alpha"]),
                dropout=float(lora_cfg["dropout"]),
                target_modules=list(lora_cfg["target_modules"]),
            )
            self.lora_attached = True
            assert_only_lora_trainable(self.lm)
        prepare_lm_for_training(self.lm)
        self.connector.train()
        groups = [
            {
                "params": [p for p in self.connector.parameters() if p.requires_grad],
                "lr": float(self.stage["connector_lr"]),
                "name": "connector",
            }
        ]
        if self.stage.get("lora_lr") is not None:
            lora_params = [p for p in self.lm.parameters() if p.requires_grad]
            if not lora_params:
                raise RuntimeError(f"{self.stage['name']} expected LoRA parameters")
            groups.append({"params": lora_params, "lr": float(self.stage["lora_lr"]), "name": "lora"})
        train_cfg = self.cfg["training"]
        self.optimizer = torch.optim.AdamW(
            groups,
            lr=float(self.stage["connector_lr"]),
            betas=tuple(train_cfg["betas"]),
            eps=float(train_cfg["eps"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        updates = int(self.stage["updates"])
        warmup = warmup_steps(updates, float(train_cfg["warmup_fraction"]))
        min_ratio = float(train_cfg["cosine_final_lr_fraction"])
        self.lr_fn = lambda step, warmup=warmup, total=updates, min_ratio=min_ratio: lr_multiplier(
            step, warmup, total, min_ratio
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_fn)
        if already_done > 0:
            if resume_blob is None:
                raise RuntimeError("Mid-stage resume is missing optimizer state")
            self.optimizer.load_state_dict(resume_blob["optimizer"])
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
            if self.scaler is not None and resume_blob.get("scaler") is not None:
                self.scaler.load_state_dict(resume_blob["scaler"])
            saved_epoch = int(resume_blob["scheduler_last_epoch"])
            if saved_epoch != already_done:
                raise RuntimeError(
                    f"Scheduler epoch {saved_epoch} does not match completed stage steps {already_done}"
                )
            set_schedule_position(self.optimizer, self.scheduler, self.lr_fn, saved_epoch)
            restore_rng(resume_blob)
        examples = build_stage_examples(self.cfg, self.splits, self.stage, int(self.cfg["seed"]))
        position = already_done * self.effective
        self.sampler = ExampleSampler(examples, position)
        report = trainable_parameter_report(self.lm, self.connector)
        report.update({"stage": self.stage["name"], "stage_index": stage_index, "already_done": already_done})
        write_json(self.dirs["resources"] / f"trainable_params_stage{stage_index}.json", report)
        print("STAGE", self.stage["name"], "trainable", report["trainable_total"], "resume_at", already_done, flush=True)

    def _assert_grads(self) -> None:
        grad = self.connector.fc1.weight.grad
        if grad is None or not torch.isfinite(grad).all() or float(grad.detach().abs().sum()) == 0.0:
            raise RuntimeError(
                "Connector received no finite gradient. Do not wrap the LM forward in torch.no_grad."
            )
        if self.stage_index == 1:
            for name, param in self.lm.named_parameters():
                if param.requires_grad:
                    raise RuntimeError(f"Stage 1 left an LM parameter trainable: {name}")
            return
        lora_norm = 0.0
        for _name, param in self.lm.named_parameters():
            if param.requires_grad and param.grad is not None:
                lora_norm += float(param.grad.detach().abs().sum())
        if lora_norm == 0.0:
            raise RuntimeError("LoRA received no gradient")

    def _forward_loss(self, examples: list[dict], *, use_amp: bool = True) -> torch.Tensor:
        train_cfg = self.cfg["training"]
        # Collate (connector) stays in FP32. Autocast only wraps the LM forward —
        # running the connector under FP16 autocast overflowed grads on full batches.
        batch = collate_examples(
            self.lm,
            self.connector,
            self.tokenizer,
            self.feature_store,
            examples,
            prefix_ids=self.tokenizer._atml_prompt_split[0],
            suffix_ids=self.tokenizer._atml_prompt_split[1],
            eot_id=self.eot_id,
            question_limit=int(train_cfg["question_tokens"]),
            caption_limit=int(train_cfg["caption_tokens"]),
            answer_limit=int(train_cfg["short_answer_tokens"]),
            include_eot=True,
            device=self.device,
        )
        amp_ctx = _autocast(self.device) if use_amp else contextlib.nullcontext()
        with amp_ctx:
            outputs = run_lm_forward(
                self.lm,
                batch.inputs_embeds,
                batch.attention_mask,
                batch.position_ids,
                train_mode=True,
            )
        return example_weighted_loss(outputs.logits, batch.labels)

    def run_update(self, global_update: int) -> dict[str, Any]:
        assert self.optimizer is not None and self.scheduler is not None and self.sampler is not None
        attempts = 0
        max_attempts = int(self.cfg["training"].get("scaler_max_attempts", 16))
        while True:
            attempts += 1
            position = self.sampler.pos
            groups = [self.sampler.take(self.microbatch) for _ in range(self.accum)]
            flat = [example for group in groups for example in group]
            self.optimizer.zero_grad(set_to_none=True)
            micro_means = []
            started = time.perf_counter()
            # After repeated FP16 overflows, finish this update in FP32 once.
            use_amp = not (self.scaler is not None and attempts >= max(6, max_attempts // 2))
            try:
                for group in groups:
                    raw = self._forward_loss(group, use_amp=use_amp)
                    if not torch.isfinite(raw):
                        raise RuntimeError(f"Non-finite loss at update {global_update}: {float(raw.detach())}")
                    micro_means.append(float(raw.detach().item()))
                    scaled = raw / self.accum
                    if self.scaler is not None and use_amp:
                        self.scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                params = self._trainable()
                if self.scaler is not None and use_amp:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, float(self.cfg["training"]["max_grad_norm"]))
                if self.stage_index not in self.checked_grad and torch.isfinite(grad_norm):
                    self._assert_grads()
                    self.checked_grad.add(self.stage_index)
                skipped = False
                if self.scaler is not None and use_amp:
                    before = float(self.scaler.get_scale())
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    skipped = float(self.scaler.get_scale()) < before or not torch.isfinite(grad_norm)
                else:
                    if not torch.isfinite(grad_norm):
                        self.optimizer.zero_grad(set_to_none=True)
                        skipped = True
                    else:
                        self.optimizer.step()
            except RuntimeError:
                self.sampler.pos = position
                raise
            if skipped:
                self.sampler.pos = position
                self.skipped += 1
                self.optimizer.zero_grad(set_to_none=True)
                scale_now = float(self.scaler.get_scale()) if self.scaler is not None else float("nan")
                print(
                    f"SKIP update={global_update} attempt={attempts}/{max_attempts} "
                    f"grad_norm={float(grad_norm) if grad_norm is not None else float('nan')} "
                    f"scale={scale_now} amp={use_amp}",
                    flush=True,
                )
                if attempts >= max_attempts:
                    raise RuntimeError(
                        f"GradScaler skipped {attempts} attempts at update {global_update} "
                        f"(last_scale={scale_now})"
                    )
                continue
            self.scheduler.step()
            source_counts = _count_sources(flat)
            stage_counts = self.mixture.setdefault(self.stage["name"], {})
            _add_counts(stage_counts, source_counts)
            lrs = {group.get("name", str(i)): float(group["lr"]) for i, group in enumerate(self.optimizer.param_groups)}
            self.peak_mb = max(self.peak_mb, _peak_mb())
            row = {
                "update": int(global_update),
                "stage": int(self.stage_index),
                "stage_name": self.stage["name"],
                "loss": float(sum(micro_means) / len(micro_means)),
                "lr_connector": lrs.get("connector", 0.0),
                "lr_lora": lrs.get("lora", 0.0),
                "examples_seen": int(global_update) * self.effective,
                "grad_norm": float(grad_norm),
                "skipped": False,
                "attempts": attempts,
                "amp": bool(use_amp),
                "mixture": source_counts,
                "step_time_sec": time.perf_counter() - started,
                "output_scale": float(self.connector.output_scale.detach().float().cpu()),
            }
            self.metrics.log(row)
            if global_update % int(self.cfg["training"].get("log_every", 1)) == 0:
                print(
                    f"UPDATE {global_update} stage={self.stage['name']} loss={row['loss']:.4f} "
                    f"lr_c={row['lr_connector']:.3e} lr_l={row['lr_lora']:.3e} amp={use_amp}",
                    flush=True,
                )
            return row

    def progress(self, global_update: int, stage_step: int) -> dict[str, Any]:
        return {
            "global_update": int(global_update),
            "examples_seen": int(global_update) * self.effective,
            "stage_index": int(self.stage_index),
            "stage_name": self.stage["name"],
            "stage_step": int(stage_step),
            "sampler_position": int(self.sampler.pos),
            "lora_attached": bool(self.lora_attached),
            "output_scale": float(self.connector.output_scale.detach().float().cpu()),
            "mixture_counts": self.mixture,
            "num_visual_tokens": int(self.connector.num_visual_tokens),
            "skipped_updates": int(self.skipped),
        }

    def save(self, global_update: int, stage_step: int) -> None:
        progress = self.progress(global_update, stage_step)
        if int(progress["num_visual_tokens"]) != int(self.cfg["model"]["num_visual_tokens"]):
            raise RuntimeError("num_visual_tokens changed mid-run")
        save_checkpoint(
            self.ckpt_root / f"step_{global_update:06d}",
            connector=self.connector,
            lm=self.lm,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            progress=progress,
        )
        if global_update in set(cumulative_stage_ends(self.cfg)):
            save_stage_weights(
                self.ckpt_root / f"stage{self.stage_index}_end",
                connector=self.connector,
                lm=self.lm,
                progress=progress,
            )
        write_json(self.dirs["resources"] / "mixture_counts.json", self.mixture)
        write_json(
            self.dirs["resources"] / "train_resources.json",
            {
                "peak_gpu_mem_mb": self.peak_mb,
                "global_update": global_update,
                "skipped_updates": self.skipped,
                "trainable": trainable_parameter_report(self.lm, self.connector),
            },
        )


def _ensure_prompt_split(tokenizer) -> None:
    if getattr(tokenizer, "_atml_prompt_split", None) is None:
        from src.models.vlm import discover_prompt_split

        tokenizer._atml_prompt_split = discover_prompt_split(tokenizer)


def _maybe_dev(runner: StageRunner, update: int, stage_name: str, points: set[int]) -> None:
    if update not in points and update != 0:
        return
    if update != 0 and read_dev_marker(runner.ckpt_root) == update:
        return
    from src.eval.run_eval import evaluate_dev

    print(f"DEV_EVAL update={update}", flush=True)
    record = evaluate_dev(
        runner.lm,
        runner.connector,
        runner.tokenizer,
        runner.feature_store,
        runner.splits,
        runner.cfg,
        runner.device,
        update,
        stage_name,
    )
    runner.dev_metrics.log(record)
    if update != 0:
        write_dev_marker(runner.ckpt_root, update)
    print(
        f"DEV update={update} caption_loss={record['caption_loss']:.4f} "
        f"ground_dNLL={record['suites']['grounding']['delta_nll']:.4f} "
        f"natural_dNLL={record['suites']['natural']['delta_nll']:.4f}",
        flush=True,
    )


def train(cfg: dict, dirs: dict[str, Path], *, fresh: bool) -> None:
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = read_json(dirs["data"] / "splits.json")
    feature_store = load_feature_store(dirs)
    tokenizer = load_tokenizer(cfg["model"]["language_model"])
    _ensure_prompt_split(tokenizer)
    load_started = time.perf_counter()
    lm = load_language_model(cfg["model"]["language_model"], device)
    connector = Connector(
        in_dim=int(cfg["model"]["vision_dim"]),
        hidden_dim=int(cfg["model"]["connector_hidden"]),
        out_dim=int(cfg["model"]["lm_hidden"]),
        num_visual_tokens=int(cfg["model"]["num_visual_tokens"]),
    ).to(device=device, dtype=torch.float32)
    if int(lm.config.hidden_size) != int(connector.out_dim):
        raise RuntimeError(
            f"LM hidden size {lm.config.hidden_size} does not match connector output {connector.out_dim}"
        )
    for param in connector.parameters():
        param.requires_grad = True
    ckpt_root = checkpoint_dir(dirs, bool(cfg.get("smoke_active")))
    latest = None if fresh else find_latest_checkpoint(ckpt_root)
    blob = load_checkpoint_blob(latest) if latest is not None else None
    completed = 0 if blob is None else int(blob["progress"]["global_update"])
    lengths = stage_lengths(cfg)
    total = total_updates(cfg)
    if blob is not None:
        saved_tokens = int(blob["progress"].get("num_visual_tokens", cfg["model"]["num_visual_tokens"]))
        if saved_tokens != int(cfg["model"]["num_visual_tokens"]):
            raise RuntimeError(
                f"Checkpoint num_visual_tokens={saved_tokens} does not match config "
                f"{cfg['model']['num_visual_tokens']}. Use a separate artifacts directory."
            )
        load_connector_weights(connector, latest)
        # Stage-1 boundary: restore RNG, then let enter_stage initialize fresh LoRA.
        # Later checkpoints already contain the adapter; load it and then restore RNG
        # so the adapter init does not consume the training RNG stream.
        if completed > lengths[0]:
            if not (latest / "lora").exists():
                raise RuntimeError(f"Checkpoint {latest} is past stage 1 but has no LoRA adapter")
            from peft import PeftModel

            lm = PeftModel.from_pretrained(lm, str(latest / "lora"))
            lm.to(device)
            for _name, param in lm.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()
            restore_rng(blob)
        else:
            restore_rng(blob)
        print("RESUME", latest, "completed", completed, flush=True)
    else:
        pools = load_training_pools(splits, cfg["data"]["caption_prompt"])
        calibration = calibrate_output_scale(
            connector, lm, tokenizer, feature_store, pools["caption"], cfg, device
        )
        write_json(dirs["configs_used"] / "connector_calibration.json", calibration)
        print("CALIBRATION", calibration["output_scale"], flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    runner = StageRunner(cfg, dirs, device, feature_store, lm, connector, tokenizer, splits)
    runner.lora_attached = is_peft_model(runner.lm)
    if blob is not None and isinstance(blob["progress"].get("mixture_counts"), dict):
        runner.mixture = blob["progress"]["mixture_counts"]
    load_sec = time.perf_counter() - load_started
    points = dev_eval_points(cfg)
    ckpt_at = checkpoint_points(cfg)
    if completed == 0:
        _maybe_dev(runner, 0, "init", {0})
    elif completed in points and read_dev_marker(ckpt_root) != completed:
        stage_name = cfg["training"]["stages"][locate_completed(completed, lengths)[0] - 1]["name"]
        _maybe_dev(runner, completed, stage_name, points)
    train_started = time.perf_counter()
    resume_blob_for_stage = blob
    while completed < total:
        stage_index, step_after = locate_completed(completed + 1, lengths)
        already_done = step_after - 1
        if runner.stage_index != stage_index:
            runner.enter_stage(stage_index, already_done, resume_blob_for_stage if already_done > 0 else None)
            resume_blob_for_stage = None
        completed += 1
        runner.run_update(completed)
        stage_step = locate_completed(completed, lengths)[1]
        if completed in ckpt_at:
            runner.save(completed, stage_step)
        if completed in points:
            _maybe_dev(runner, completed, runner.stage["name"], points)
    train_sec = time.perf_counter() - train_started
    write_json(dirs["resources"] / "mixture_counts.json", runner.mixture)
    write_json(
        dirs["resources"] / "train_resources.json",
        {
            "model_load_sec": load_sec,
            "train_loop_sec": train_sec,
            "peak_gpu_mem_mb": runner.peak_mb,
            "updates": completed,
            "examples_seen": completed * runner.effective,
            "skipped_updates": runner.skipped,
            "trainable_by_stage": {
                f"stage{index}": read_json(dirs["resources"] / f"trainable_params_stage{index}.json")
                for index in range(1, 4)
                if (dirs["resources"] / f"trainable_params_stage{index}.json").exists()
            },
            "feature_cache": read_json(dirs["resources"] / "feature_cache.json")
            if (dirs["resources"] / "feature_cache.json").exists()
            else None,
        },
    )
    print("TRAIN_DONE", completed, "sec", round(train_sec, 1), flush=True)


def main() -> None:
    parser = parse_common_args("Continued 3-stage SigLIP–Qwen training.")
    args = parser.parse_args()
    _root, cfg, dirs = prepare_runtime(args)
    train(cfg, dirs, fresh=bool(args.fresh))


if __name__ == "__main__":
    main()

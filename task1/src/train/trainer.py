from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast

from src.common.logging_utils import JsonlLogger, write_json
from src.common.seed import seed_worker
from src.data.dataset import MetaMathSFTDataset, sft_collate_fn
from src.models.load_qwen import count_trainable_parameters, load_base_model, load_tokenizer
from src.models.lora import attach_lora, lora_parameter_report
from src.models.oplora import attach_oplora, oplora_parameter_report, verify_oplora_constraints
from src.train.checkpointing import save_training_checkpoint
from src.train.optimizer import (
    build_cosine_scheduler,
    build_optimizer,
    collect_starting_point,
    full_model_displacement,
    l2_sp_penalty,
)


class SFTTrainer:
    def __init__(
        self,
        shared_cfg: dict,
        variant_cfg: dict,
        artifacts_dirs: dict[str, Path],
        device: torch.device | None = None,
        resume_from: Path | None = None,
    ):
        self.shared = shared_cfg
        self.variant = variant_cfg
        self.dirs = artifacts_dirs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resume_from = resume_from

        self.model_name = shared_cfg["model_name"]
        self.method = variant_cfg["method"]
        self.variant_name = variant_cfg["name"]
        self.peak_lr = float(variant_cfg["peak_lr"])
        self.lambda_sp = float(variant_cfg.get("lambda_sp", 0.0))

        train_cfg = shared_cfg["training"]
        self.microbatch = int(train_cfg["microbatch_size"])
        self.grad_accum = int(train_cfg["grad_accum"])
        # Full FT on ~15GB Kaggle GPUs: keep effective batch 32 with microbatch 1.
        if self.method in {"full_ft", "full_ft_l2sp"} and self.microbatch > 1:
            scale = self.microbatch
            self.microbatch = 1
            self.grad_accum = int(self.grad_accum * scale)
        self.num_updates = int(train_cfg["num_updates"])
        self.warmup = int(train_cfg["warmup_updates"])
        self.max_grad_norm = float(train_cfg["max_grad_norm"])
        self.max_seq_len = int(train_cfg["max_seq_len"])
        self.log_every = int(train_cfg.get("log_every", 10))
        self.ckpt_updates = set(train_cfg["checkpoint_update_counts"])
        # Full-FT on Kaggle: only keep eval-needed steps as model-only (~1.2GB each).
        # Prior runs saved optimizer at 312/625/937/1250 (~2.7GB×4) → mid/final got
        # pruned from kernel output and eval fell back to old NaN Drive weights.
        if self.method == "full_ft":
            self.ckpt_updates = {625, 1250}
        elif self.method == "full_ft_l2sp":
            self.ckpt_updates = {1250}

        self.tokenizer = load_tokenizer(self.model_name)
        self.metrics_logger = JsonlLogger(
            self.dirs["logs"] / f"train_metrics_{self.variant_name}.jsonl"
        )
        self.resource_logger = JsonlLogger(
            self.dirs["logs"] / f"resources_{self.variant_name}.jsonl"
        )

        self.theta0 = None
        self.theta0_path = None
        self.oplora_meta = None
        # GradScaler only accepts FP32 gradients. Prior full-FT used pure FP16 weights
        # with scaler OFF → overflow → NaN from step 2. Fix: standard AMP for full_ft*
        # (FP32 masters + autocast + GradScaler + AdamW8bit, microbatch=1).
        # AdamW8bit is required on 15GB T4 — FP32 Adam states OOM around step ~25.
        # LoRA/OPLoRA keep a frozen FP16 backbone with FP32 adapters (+ GradScaler).
        self.use_scaler = self.device.type == "cuda"
        # Lower init_scale avoids a long streak of warmup skips on full-FT.
        init_scale = 2**10 if self.method in {"full_ft", "full_ft_l2sp"} else 2**16
        try:
            self.scaler = GradScaler("cuda", enabled=self.use_scaler, init_scale=init_scale)
        except TypeError:
            try:
                self.scaler = GradScaler(enabled=self.use_scaler, init_scale=init_scale)
            except TypeError:
                self.scaler = GradScaler(enabled=self.use_scaler)
        self.skipped_updates = 0
        self.consecutive_skips = 0
        # Abort before burning a full GPU session on a dead NaN run.
        self.max_consecutive_skips = int(os.environ.get("ATML_MAX_CONSECUTIVE_SKIPS", "25"))

        self._build_model()
        self._build_data()
        self._build_optim()

        self.global_step = 0
        self.examples_seen = 0
        self.peak_mem_bytes = 0

        if resume_from is not None:
            self._resume(resume_from)

    def _build_model(self) -> None:
        if self.method in {"full_ft", "full_ft_l2sp"}:
            # Standard mixed precision: FP32 params so GradScaler can unscale.
            # microbatch forced to 1 above; grad checkpointing keeps T4 15GB feasible.
            model = load_base_model(
                self.model_name,
                torch_dtype=torch.float32,
                gradient_checkpointing=True,
                device=None,
            )
            model = model.to(self.device)
            self.theta0_path = None
            if self.method == "full_ft_l2sp":
                self.theta0 = collect_starting_point(model)
            else:
                self.theta0 = None
            self.param_report = count_trainable_parameters(model)
            print(
                "FULL_FT_MEM_MODE",
                {
                    "dtype": "float32",
                    "amp": "autocast_fp16+GradScaler",
                    "optimizer": "AdamW8bit",
                    "use_scaler": self.use_scaler,
                    "scaler_enabled": bool(
                        getattr(self.scaler, "is_enabled", lambda: self.use_scaler)()
                    ),
                    "max_consecutive_skips": self.max_consecutive_skips,
                    "microbatch": self.microbatch,
                    "grad_accum": self.grad_accum,
                },
                flush=True,
            )
            self.model = model
            write_json(
                self.dirs["resources"] / f"params_{self.variant_name}.json",
                self.param_report,
            )
            return

        model = load_base_model(
            self.model_name,
            torch_dtype=torch.float16,
            gradient_checkpointing=True,
            device=None,
        )

        if self.method == "lora":
            lcfg = self.variant["lora"]
            model = attach_lora(
                model,
                r=int(lcfg["r"]),
                alpha=int(lcfg["alpha"]),
                dropout=float(lcfg.get("dropout", 0.05)),
                target_modules=lcfg.get("target_modules"),
            )
            model = model.to(self.device)
            self.param_report = lora_parameter_report(model)
        elif self.method == "oplora":
            lcfg = self.variant["lora"]
            ocfg = self.variant["oplora"]
            model = model.to(self.device)
            model, self.oplora_meta = attach_oplora(
                model,
                r=int(lcfg["r"]),
                alpha=int(lcfg["alpha"]),
                k=int(ocfg["protection_rank_k"]),
                dropout=float(lcfg.get("dropout", 0.05)),
                target_modules=lcfg.get("target_modules"),
            )
            write_json(
                self.dirs["resources"] / f"oplora_basis_{self.variant_name}.json",
                self.oplora_meta,
            )
            constraint = verify_oplora_constraints(model)
            write_json(
                self.dirs["diagnostics"] / f"oplora_constraint_init_{self.variant_name}.json",
                constraint,
            )
            self.param_report = oplora_parameter_report(model)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.model = model
        write_json(
            self.dirs["resources"] / f"params_{self.variant_name}.json",
            self.param_report,
        )

    def _build_data(self) -> None:
        data_path = self.dirs["data"] / "metamath_40k.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Missing prepared data at {data_path}. Run prepare_metamath first."
            )
        self.dataset = MetaMathSFTDataset(
            data_path,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            enable_thinking=self.shared.get("enable_thinking", False),
        )
        g = torch.Generator()
        g.manual_seed(self.shared["seed"])
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.microbatch,
            shuffle=False,  # source order, no shuffle
            num_workers=0,  # more reliable on Kaggle; avoids worker IPC issues
            pin_memory=self.device.type == "cuda",
            collate_fn=lambda feats: sft_collate_fn(feats, self.tokenizer.pad_token_id),
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=False,
        )

    def _build_optim(self) -> None:
        wd = float(self.shared["training"]["weight_decay"])
        betas = tuple(self.shared["training"]["betas"])
        eps = float(self.shared["training"]["eps"])
        use_8bit = self.method in {"full_ft", "full_ft_l2sp"} and self.device.type == "cuda"
        self.optimizer = build_optimizer(
            self.model,
            lr=self.peak_lr,
            weight_decay=wd,
            betas=betas,
            eps=eps,
            adam8bit=use_8bit,
        )
        print(
            "OPTIMIZER",
            {"class": type(self.optimizer).__name__, "adam8bit": use_8bit, "eps": eps},
            flush=True,
        )
        self.scheduler = build_cosine_scheduler(
            self.optimizer, num_updates=self.num_updates, warmup_updates=self.warmup
        )

    def _resume(self, checkpoint_dir: Path) -> None:
        from src.train.checkpointing import load_trainer_state, restore_rng_from_checkpoint

        state = load_trainer_state(checkpoint_dir)
        self.global_step = int(state["step"])
        self.examples_seen = int(state["examples_seen"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])
        restore_rng_from_checkpoint(checkpoint_dir)

        model_dir = checkpoint_dir / "model"
        if self.method == "lora":
            # Reload PEFT adapters
            from peft import PeftModel
            from src.models.lora import _neutralize_torchao_check

            _neutralize_torchao_check()
            base = load_base_model(
                self.model_name, torch_dtype=torch.float16, gradient_checkpointing=True
            )
            self.model = PeftModel.from_pretrained(base, model_dir, is_trainable=True)
            self.model = self.model.to(self.device)
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        elif self.method == "oplora":
            adapter_path = model_dir / "oplora_adapters.pt"
            legacy = model_dir / "oplora_state.pt"
            blob = torch.load(
                adapter_path if adapter_path.exists() else legacy,
                map_location="cpu",
                weights_only=False,
            )
            if adapter_path.exists():
                name_to_mod = dict(self.model.named_modules())
                for key, tensor in blob.items():
                    if key.endswith(".lora_A"):
                        name_to_mod[key[: -len(".lora_A")]].lora_A.data.copy_(tensor)
                    elif key.endswith(".lora_B"):
                        name_to_mod[key[: -len(".lora_B")]].lora_B.data.copy_(tensor)
            else:
                self.model.load_state_dict(blob, strict=False)
            self.model = self.model.to(self.device)
        else:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=torch.float32, trust_remote_code=True
            ).to(self.device)
            if self.theta0_path and self.theta0_path.exists():
                self.theta0 = torch.load(self.theta0_path, map_location="cpu")

    def _maybe_update_peak_mem(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            self.peak_mem_bytes = max(self.peak_mem_bytes, torch.cuda.max_memory_allocated())

    def train(self) -> dict[str, Any]:
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        data_iter = iter(self.loader)
        micro_in_epoch = 0
        # Skip already-consumed microbatches when resuming
        target_micro = self.examples_seen // self.microbatch

        pbar = tqdm(total=self.num_updates, initial=self.global_step, desc=f"train:{self.variant_name}")

        while self.global_step < self.num_updates:
            self.optimizer.zero_grad(set_to_none=True)
            token_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            token_count = torch.zeros((), device=self.device, dtype=torch.float32)
            omega_val = None
            skipped = False

            for _ in range(self.grad_accum):
                # Advance iterator, skipping if needed for resume
                while True:
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.loader)
                        batch = next(data_iter)
                    if micro_in_epoch < target_micro:
                        micro_in_epoch += 1
                        continue
                    micro_in_epoch += 1
                    break

                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                # Autocast + GradScaler for all CUDA methods (including FP16 full-FT).
                amp_on = self.use_scaler and self.device.type == "cuda"
                try:
                    amp_ctx = autocast("cuda", enabled=amp_on, dtype=torch.float16)
                except TypeError:
                    amp_ctx = autocast(enabled=amp_on, dtype=torch.float16)
                with amp_ctx:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    labels = batch["labels"]
                    n_tok = int((labels != -100).sum().item())
                    if n_tok == 0:
                        loss_sum = torch.zeros((), device=self.device)
                    else:
                        loss_sum = outputs.loss.float() * n_tok

                token_loss_sum = token_loss_sum + loss_sum.detach()
                token_count = token_count + float(n_tok)

                # Accumulate Σloss grads; divide by total tokens after the accum loop.
                if self.use_scaler:
                    self.scaler.scale(loss_sum).backward()
                else:
                    loss_sum.backward()
                self.examples_seen += batch["input_ids"].size(0)

            total_tokens = float(token_count.item())
            if total_tokens <= 0:
                raise RuntimeError("Effective batch contained zero supervised tokens")

            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.detach().div_(total_tokens)

            sft_loss = float((token_loss_sum / token_count).item())
            loss_nonfinite = not math.isfinite(sft_loss)

            if self.method == "full_ft_l2sp" and self.lambda_sp > 0:
                omega = l2_sp_penalty(self.model, self.theta0, self.device)
                omega_val = float(omega.detach().item())
                if not math.isfinite(omega_val):
                    loss_nonfinite = True
                if self.use_scaler:
                    self.scaler.scale(self.lambda_sp * omega).backward()
                else:
                    (self.lambda_sp * omega).backward()
            else:
                omega_val = 0.0 if self.method == "full_ft_l2sp" else None

            if self.use_scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            found_inf = bool(loss_nonfinite)
            if not found_inf:
                for p in self.model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        found_inf = True
                        break
            if found_inf:
                self.skipped_updates += 1
                self.consecutive_skips += 1
                skipped = True
                self.optimizer.zero_grad(set_to_none=True)
                if self.use_scaler:
                    self.scaler.update()
            else:
                self.consecutive_skips = 0
                if self.use_scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()

            self.global_step += 1
            self._maybe_update_peak_mem()
            lr = self.scheduler.get_last_lr()[0]

            record = {
                "variant": self.variant_name,
                "method": self.method,
                "step": self.global_step,
                "examples_seen": self.examples_seen,
                "sft_loss": sft_loss,
                "lr": lr,
                "grad_norm": float(grad_norm) if not skipped else None,
                "skipped": skipped,
                "consecutive_skips": self.consecutive_skips,
                "tokens_in_effective_batch": total_tokens,
            }
            if omega_val is not None:
                record["omega"] = omega_val
                record["lambda_omega"] = self.lambda_sp * omega_val
                record["total_loss"] = (
                    (sft_loss + self.lambda_sp * omega_val)
                    if math.isfinite(sft_loss) and math.isfinite(omega_val)
                    else float("nan")
                )
            self.metrics_logger.log(record)

            if self.consecutive_skips >= self.max_consecutive_skips:
                raise RuntimeError(
                    f"Aborting {self.variant_name}: {self.consecutive_skips} consecutive "
                    f"skipped/NaN updates (total skipped={self.skipped_updates}). "
                    "Full-FT is numerically unstable; fix amp/scaler before retraining."
                )

            if self.global_step % self.log_every == 0 or self.global_step in self.ckpt_updates:
                pbar.set_postfix(
                    loss=f"{sft_loss:.4f}" if math.isfinite(sft_loss) else "nan",
                    lr=f"{lr:.2e}",
                    skip=self.skipped_updates,
                )
            pbar.update(1)

            if self.global_step in self.ckpt_updates:
                # Never persist a NaN full-FT checkpoint (previous runs saved 100% NaN).
                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if p.is_floating_point() and not torch.isfinite(p).all():
                            raise RuntimeError(
                                f"Non-finite weight `{name}` at step {self.global_step}; "
                                "refusing checkpoint save"
                            )
                ckpt_dir = (
                    self.dirs["checkpoints"]
                    / self.variant_name
                    / f"step_{self.global_step:04d}_ex{self.examples_seen}"
                )
                extra = {
                    "skipped_updates": self.skipped_updates,
                    "consecutive_skips": self.consecutive_skips,
                    "use_scaler": self.use_scaler,
                }
                if self.method == "full_ft_l2sp" and self.theta0 is not None:
                    disp = full_model_displacement(self.model, self.theta0)
                    extra["l2_displacement"] = disp
                    write_json(
                        self.dirs["diagnostics"]
                        / f"displacement_{self.variant_name}_step{self.global_step}.json",
                        {"step": self.global_step, "l2_displacement": disp},
                    )
                elif self.method == "full_ft" and self.global_step in {625, 1250}:
                    # Displacement vs freshly loaded base (no multi-GB theta0 dump).
                    try:
                        base_ref = load_base_model(
                            self.model_name,
                            torch_dtype=torch.float16,
                            gradient_checkpointing=False,
                            device="cpu",
                        )
                        ref = collect_starting_point(base_ref)
                        disp = full_model_displacement(self.model, ref)
                        del base_ref, ref
                        extra["l2_displacement"] = disp
                        write_json(
                            self.dirs["diagnostics"]
                            / f"displacement_{self.variant_name}_step{self.global_step}.json",
                            {"step": self.global_step, "l2_displacement": disp},
                        )
                    except Exception as e:
                        extra["l2_displacement_error"] = str(e)
                save_opt = self.method not in {"full_ft", "full_ft_l2sp"}
                meta = save_training_checkpoint(
                    ckpt_dir,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    step=self.global_step,
                    examples_seen=self.examples_seen,
                    variant_name=self.variant_name,
                    method=self.method,
                    model_name=self.model_name,
                    theta0_path=self.theta0_path,
                    oplora_meta=self.oplora_meta,
                    extra=extra,
                    save_optimizer=save_opt,
                )
                if self.method in {"full_ft", "full_ft_l2sp"}:
                    from src.train.checkpointing import prune_unkept_checkpoints

                    prune_unkept_checkpoints(
                        self.dirs["checkpoints"] / self.variant_name,
                        set(self.ckpt_updates),
                    )
                self.resource_logger.log(
                    {
                        "event": "checkpoint",
                        "variant": self.variant_name,
                        **meta,
                        "peak_mem_bytes": self.peak_mem_bytes,
                    }
                )

        pbar.close()
        # Always persist a final checkpoint if the recipe requested any checkpoints.
        if self.ckpt_updates and self.global_step not in self.ckpt_updates:
            ckpt_dir = (
                self.dirs["checkpoints"]
                / self.variant_name
                / f"step_{self.global_step:04d}_ex{self.examples_seen}"
            )
            save_opt = self.method not in {"full_ft", "full_ft_l2sp"}
            save_training_checkpoint(
                ckpt_dir,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                step=self.global_step,
                examples_seen=self.examples_seen,
                variant_name=self.variant_name,
                method=self.method,
                model_name=self.model_name,
                theta0_path=self.theta0_path,
                oplora_meta=self.oplora_meta,
                extra={"skipped_updates": self.skipped_updates, "final": True},
                save_optimizer=save_opt,
            )
            if self.method in {"full_ft", "full_ft_l2sp"}:
                from src.train.checkpointing import prune_unkept_checkpoints

                prune_unkept_checkpoints(
                    self.dirs["checkpoints"] / self.variant_name,
                    set(self.ckpt_updates) | {self.global_step},
                )

        # Gate: Full-FT that emits "!!!!..." is useless for eval (prior NaN symptom).
        if self.method in {"full_ft", "full_ft_l2sp"}:
            try:
                self.model.eval()
                prompt = "Q: What is 2+2?\nA:"
                enc = self.tokenizer(prompt, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.no_grad():
                    out = self.model.generate(
                        **enc,
                        max_new_tokens=32,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                text = self.tokenizer.decode(out[0], skip_special_tokens=True)
                bang_frac = text.count("!") / max(len(text), 1)
                print("FULL_FT_GEN_SMOKE", {"text": text[:200], "bang_frac": bang_frac}, flush=True)
                if bang_frac > 0.5 or set(text.strip()) <= {"!"}:
                    raise RuntimeError(
                        f"Full-FT generation collapsed to bangs/garbage: {text[:120]!r}"
                    )
            except RuntimeError:
                raise
            except Exception as e:
                print("FULL_FT_GEN_SMOKE_WARN", str(e), flush=True)

        elapsed = time.perf_counter() - start
        summary = {
            "variant": self.variant_name,
            "method": self.method,
            "steps": self.global_step,
            "examples_seen": self.examples_seen,
            "train_seconds": elapsed,
            "peak_mem_bytes": self.peak_mem_bytes,
            "peak_mem_gb": self.peak_mem_bytes / (1024**3),
            "trainable_params": self.param_report,
            "skipped_updates": self.skipped_updates,
        }
        write_json(self.dirs["resources"] / f"train_summary_{self.variant_name}.json", summary)
        return summary

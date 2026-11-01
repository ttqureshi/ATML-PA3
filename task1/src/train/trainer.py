from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

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
        self.num_updates = int(train_cfg["num_updates"])
        self.warmup = int(train_cfg["warmup_updates"])
        self.max_grad_norm = float(train_cfg["max_grad_norm"])
        self.max_seq_len = int(train_cfg["max_seq_len"])
        self.log_every = int(train_cfg.get("log_every", 10))
        self.ckpt_updates = set(train_cfg["checkpoint_update_counts"])

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
        try:
            self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")
        except TypeError:
            self.scaler = GradScaler(enabled=self.device.type == "cuda")
        self.skipped_updates = 0

        self._build_model()
        self._build_data()
        self._build_optim()

        self.global_step = 0
        self.examples_seen = 0
        self.peak_mem_bytes = 0

        if resume_from is not None:
            self._resume(resume_from)

    def _build_model(self) -> None:
        # Full FT: FP16 compute + FP32 master via GradScaler/master params approach.
        # Adapter variants: frozen FP16 backbone + FP32 adapters.
        model = load_base_model(
            self.model_name,
            torch_dtype=torch.float16,
            gradient_checkpointing=True,
            device=None,
        )

        if self.method in {"full_ft", "full_ft_l2sp"}:
            # FP16 weights + AMP. AdamW keeps FP32 optimizer states.
            # Only L2-SP keeps a CPU FP32 starting-point copy in RAM (not written to disk:
            # a full theta0.pt is multi-GB and blew Kaggle working storage).
            model = model.to(self.device)
            self.theta0_path = None
            if self.method == "full_ft_l2sp":
                self.theta0 = collect_starting_point(model)
            else:
                self.theta0 = None
            self.param_report = count_trainable_parameters(model)
        elif self.method == "lora":
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
            num_workers=2,
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
        self.optimizer = build_optimizer(
            self.model, lr=self.peak_lr, weight_decay=wd, betas=betas, eps=eps
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

            base = load_base_model(
                self.model_name, torch_dtype=torch.float16, gradient_checkpointing=True
            )
            self.model = PeftModel.from_pretrained(base, model_dir, is_trainable=True)
            self.model = self.model.to(self.device)
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        elif self.method == "oplora":
            state_dict = torch.load(model_dir / "oplora_state.pt", map_location="cpu")
            self.model.load_state_dict(state_dict)
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
                try:
                    amp_ctx = autocast("cuda", enabled=self.device.type == "cuda", dtype=torch.float16)
                except TypeError:
                    amp_ctx = autocast(enabled=self.device.type == "cuda", dtype=torch.float16)
                with amp_ctx:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    # HF returns mean over non -100 tokens in the microbatch.
                    # Reconstruct sum to normalize over the full effective batch later.
                    labels = batch["labels"]
                    n_tok = int((labels != -100).sum().item())
                    if n_tok == 0:
                        loss_sum = torch.zeros((), device=self.device)
                    else:
                        loss_sum = outputs.loss.float() * n_tok

                # Scale for accumulation: we'll divide by total tokens after loop.
                # Store unscaled token-sum contribution; GradScaler scales the tensor we backward.
                token_loss_sum = token_loss_sum + loss_sum.detach()
                token_count = token_count + float(n_tok)

                # Backward with placeholder: we need a differentiable loss.
                # Use (loss_sum / expected_tokens_approx) then correct? Better approach:
                # backward loss_sum / TOTAL_TOKENS_CONSTANT where TOTAL = token_count after loop —
                # but token_count unknown mid-loop. Assignment: normalize over all valid
                # response tokens in the effective batch.
                # Practical approach: accumulate gradients of (loss_sum), then manually
                # divide grads by total_token_count before optimizer step.
                self.scaler.scale(loss_sum).backward()
                self.examples_seen += batch["input_ids"].size(0)

            total_tokens = float(token_count.item())
            if total_tokens <= 0:
                raise RuntimeError("Effective batch contained zero supervised tokens")

            # Convert accumulated Σloss into mean over effective-batch tokens by scaling grads.
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.detach().div_(total_tokens)

            sft_loss = float((token_loss_sum / token_count).item())

            # Optional L2-SP penalty added after SFT grad accumulation.
            if self.method == "full_ft_l2sp" and self.lambda_sp > 0:
                omega = l2_sp_penalty(self.model, self.theta0, self.device)
                omega_val = float(omega.detach().item())
                # grads for lambda * omega
                self.scaler.scale(self.lambda_sp * omega).backward()
            else:
                omega_val = 0.0 if self.method == "full_ft_l2sp" else None

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            # Detect non-finite grads after unscale
            found_inf = False
            for p in self.model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    found_inf = True
                    break
            if found_inf:
                self.skipped_updates += 1
                skipped = True
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
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
                "tokens_in_effective_batch": total_tokens,
            }
            if omega_val is not None:
                record["omega"] = omega_val
                record["lambda_omega"] = self.lambda_sp * omega_val
                record["total_loss"] = sft_loss + self.lambda_sp * omega_val
            self.metrics_logger.log(record)

            if self.global_step % self.log_every == 0 or self.global_step in self.ckpt_updates:
                pbar.set_postfix(loss=f"{sft_loss:.4f}", lr=f"{lr:.2e}")
            pbar.update(1)

            if self.global_step in self.ckpt_updates:
                ckpt_dir = (
                    self.dirs["checkpoints"]
                    / self.variant_name
                    / f"step_{self.global_step:04d}_ex{self.examples_seen}"
                )
                extra = {"skipped_updates": self.skipped_updates}
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
            )

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

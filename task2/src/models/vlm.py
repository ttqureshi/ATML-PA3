from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LMBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    example_ids: list[str]


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_language_model(model_name: str, device: torch.device):
    from transformers import AutoModelForCausalLM

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.config.use_cache = False
    return model


def end_of_turn_id(tokenizer) -> int:
    for token in ("<|im_end|>", getattr(tokenizer, "eos_token", None)):
        if not token:
            continue
        index = tokenizer.convert_tokens_to_ids(token)
        unk = getattr(tokenizer, "unk_token_id", None)
        if index is not None and index != unk:
            return int(index)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no end-of-turn id")
    return int(tokenizer.eos_token_id)


def _coerce_token_ids(out) -> list[int]:
    """Normalize apply_chat_template outputs across transformers versions.

    Recent Qwen/transformers builds may return a BatchEncoding / dict with an
    ``input_ids`` field instead of a plain ``list[int]``.
    """
    if out is None:
        raise RuntimeError("apply_chat_template returned None")
    if isinstance(out, dict) or hasattr(out, "input_ids"):
        if "input_ids" not in out:
            raise RuntimeError(f"Tokenized chat output missing input_ids: {type(out)} keys={list(out)}")
        out = out["input_ids"]
    if isinstance(out, torch.Tensor):
        out = out.tolist()
    if isinstance(out, tuple):
        out = list(out)
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    if not isinstance(out, list):
        raise RuntimeError(f"Unexpected token id container: {type(out)}")
    return [int(token) for token in out]


def _apply_ids(tokenizer, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
        "return_dict": False,
    }
    try:
        out = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            out = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("return_dict", None)
            out = tokenizer.apply_chat_template(messages, **kwargs)
    return _coerce_token_ids(out)


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    width = len(needle)
    for start in range(len(haystack) - width + 1):
        if haystack[start : start + width] == needle:
            return start
    return -1


def discover_prompt_split(tokenizer) -> tuple[list[int], list[int]]:
    """Prefix before the user text, and the generation suffix after it.

    ``enable_thinking=False`` is always requested. Visual tokens are inserted
    between the prefix and the question tokens.
    """
    probes = ("Hello?", "How many cats can you see?")
    encoded = []
    for probe in probes:
        full = _apply_ids(tokenizer, [{"role": "user", "content": probe}], True)
        probe_ids = tokenizer.encode(probe, add_special_tokens=False)
        index = _find_subsequence(full, probe_ids)
        if index < 0:
            raise RuntimeError(
                "Chat template did not contain the probe as a contiguous token span: "
                + tokenizer.decode(full)
            )
        encoded.append((full[:index], full[index + len(probe_ids) :]))
    if encoded[0] != encoded[1]:
        raise RuntimeError("Chat template prefix/suffix depends on the user text; refusing to splice visual tokens")
    prefix, suffix = encoded[0]
    think_id = tokenizer.convert_tokens_to_ids("</think>")
    unk = getattr(tokenizer, "unk_token_id", None)
    if think_id is not None and think_id != unk and int(think_id) not in suffix:
        extra = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
        if extra and int(think_id) in extra:
            suffix = list(suffix) + list(extra)
    print("PROMPT_PREFIX", repr(tokenizer.decode(prefix)), flush=True)
    print("PROMPT_SUFFIX", repr(tokenizer.decode(suffix)), flush=True)
    print("THINKING_DISABLED", "</think>" in tokenizer.decode(suffix) or (
        think_id is not None and int(think_id) in suffix
    ), flush=True)
    return list(prefix), list(suffix)


def _embed_ids(lm, ids: list[int], device: torch.device) -> torch.Tensor:
    hidden = lm.get_input_embeddings().weight.shape[-1]
    if not ids:
        return torch.zeros(0, hidden, device=device, dtype=lm.get_input_embeddings().weight.dtype)
    tokens = torch.tensor(ids, dtype=torch.long, device=device)
    return lm.get_input_embeddings()(tokens)


def example_weighted_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean of per-example token means. Supervised positions are labels != -100.

    Causal shift: logits[t] predicts labels[t + 1].
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    losses = []
    for index in range(shift_logits.size(0)):
        mask = shift_labels[index] != -100
        count = int(mask.sum().item())
        if count == 0:
            raise RuntimeError(f"Example {index} has no supervised tokens")
        losses.append(
            F.cross_entropy(shift_logits[index][mask], shift_labels[index][mask], reduction="mean")
        )
    return torch.stack(losses).mean()


def per_example_nll(logits: torch.Tensor, labels: torch.Tensor) -> list[float]:
    """Teacher-forced mean token NLL. Call this with termination already masked out."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    values: list[float] = []
    for index in range(shift_logits.size(0)):
        mask = shift_labels[index] != -100
        count = int(mask.sum().item())
        if count == 0:
            raise RuntimeError(f"Example {index} has no answer tokens for NLL")
        nll = F.cross_entropy(shift_logits[index][mask], shift_labels[index][mask], reduction="mean")
        values.append(float(nll.item()))
    return values


def run_lm_forward(lm, inputs_embeds, attention_mask, position_ids, *, train_mode: bool):
    """Forward the language model on ``inputs_embeds``.

    This function intentionally does not use ``torch.no_grad``. Frozen weights
    stay frozen via ``requires_grad=False``; connector gradients must still flow
    through ``inputs_embeds``. When the prompt has no visual tokens, the embed
    tensor is marked as requiring grad so checkpointing can reach LoRA.
    """
    embeds = inputs_embeds
    if train_mode and not embeds.requires_grad:
        # Only synthesize a grad-enabled leaf when there is no connector graph.
        # If visual tokens are present, ``requires_grad`` must already be True;
        # detaching here would cut connector gradients.
        if embeds.grad_fn is not None:
            raise RuntimeError(
                "inputs_embeds has a grad_fn but requires_grad=False; refusing to detach "
                "because that would cut connector gradients"
            )
        embeds = embeds.detach().requires_grad_(True)
    return lm(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )


def _token_ids(tokenizer, text: str, limit: int) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return list(ids[:limit])


def collate_examples(
    lm,
    connector,
    tokenizer,
    feature_store,
    examples: list[dict],
    *,
    prefix_ids: list[int],
    suffix_ids: list[int],
    eot_id: int,
    question_limit: int,
    caption_limit: int,
    answer_limit: int,
    include_eot: bool,
    device: torch.device,
) -> LMBatch:
    has_image = [ex.get("image_id") is not None for ex in examples]
    if any(has_image) and not all(has_image):
        raise RuntimeError("Refusing a batch that mixes image and no-image examples")
    embed_dtype = lm.get_input_embeddings().weight.dtype
    visual = None
    if all(has_image):
        feats = torch.stack(
            [feature_store.get(ex["image_split"], int(ex["image_id"])) for ex in examples],
            dim=0,
        )
        # Connector stays FP32; cast only the tokens that enter the LM sequence.
        feats = feats.to(device=device, dtype=torch.float32)
        visual = connector(feats).to(dtype=embed_dtype)
    embed_rows = []
    label_rows = []
    for index, example in enumerate(examples):
        q_ids = _token_ids(tokenizer, example["user_text"], question_limit)
        limit = caption_limit if example["target_kind"] == "caption" else answer_limit
        a_ids = _token_ids(tokenizer, example["target_text"], limit)
        if not a_ids:
            raise RuntimeError(f"Target tokenized to empty for {example['example_id']}")
        prefix = _embed_ids(lm, prefix_ids, device)
        question = _embed_ids(lm, q_ids, device)
        suffix = _embed_ids(lm, suffix_ids, device)
        answer = _embed_ids(lm, a_ids + [eot_id], device)
        pieces = [prefix]
        n_visual = 0
        if visual is not None:
            pieces.append(visual[index])
            n_visual = int(visual.size(1))
        pieces.extend([question, suffix, answer])
        embeds = torch.cat(pieces, dim=0)
        labels = (
            [-100] * prefix.size(0)
            + [-100] * n_visual
            + [-100] * question.size(0)
            + [-100] * suffix.size(0)
            + a_ids
            + ([eot_id] if include_eot else [-100])
        )
        if len(labels) != embeds.size(0):
            raise RuntimeError(
                f"Label/embed length mismatch for {example['example_id']}: {len(labels)} vs {embeds.size(0)}"
            )
        embed_rows.append(embeds)
        label_rows.append(labels)
    return _pad_right(embed_rows, label_rows, [ex["example_id"] for ex in examples])


def _pad_right(embed_rows: list[torch.Tensor], label_rows: list[list[int]], example_ids: list[str]) -> LMBatch:
    batch = len(embed_rows)
    width = max(row.size(0) for row in embed_rows)
    hidden = embed_rows[0].size(1)
    device = embed_rows[0].device
    dtype = embed_rows[0].dtype
    inputs = torch.zeros(batch, width, hidden, device=device, dtype=dtype)
    labels = torch.full((batch, width), -100, device=device, dtype=torch.long)
    mask = torch.zeros(batch, width, device=device, dtype=torch.long)
    positions = torch.zeros(batch, width, device=device, dtype=torch.long)
    for index, (embeds, labs) in enumerate(zip(embed_rows, label_rows)):
        length = embeds.size(0)
        inputs[index, :length] = embeds
        labels[index, :length] = torch.tensor(labs, dtype=torch.long, device=device)
        mask[index, :length] = 1
        positions[index, :length] = torch.arange(length, device=device)
    return LMBatch(inputs, mask, positions, labels, example_ids)


def prompt_embeds_for_generation(
    lm,
    connector,
    tokenizer,
    feature_store,
    examples: list[dict],
    *,
    prefix_ids: list[int],
    suffix_ids: list[int],
    question_limit: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Prompt embeds only (no answer). No-image examples omit visual tokens."""
    prompts: list[torch.Tensor] = []
    embed_dtype = lm.get_input_embeddings().weight.dtype
    image_rows = [i for i, ex in enumerate(examples) if ex.get("image_id") is not None]
    visual = None
    if image_rows:
        feats = torch.stack(
            [feature_store.get(examples[i]["image_split"], int(examples[i]["image_id"])) for i in image_rows],
            dim=0,
        ).to(device=device, dtype=torch.float32)
        visual = connector(feats).to(dtype=embed_dtype)
    cursor = 0
    for example in examples:
        q_ids = _token_ids(tokenizer, example["user_text"], question_limit)
        pieces = [_embed_ids(lm, prefix_ids, device)]
        if example.get("image_id") is not None:
            pieces.append(visual[cursor])
            cursor += 1
        pieces.append(_embed_ids(lm, q_ids, device))
        pieces.append(_embed_ids(lm, suffix_ids, device))
        prompts.append(torch.cat(pieces, dim=0))
    return prompts


def _left_pad(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(rows)
    width = max(row.size(0) for row in rows)
    hidden = rows[0].size(1)
    device = rows[0].device
    dtype = rows[0].dtype
    inputs = torch.zeros(batch, width, hidden, device=device, dtype=dtype)
    mask = torch.zeros(batch, width, device=device, dtype=torch.long)
    positions = torch.zeros(batch, width, device=device, dtype=torch.long)
    for index, row in enumerate(rows):
        length = row.size(0)
        inputs[index, width - length :] = row
        mask[index, width - length :] = 1
        positions[index, width - length :] = torch.arange(length, device=device)
    return inputs, mask, positions


@torch.inference_mode()
def greedy_generate(lm, prompt_rows: list[torch.Tensor], *, max_new_tokens: int, eos_ids: set[int]) -> list[list[int]]:
    """Greedy decoding without a KV cache.

    Each step re-forwards the full left-padded prefix. That keeps position ids
    and the attention mask correct when prompts differ in length. Batch size is
    the caller's (final eval uses 8).
    """
    sequences = [row.detach() for row in prompt_rows]
    generated: list[list[int]] = [[] for _ in sequences]
    finished = [False] * len(sequences)
    embed = lm.get_input_embeddings()
    for _step in range(max_new_tokens):
        inputs, mask, positions = _left_pad(sequences)
        outputs = lm(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
        )
        next_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        if all(finished):
            break
        for index, token in enumerate(next_id.tolist()):
            if finished[index]:
                continue
            if int(token) in eos_ids:
                finished[index] = True
                continue
            generated[index].append(int(token))
            step_embed = embed(next_id[index : index + 1]).squeeze(0)
            sequences[index] = torch.cat([sequences[index], step_embed.unsqueeze(0)], dim=0)
        if all(finished):
            break
    return generated


def prepare_lm_for_training(lm) -> None:
    from src.models.lora import enable_input_require_grads

    lm.config.use_cache = False
    try:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        lm.gradient_checkpointing_enable()
    enable_input_require_grads(lm)
    lm.train()


def calibrate_output_scale(connector, lm, tokenizer, feature_store, caption_examples: list[dict], cfg: dict, device) -> dict:
    """Match connector output L2 to text-embedding L2 using training captions only."""
    connector.eval()
    n_images = min(int(cfg["training"]["calibration_images"]), len(caption_examples))
    chosen = []
    seen = set()
    for example in caption_examples:
        key = (example["image_split"], int(example["image_id"]))
        if key in seen:
            continue
        seen.add(key)
        chosen.append(key)
        if len(chosen) >= n_images:
            break
    if not chosen:
        raise RuntimeError("No training images available for connector calibration")
    token_budget = int(cfg["training"]["calibration_text_tokens"])
    token_ids: list[int] = []
    for example in caption_examples:
        token_ids.extend(tokenizer.encode(example["target_text"], add_special_tokens=False))
        if len(token_ids) >= token_budget:
            break
    token_ids = token_ids[:token_budget]
    if not token_ids:
        raise RuntimeError("No training caption tokens available for connector calibration")
    previous = float(connector.output_scale.detach().float().cpu().item())
    connector.output_scale.fill_(1.0)
    with torch.no_grad():
        feats = torch.stack([feature_store.get(split, image_id) for split, image_id in chosen], dim=0)
        feats = feats.to(device=device, dtype=torch.float32)
        visual = connector(feats).float()
        visual_norm = float(visual.norm(dim=-1).mean().item())
        text = lm.get_input_embeddings()(torch.tensor(token_ids, dtype=torch.long, device=device)).float()
        text_norm = float(text.norm(dim=-1).mean().item())
    scale = 1.0 if visual_norm < 1e-6 else text_norm / visual_norm
    connector.output_scale.fill_(scale)
    return {
        "rule": "output_scale = mean_L2(text_token_embeddings) / mean_L2(connector_outputs_before_scale)",
        "mean_text_l2": text_norm,
        "mean_connector_l2_before_scale": visual_norm,
        "output_scale": float(scale),
        "num_text_tokens": len(token_ids),
        "num_images": len(chosen),
        "num_visual_tokens_per_image": int(visual.size(1)),
        "source": "training captions and training-image features only",
        "seed": int(cfg["seed"]),
        "previous_scale": previous,
    }

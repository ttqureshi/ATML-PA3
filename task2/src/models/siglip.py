from __future__ import annotations

import torch


def load_siglip(model_name: str, device: torch.device):
    """Load the vision tower only. The caller must delete it before Qwen training."""
    from transformers import AutoImageProcessor, SiglipVisionModel

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SiglipVisionModel.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, processor


@torch.inference_mode()
def extract_patch_features(vision_model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return ``[B, 196, 768]`` final patch tokens in 14×14 spatial order.

    ``SiglipVisionModel.forward`` applies ``post_layernorm`` exactly once and
    exposes the result as ``last_hidden_state``. This function does not call
    ``post_layernorm`` again.
    """
    outputs = vision_model(pixel_values=pixel_values)
    tokens = outputs.last_hidden_state
    if tokens.dim() != 3 or tokens.size(1) != 196:
        raise RuntimeError(
            f"Expected SigLIP patch tokens [B, 196, 768], got {tuple(tokens.shape)}. "
            "Do not use the pooled image embedding."
        )
    return tokens


def preprocess_images(processor, images) -> torch.Tensor:
    """Deterministic SigLIP preprocessing (resize / rescale / normalize, no augmentation)."""
    encoded = processor(images=images, return_tensors="pt")
    return encoded["pixel_values"]

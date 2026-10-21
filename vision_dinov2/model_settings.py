"""Shared model selection for memory building and replay (no model download)."""

DEFAULT_MODEL = "facebook/dinov2-small"
DINO3_MODELS = {
    f"facebook/dinov3-{size}-pretrain-lvd1689m"
    for size in ("vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16")
}


def model_settings(config, default_size=518):
    model = config.get("dino_model_name", DEFAULT_MODEL)
    if model != DEFAULT_MODEL and model not in DINO3_MODELS:
        raise ValueError("dino_model_name must be facebook/dinov2-small or a supported DINOv3 LVD ViT checkpoint.")
    patch_size = 16 if model in DINO3_MODELS else 14
    size = int(config.get("dino_input_size", default_size))
    if size < patch_size or size % patch_size:
        raise ValueError(f"dino_input_size must be a positive multiple of {patch_size} for {model}.")
    return model, size

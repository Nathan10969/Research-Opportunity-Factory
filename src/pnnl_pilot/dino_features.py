from __future__ import annotations

import importlib
import sys
from pathlib import Path


OFFICIAL_DINOV3_WIDTHS = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vith16plus": 1280,
    "dinov3_vit7b16": 4096,
    "dinov3_convnext_tiny": 768,
    "dinov3_convnext_small": 768,
    "dinov3_convnext_base": 1024,
    "dinov3_convnext_large": 1536,
}


def resolve_official_dinov3_entrypoint(name: str) -> str:
    if name not in OFFICIAL_DINOV3_WIDTHS:
        allowed = ", ".join(sorted(OFFICIAL_DINOV3_WIDTHS))
        raise ValueError(
            f"unsupported official DINOv3 entrypoint {name!r}; allowed: {allowed}"
        )
    return name


def build_official_dinov3_transform():
    from torchvision.transforms import (
        CenterCrop,
        Compose,
        InterpolationMode,
        Normalize,
        Resize,
        ToTensor,
    )

    return Compose(
        [
            Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop(224),
            ToTensor(),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def load_official_dinov3(repo: Path, entrypoint: str, weights: Path):
    repo = repo.resolve()
    weights = weights.resolve()
    if not (repo / "dinov3" / "hub" / "backbones.py").is_file():
        raise FileNotFoundError(f"official DINOv3 repository is invalid: {repo}")
    if not weights.is_file():
        raise FileNotFoundError(f"official DINOv3 weights are missing: {weights}")
    entrypoint = resolve_official_dinov3_entrypoint(entrypoint)
    repo_text = str(repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    backbones = importlib.import_module("dinov3.hub.backbones")
    constructor = getattr(backbones, entrypoint)
    model = constructor(weights=str(weights))
    return model, OFFICIAL_DINOV3_WIDTHS[entrypoint]


def pool_official_dinov3_outputs(outputs):
    required = {"x_norm_clstoken", "x_norm_patchtokens"}
    missing = required - set(outputs)
    if missing:
        raise KeyError(f"official DINOv3 outputs are missing: {sorted(missing)}")
    cls = outputs["x_norm_clstoken"]
    patch_tokens = outputs["x_norm_patchtokens"]
    if cls.ndim != 2 or patch_tokens.ndim != 3:
        raise ValueError(
            "unexpected official DINOv3 output ranks: "
            f"cls={cls.ndim}, patches={patch_tokens.ndim}"
        )
    if cls.shape[0] != patch_tokens.shape[0] or cls.shape[1] != patch_tokens.shape[2]:
        raise ValueError(
            "official DINOv3 CLS and patch-token shapes are incompatible: "
            f"cls={tuple(cls.shape)}, patches={tuple(patch_tokens.shape)}"
        )
    return cls, patch_tokens.mean(dim=1)

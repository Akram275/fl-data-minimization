"""
Pre-trained, frozen image encoders pulled from the Hugging Face Hub.

Every client runs the SAME frozen encoder locally, so embeddings from different
clients live in one shared space and their sketches are directly comparable.
Nothing is fine-tuned on client data.

Registry (all small enough to run on CPU; uses CUDA automatically when available):
  sd-vae       stabilityai/sd-vae-ft-mse   VAE posterior mean, 4x8x8 latent at
                                           64px input -> 256-d
  dinov2-small facebook/dinov2-small       self-supervised ViT-S/14, CLS token, 384-d
  clip-b32     openai/clip-vit-base-patch32 CLIP image embedding, 512-d
  resnet18     microsoft/resnet-18         ImageNet ResNet-18, pooled features, 512-d
"""
import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
from PIL import Image

# The encoders are PyTorch-only: keep transformers and diffusers from importing
# TensorFlow as a side effect when it happens to be installed.
os.environ.setdefault("USE_TF", "0")

# ImageNet statistics used by DINOv2 / ResNet preprocessing; CLIP has its own.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _default_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _to_tensor(images: List[Image.Image], size: int, mean: np.ndarray, std: np.ndarray,
               device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Resize (bicubic, square - face crops are already roughly centered) and
    normalize. Done in numpy/PIL rather than torchvision, whose compiled image
    ops are broken in this environment."""
    arr = np.stack([
        np.asarray(img.convert('RGB').resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
        for img in images
    ])
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)


class FrozenEncoder:
    def __init__(self, name: str, dim: int, encode_batch: Callable[[List[Image.Image]], torch.Tensor]):
        self.name = name
        self.dim = dim
        self._encode_batch = encode_batch

    @torch.no_grad()
    def encode(self, image_paths: List[str], batch_size: int = 64, verbose: bool = True) -> np.ndarray:
        out = []
        for start in range(0, len(image_paths), batch_size):
            batch = [Image.open(p) for p in image_paths[start:start + batch_size]]
            out.append(self._encode_batch(batch).float().cpu().numpy())
            if verbose and (start // batch_size) % 20 == 0:
                print(f"  [{self.name}] encoded {start + len(batch):,}/{len(image_paths):,}")
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), dtype=np.float32)


def _load_sd_vae(device: torch.device) -> FrozenEncoder:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval().to(device)
    size = 64

    def encode_batch(images):
        x = _to_tensor(images, size, np.float32(0.5), np.float32(0.5), device)  # VAE expects [-1, 1]
        # Posterior mean, not a sample: a deterministic snapshot per record.
        return vae.encode(x).latent_dist.mean.flatten(1)

    return FrozenEncoder('sd-vae', 4 * (size // 8) ** 2, encode_batch)


def _load_dinov2_small(device: torch.device) -> FrozenEncoder:
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval().to(device)

    def encode_batch(images):
        x = _to_tensor(images, 224, IMAGENET_MEAN, IMAGENET_STD, device)
        return model(pixel_values=x).pooler_output

    return FrozenEncoder('dinov2-small', model.config.hidden_size, encode_batch)


def _load_clip_b32(device: torch.device) -> FrozenEncoder:
    from transformers import CLIPVisionModelWithProjection
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)

    def encode_batch(images):
        x = _to_tensor(images, 224, CLIP_MEAN, CLIP_STD, device)
        return model(pixel_values=x).image_embeds

    return FrozenEncoder('clip-b32', model.config.projection_dim, encode_batch)


def _load_resnet18(device: torch.device) -> FrozenEncoder:
    from transformers import ResNetModel
    model = ResNetModel.from_pretrained("microsoft/resnet-18").eval().to(device)

    def encode_batch(images):
        x = _to_tensor(images, 224, IMAGENET_MEAN, IMAGENET_STD, device)
        return model(pixel_values=x).pooler_output.flatten(1)

    return FrozenEncoder('resnet18', model.config.hidden_sizes[-1], encode_batch)


ENCODER_REGISTRY: Dict[str, Callable[[torch.device], FrozenEncoder]] = {
    'sd-vae': _load_sd_vae,
    'dinov2-small': _load_dinov2_small,
    'clip-b32': _load_clip_b32,
    'resnet18': _load_resnet18,
}


def load_encoder(name: str, device: str = None) -> FrozenEncoder:
    """device: 'cuda', 'cpu', ... (default: cuda when available)."""
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Available: {list(ENCODER_REGISTRY)}")
    device = torch.device(device) if device else _default_device()
    print(f"  [{name}] loading on {device}")
    return ENCODER_REGISTRY[name](device)


def encode_cached(image_paths: List[str], encoder_name: str, cache_dir: str = "data/embeddings",
                  tag: str = "", batch_size: int = 64, device: str = None) -> np.ndarray:
    """
    Encodes image_paths with a registry encoder, caching the result on disk
    keyed by (encoder, exact path list) so re-runs with different selection
    settings don't re-encode.
    """
    key = hashlib.sha1("\n".join(image_paths).encode()).hexdigest()[:12]
    cache_path = Path(cache_dir) / f"{tag}{'_' if tag else ''}{encoder_name}_{len(image_paths)}_{key}.npy"
    if cache_path.exists():
        print(f"  [{encoder_name}] loaded cached embeddings: {cache_path}")
        return np.load(cache_path)

    encoder = load_encoder(encoder_name, device)
    print(f"  [{encoder_name}] encoding {len(image_paths):,} images (dim={encoder.dim})...")
    Z = encoder.encode(image_paths, batch_size=batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, Z)
    return Z

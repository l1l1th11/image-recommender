import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)

_preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # these are standard ImageNet means and stds
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

_model_cache = {}  # cache models to avoid reloading
_embedding_dims = {}


def _get_default_device() -> str:
    return "cpu"  # <-- if necessary, modify to use GPU with "cuda"


_weights = {
    "resnet18": ResNet18_Weights.DEFAULT,
    "resnet34": ResNet34_Weights.DEFAULT,
    "resnet50": ResNet50_Weights.DEFAULT,
    "resnet101": ResNet101_Weights.DEFAULT,
    "resnet152": ResNet152_Weights.DEFAULT,
}


def _load_model(
    model_name: str, pretrained: bool = True, device: str | None = None
) -> torch.nn.Module:
    """
    Loads a ResNet model.
    Caches models for deterministic output.
    """
    if device is None:
        device = _get_default_device()

    key = (model_name, pretrained, device)  # key for caching
    if key in _model_cache:
        return _model_cache[key]  # avoid reloading of cached models

    model_cls = models.__dict__[model_name]

    if model_cls is None:
        raise ValueError(f"Unsupported model_name: {model_name}")

    if pretrained:
        model = model_cls(weights=_weights[model_name])
    else:
        model = model_cls(weights=None)

    model = torch.nn.Sequential(*list(model.children())[:-1])  # all layers without fully-connected
    model.eval()  # inference
    model.to(device)  # either cpu or gpu (cuda)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)  # dummy input for dimension inference
        out = model(dummy)  # inference (forward pass)
        _embedding_dims[model_name] = int(out.numel())  # dimension of output embedding

    _model_cache[key] = model
    return model


def get_embedding_dim(model_name: str = "resnet18") -> int:  # set resnet18 as default
    if model_name not in _embedding_dims:  # If not cached...
        _load_model(model_name, pretrained=False)  # ... load model to get dimension
    return _embedding_dims[model_name]


def extract_embedding(  # main function to extract embedding from RGB image
    img_rgb: np.ndarray,
    *,
    model_name: str = "resnet18",
    pretrained: bool = True,
    device: str | None = None,
) -> np.ndarray:
    """
    Extracts an embedding from a single RGB image.
    Args:
        - img_rgb: Input image as a NumPy array of shape (H, W, 3), expects uint8 RGB.
        - model_name: Name of the ResNet model to use.
        - pretrained: Whether to use pretrained weights.
        - device: Device to run inference on.
    """

    if device is None:
        device = _get_default_device()

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:  # image must be RGB (3 channels)
        raise ValueError("Input must be RGB image with shape (H, W, 3)")

    img = Image.fromarray(img_rgb)  # NumPy --> PIL Image
    x = _preprocess(img).unsqueeze(0).to(device)  # preprocess and add batch dimension

    model = _load_model(model_name, pretrained=pretrained, device=device)  # get model
    with torch.no_grad():  # If inference is without gradient computation...
        y = model(x)
        emb = y.view(-1).cpu().numpy().astype(np.float32)
        return emb


def extract_embeddings_batch(
    imgs_rgb: list[np.ndarray],  # list of images
    *,  # folliwing parameters are keyword-only
    model_name: str = "resnet18",
    pretrained: bool = True,
    device: str | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Extracts embeddings for multiple RGB images in batches.
    Args:
        - imgs_rgb: List of RGB images as NumPy arrays of shape (H, W, 3), expects uint8 RGB.
        - model_name: Name of the ResNet model to use.
        - pretrained: Whether to use pretrained weights.
        - device: Device to run inference on.
        - batch_size: Number of images to process in each batch.
    """
    if device is None:
        device = _get_default_device()

    if not imgs_rgb:  # If there are no images...
        return np.empty(
            (0, get_embedding_dim(model_name)), dtype=np.float32
        )  # ... return an empty array

    model = _load_model(model_name, pretrained=pretrained, device=device)
    embeddings = []

    for i in range(0, len(imgs_rgb), batch_size):  # iterate over imgaes in batches
        batch_imgs = imgs_rgb[i : i + batch_size]
        tensors = [_preprocess(Image.fromarray(img)).unsqueeze(0) for img in batch_imgs]
        x = torch.cat(tensors, dim=0).to(device)
        with torch.no_grad():
            y = model(x)
        batch_emb = y.view(len(batch_imgs), -1).cpu().numpy()  # flatten and convert to NumPy
        embeddings.append(batch_emb)

    return np.vstack(embeddings).astype(np.float32)  # stack all batches into a single array

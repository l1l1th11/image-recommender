import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

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


def _load_model(model_name: str, pretrained: bool = True, device: str = "cpu") -> torch.nn.Module:
    """
    Loads a ResNet model.
    Caches models for deterministic output.
    """
    key = (model_name, pretrained, device)  # key for caching
    if key in _model_cache:
        return _model_cache[key]  # avoid reloading of cached models

    model_choices = {
        "resnet18": models.resnet18,  # default
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }

    model_cls = model_choices.get(model_name)  # model class --> instantiate model
    if model_cls is None:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model = model_cls(pretrained=pretrained)  # either pretrained or random init
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
    device: str = "cpu",
) -> np.ndarray:

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:  # image must be RGB (3 channels)
        raise ValueError("Input must be RGB image with shape (H, W, 3)")

    img = Image.fromarray(img_rgb)  # NumPy --> PIL Image
    x = _preprocess(img).unsqueeze(0).to(device)  # preprocess and add batch dimension

    model = _load_model(model_name, pretrained=pretrained, device=device)  # get model
    with torch.no_grad():  # If inference is without gradient computation...
        y = model(x)  # ... return embedding in shape of (1, D, 1, 1)

    return (
        y.squeeze().cpu().numpy().astype(np.float32)
    )  # remove batch dimension and convert to NumPy array

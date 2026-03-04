import json
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

from .modeling import PIAA_MIR, PIAA_ICI
from .encoding import load_encoders, encode_demographics, PERSONAL_TRAITS, BIG5


ImageLike = Union[str, Path, Image.Image]


class UnifiedIAA:
    def __init__(self, repo_id: str, device: Optional[str] = None):
        self.repo_id = repo_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._models = {}
        self._compat = None
        self._encoders = None
        self._prior = None
        self._tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @classmethod
    def from_pretrained(cls, repo_id: str, token: Optional[str] = None, cache_dir: Optional[str] = None, device: Optional[str] = None):
        obj = cls(repo_id=repo_id, device=device)
        compat_path = hf_hub_download(repo_id=repo_id, filename="configs/compatibility.json", token=token, cache_dir=cache_dir)
        obj._compat = json.loads(Path(compat_path).read_text(encoding="utf-8"))

        enc_path = hf_hub_download(repo_id=repo_id, filename="inference/demographics_encoder.json", token=token, cache_dir=cache_dir)
        obj._encoders = load_encoders(enc_path)

        prior_path = hf_hub_download(repo_id=repo_id, filename="inference/prior_mean_vector.pt", token=token, cache_dir=cache_dir)
        obj._prior = torch.load(prior_path, map_location="cpu").float()
        return obj

    def _artifact_checkpoint(self, task: str, backbone: str) -> str:
        task_key = "PIAA-MIR" if task.lower() == "mir" else "PIAA-ICI"
        for item in self._compat["artifacts"]:
            if item.get("task") == task_key and item.get("backbone") == backbone:
                return item["checkpoint"]
        raise ValueError(f"No checkpoint for task={task_key}, backbone={backbone}")

    def _load_model(self, task: str, backbone: str, token: Optional[str] = None, cache_dir: Optional[str] = None):
        k = (task.lower(), backbone)
        if k in self._models:
            return self._models[k]

        ckpt_name = self._artifact_checkpoint(task, backbone)
        ckpt_path = hf_hub_download(self.repo_id, filename=f"models/{ckpt_name}", token=token, cache_dir=cache_dir)

        if task.lower() == "mir":
            model = PIAA_MIR(9, 8, 70, dropout=0.0, backbone=backbone)
        else:
            model = PIAA_ICI(9, 8, 70, dropout=0.0, backbone=backbone)

        sd = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(sd)
        model.to(self.device).eval()
        self._models[k] = model
        return model

    def _to_image_tensor(self, image: ImageLike) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        return self._tfm(img).unsqueeze(0).to(self.device)

    def _to_pt_tensor(self, demographics: Dict[str, str], big5: Dict[str, float]) -> torch.Tensor:
        vec = encode_demographics(demographics, big5, self._encoders)
        return vec.unsqueeze(0).to(self.device)

    def predict_piaa(self, image: ImageLike, demographics: Dict[str, str], big5: Dict[str, float], task: str = "mir", backbone: str = "vit_small_patch16_224") -> float:
        model = self._load_model(task, backbone)
        x = self._to_image_tensor(image)
        pt = self._to_pt_tensor(demographics, big5)
        with torch.no_grad():
            pred = model(x, pt)
        return float(pred.squeeze().item())

    def predict_giaa_prior(self, image: ImageLike, task: str = "ici", backbone: str = "swin_tiny_patch4_window7_224") -> float:
        model = self._load_model(task, backbone)
        x = self._to_image_tensor(image)
        pt = self._prior.unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = model(x, pt)
        return float(pred.squeeze().item())

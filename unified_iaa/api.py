import json
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

from .modeling import PIAA_MIR, PIAA_ICI
from .encoding import load_encoders, encode_demographics, encode_lapis_inputs, PERSONAL_TRAITS, BIG5, LAPIS_VAIAK


ImageLike = Union[str, Path, Image.Image]


def _remap_legacy_keys(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        nk = k
        if k.startswith("nima_attr.resnet."):
            nk = "nima_attr.backbone." + k[len("nima_attr.resnet."):]
        elif k.startswith("resnet."):
            nk = "backbone." + k[len("resnet."):]
        out[nk] = v
    return out


def _normalize_task_model(task: str, model: Optional[str] = None):
    t = (task or "").strip().lower()
    m = (model or "").strip().lower() if model is not None else None

    # backward compatibility: task="mir"/"ici"
    if t in {"mir", "ici"} and not m:
        return "piaa", t

    if t not in {"piaa", "giaa"}:
        raise ValueError(f"Unsupported task={task}. Use 'PIAA'/'GIAA' (or legacy 'mir'/'ici').")

    if m is None or m == "":
        m = "ici"
    if m not in {"mir", "ici"}:
        raise ValueError(f"Unsupported model={model}. Use 'mir' or 'ici'.")

    return t, m


class UnifiedIAA:
    def __init__(self, repo_id: str, device: Optional[str] = None):
        self.repo_id = repo_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._models = {}
        self._compat = None
        self._encoders = None
        self._prior = None
        self._lapis_trait_encoders = None
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

        # Optional LAPIS categorical encoder mapping for user-friendly LAPIS input API
        obj._lapis_trait_encoders = None
        try:
            lapis_opt_path = hf_hub_download(repo_id=repo_id, filename="configs/demographics_options_lapis.json", token=token, cache_dir=cache_dir)
            lapis_obj = json.loads(Path(lapis_opt_path).read_text(encoding="utf-8"))
            obj._lapis_trait_encoders = lapis_obj.get("trait_encoders")
        except Exception:
            # local fallback (editable/dev usage)
            local_lapis_opt = Path(__file__).resolve().parents[1] / "hf_release" / "configs" / "demographics_options_lapis.json"
            if local_lapis_opt.exists():
                lapis_obj = json.loads(local_lapis_opt.read_text(encoding="utf-8"))
                obj._lapis_trait_encoders = lapis_obj.get("trait_encoders")
        return obj

    def _artifact(self, task: str, backbone: str, dataset: str = "para", model: Optional[str] = None) -> dict:
        task_family, head = _normalize_task_model(task, model)
        # Current release stores checkpoints under PIAA-* keys; GIAA reuses same backbone checkpoints.
        task_key = f"PIAA-{head.upper()}" if task_family in {"piaa", "giaa"} else f"PIAA-{head.upper()}"
        for item in self._compat["artifacts"]:
            if item.get("task") == task_key and item.get("backbone") == backbone and item.get("dataset", "para") == dataset:
                return item
        raise ValueError(f"No artifact for task={task_key}, backbone={backbone}, dataset={dataset}")

    def _load_model(self, task: str, backbone: str, dataset: str = "para", model: Optional[str] = None, token: Optional[str] = None, cache_dir: Optional[str] = None):
        task_family, head = _normalize_task_model(task, model)
        k = (task_family, head, backbone, dataset)
        if k in self._models:
            return self._models[k]

        art = self._artifact(task, backbone, dataset=dataset, model=head)
        ckpt_name = art["checkpoint"]
        num_pt = int(art.get("num_pt", 70))
        ckpt_path = hf_hub_download(self.repo_id, filename=f"models/{ckpt_name}", token=token, cache_dir=cache_dir)

        if head == "mir":
            model = PIAA_MIR(9, 8, num_pt, dropout=0.0, backbone=backbone)
        else:
            model = PIAA_ICI(9, 8, num_pt, dropout=0.0, backbone=backbone)

        sd = torch.load(ckpt_path, map_location=self.device)
        sd = _remap_legacy_keys(sd)
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

    def predict_piaa(self, image: ImageLike, demographics: Dict[str, str], big5: Dict[str, float], task: str = "PIAA", model: str = "mir", backbone: str = "vit_small_patch16_224") -> float:
        model_obj = self._load_model(task, backbone, dataset="para", model=model)
        x = self._to_image_tensor(image)
        pt = self._to_pt_tensor(demographics, big5)
        with torch.no_grad():
            pred = model_obj(x, pt)
        return float(pred.squeeze().item())

    def predict_with_traits(self, image: ImageLike, traits: Union[torch.Tensor, list], task: str = "PIAA", model: str = "mir", backbone: str = "resnet50", dataset: str = "lapis") -> float:
        model_obj = self._load_model(task, backbone, dataset=dataset, model=model)
        x = self._to_image_tensor(image)
        if not torch.is_tensor(traits):
            traits = torch.tensor(traits, dtype=torch.float32)
        if traits.ndim == 1:
            traits = traits.unsqueeze(0)
        traits = traits.float().to(self.device)
        with torch.no_grad():
            pred = model_obj(x, traits)
        return float(pred.squeeze().item())

    def predict_lapis(
        self,
        image: ImageLike,
        lapis_input: Dict[str, object],
        task: str = "PIAA",
        model: str = "mir",
        backbone: str = "resnet50",
    ) -> float:
        if not self._lapis_trait_encoders:
            raise ValueError("LAPIS trait encoders not available in repo configs/demographics_options_lapis.json")
        traits = encode_lapis_inputs(lapis_input, self._lapis_trait_encoders)
        return self.predict_with_traits(image=image, traits=traits, task=task, model=model, backbone=backbone, dataset="lapis")

    def predict_giaa_prior(self, image: ImageLike, task: str = "GIAA", model: str = "ici", backbone: str = "swin_tiny_patch4_window7_224") -> float:
        model_obj = self._load_model(task, backbone, dataset="para", model=model)
        x = self._to_image_tensor(image)
        pt = self._prior.unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = model_obj(x, pt)
        return float(pred.squeeze().item())

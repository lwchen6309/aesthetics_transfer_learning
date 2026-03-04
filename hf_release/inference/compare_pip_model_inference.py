import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

from unified_iaa import UnifiedIAA
from unified_iaa.api import _remap_legacy_keys
from unified_iaa.modeling import PIAA_MIR, PIAA_ICI
from unified_iaa.encoding import load_encoders, encode_demographics


def build_tfm():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_direct_model(head: str, backbone: str, num_pt: int, ckpt_path: str, device: str):
    model = PIAA_MIR(9, 8, num_pt, dropout=0.0, backbone=backbone) if head == "mir" else PIAA_ICI(9, 8, num_pt, dropout=0.0, backbone=backbone)
    sd = torch.load(ckpt_path, map_location=device)
    sd = _remap_legacy_keys(sd)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def main():
    repo_id = "stupidog04/Unified_IAA"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = "/home/lwchen/datasets/PARA/imgs/session1/iaa_pub10_.jpg"

    root = Path(__file__).resolve().parents[1] / "configs"
    para_demo = json.loads((root / "para_demographics_template.json").read_text(encoding="utf-8"))
    lapis_traits = json.loads((root / "lapis_traits_template.json").read_text(encoding="utf-8"))["traits"]

    m = UnifiedIAA.from_pretrained(repo_id, device=device)

    compat_path = hf_hub_download(repo_id=repo_id, filename="configs/compatibility.json")
    compat = json.loads(Path(compat_path).read_text(encoding="utf-8"))

    enc_path = hf_hub_download(repo_id=repo_id, filename="inference/demographics_encoder.json")
    enc = load_encoders(enc_path)
    prior_path = hf_hub_download(repo_id=repo_id, filename="inference/prior_mean_vector.pt")

    tfm = build_tfm()
    img = tfm(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    demo = {
        "age": para_demo["age"],
        "gender": para_demo["gender"],
        "EducationalLevel": para_demo["EducationalLevel"],
        "artExperience": para_demo["artExperience"],
        "photographyExperience": para_demo["photographyExperience"],
    }
    big5 = {k: para_demo[k] for k in ["personality-E", "personality-A", "personality-N", "personality-O", "personality-C"]}
    pt_para = encode_demographics(demo, big5, enc).unsqueeze(0).to(device)
    pt_prior = torch.load(prior_path, map_location=device).float().unsqueeze(0).to(device)
    pt_lapis = torch.tensor(lapis_traits, dtype=torch.float32, device=device).unsqueeze(0)

    def artifact(task_key: str, backbone: str, dataset: str):
        for a in compat["artifacts"]:
            if a.get("task") == task_key and a.get("backbone") == backbone and a.get("dataset") == dataset:
                return a
        raise RuntimeError(f"artifact not found: {task_key} {backbone} {dataset}")

    rows = []

    # 1) GIAA + PARA (mir/vit)
    a = artifact("PIAA-MIR", "vit_small_patch16_224", "para")
    ckpt = hf_hub_download(repo_id=repo_id, filename=f"models/{a['checkpoint']}")
    dm = make_direct_model("mir", "vit_small_patch16_224", int(a["num_pt"]), ckpt, device)
    with torch.no_grad():
        direct = float(dm(img, pt_prior).squeeze().item())
    pip = float(m.predict_giaa_prior(image_path, task="GIAA", model="mir", backbone="vit_small_patch16_224"))
    rows.append(("GIAA", "PARA", pip, direct, abs(pip - direct)))

    # 2) PIAA + PARA (mir/vit)
    with torch.no_grad():
        direct = float(dm(img, pt_para).squeeze().item())
    pip = float(m.predict_piaa(image_path, demo, big5, task="PIAA", model="mir", backbone="vit_small_patch16_224"))
    rows.append(("PIAA", "PARA", pip, direct, abs(pip - direct)))

    # 3) GIAA + LAPIS (mir/resnet50)
    a = artifact("PIAA-MIR", "resnet50", "lapis")
    ckpt = hf_hub_download(repo_id=repo_id, filename=f"models/{a['checkpoint']}")
    dm = make_direct_model("mir", "resnet50", int(a["num_pt"]), ckpt, device)
    with torch.no_grad():
        direct = float(dm(img, pt_lapis).squeeze().item())
    pip = float(m.predict_with_traits(image_path, lapis_traits, task="GIAA", model="mir", backbone="resnet50", dataset="lapis"))
    rows.append(("GIAA", "LAPIS", pip, direct, abs(pip - direct)))

    # 4) PIAA + LAPIS (mir/resnet50)
    with torch.no_grad():
        direct = float(dm(img, pt_lapis).squeeze().item())
    pip = float(m.predict_with_traits(image_path, lapis_traits, task="PIAA", model="mir", backbone="resnet50", dataset="lapis"))
    rows.append(("PIAA", "LAPIS", pip, direct, abs(pip - direct)))

    print("task,dataset,pip,direct_model,abs_diff")
    for r in rows:
        print(f"{r[0]},{r[1]},{r[2]:.12f},{r[3]:.12f},{r[4]:.6e}")


if __name__ == "__main__":
    main()

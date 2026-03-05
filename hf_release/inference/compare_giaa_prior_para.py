import sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

# Ensure repo root is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unified_iaa import UnifiedIAA
from train_piaa_mir import PIAA_MIR
from train_piaa_ici import PIAA_ICI

IMG = "/mnt/d/datasets/PARA/imgs/session1/iaa_pub10_.jpg"
BASE = "/home/lwchen/active_nngp"
MODELS = [
    ("mir", "vit_small_patch16_224", "best_model_vit_small_patch16_224_piaamir_super-yogurt-742.pth"),
    ("mir", "swin_tiny_patch4_window7_224", "best_model_swin_tiny_patch4_window7_224_piaamir_fanciful-blaze-742.pth"),
    ("ici", "swin_tiny_patch4_window7_224", "best_model_swin_tiny_patch4_window7_224_piaaici_ethereal-cherry-741.pth"),
    ("ici", "vit_small_patch16_224", "best_model_vit_small_patch16_224_piaaici_laced-bird-742.pth"),
]

def build_img(path, device):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def infer_num_pt(sd):
    if "mlp1.fc1.weight" in sd:
        return int(sd["mlp1.fc1.weight"].shape[1] // 8)
    if "node_attr_user.fc1.weight" in sd:
        return int(sd["node_attr_user.fc1.weight"].shape[1])
    raise RuntimeError("cannot infer num_pt")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = build_img(IMG, device)

    prior = torch.load(f"{BASE}/hf_release/inference/prior_mean_vector.pt", map_location=device).float().unsqueeze(0)

    pip = UnifiedIAA.from_pretrained("stupidog04/Unified_IAA", device=device)

    print("model,backbone,checkpoint,num_pt,pip,direct,abs_diff")
    for mtype, backbone, ckpt in MODELS:
        sd = torch.load(f"{BASE}/models_pth/{ckpt}", map_location=device)
        num_pt = infer_num_pt(sd)
        if mtype == "mir":
            model = PIAA_MIR(9, 8, num_pt, dropout=0.0, backbone=backbone)
        else:
            model = PIAA_ICI(9, 8, num_pt, dropout=0.0, backbone=backbone)
        model.load_state_dict(sd)
        model.to(device).eval()
        with torch.no_grad():
            direct = float(model(x, prior[:, :num_pt]).squeeze().item())
        pv = float(pip.predict_giaa_prior(IMG, task="GIAA", model=mtype, backbone=backbone))
        print(f"{mtype},{backbone},{ckpt},{num_pt},{pv:.12f},{direct:.12f},{abs(pv-direct):.6e}")

if __name__ == "__main__":
    main()

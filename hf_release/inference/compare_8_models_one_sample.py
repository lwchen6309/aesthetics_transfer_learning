import json
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
from unified_iaa.encoding import load_encoders, encode_demographics, encode_lapis_inputs
from train_piaa_mir import PIAA_MIR
from train_piaa_ici import PIAA_ICI

REPO = "stupidog04/Unified_IAA"
IMG = "/mnt/d/datasets/PARA/imgs/session1/iaa_pub10_.jpg"
BASE = Path('/home/lwchen/active_nngp')

PARA_MODELS = [
    ("mir", "vit_small_patch16_224", "best_model_vit_small_patch16_224_piaamir_super-yogurt-742.pth"),
    ("mir", "swin_tiny_patch4_window7_224", "best_model_swin_tiny_patch4_window7_224_piaamir_fanciful-blaze-742.pth"),
    ("ici", "swin_tiny_patch4_window7_224", "best_model_swin_tiny_patch4_window7_224_piaaici_ethereal-cherry-741.pth"),
    ("ici", "vit_small_patch16_224", "best_model_vit_small_patch16_224_piaaici_laced-bird-742.pth"),
]
LAPIS_MODELS = [
    ("mir", "vit_small_patch16_224", "best_model_vit_small_patch16_224_piaamir_woven-wind-1160.pth"),
    ("mir", "swin_tiny_patch4_window7_224", "best_model_swin_tiny_patch4_window7_224_piaamir_electric-wind-1161.pth"),
    ("ici", "swin_tiny_patch4_window7_224", "best_model_swin_tiny_patch4_window7_224_piaaici_crimson-armadillo-1151.pth"),
    ("ici", "vit_small_patch16_224", "best_model_vit_small_patch16_224_piaaici_misunderstood-pond-1151.pth"),
]

def build_img_tensor(path, device):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return tfm(Image.open(path).convert('RGB')).unsqueeze(0).to(device)

def infer_num_pt(state_dict):
    if "mlp1.fc1.weight" in state_dict:
        return int(state_dict["mlp1.fc1.weight"].shape[1] // 8)
    if "node_attr_user.fc1.weight" in state_dict:
        return int(state_dict["node_attr_user.fc1.weight"].shape[1])
    if "interaction_mlp.fc1.weight" in state_dict:
        return int(state_dict["interaction_mlp.fc1.weight"].shape[1] // 8)
    raise RuntimeError("Cannot infer num_pt from checkpoint keys")

def load_direct(model_type, backbone, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    num_pt = infer_num_pt(sd)
    if model_type == 'mir':
        m = PIAA_MIR(9, 8, num_pt, dropout=0.0, backbone=backbone)
    else:
        m = PIAA_ICI(9, 8, num_pt, dropout=0.0, backbone=backbone)
    m.load_state_dict(sd)
    m.to(device).eval()
    return m, num_pt


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = build_img_tensor(IMG, device)

    para_tpl = json.loads((BASE / 'hf_release/configs/para_demographics_template.json').read_text())
    lapis_tpl = json.loads((BASE / 'hf_release/configs/lapis_traits_template.json').read_text())
    enc = load_encoders(str(BASE / 'hf_release/inference/demographics_encoder.json'))

    demo = {
        'age': para_tpl['age'][0],
        'gender': para_tpl['gender'][0],
        'EducationalLevel': para_tpl['EducationalLevel'][0],
        'artExperience': para_tpl['artExperience'][0],
        'photographyExperience': para_tpl['photographyExperience'][0],
    }
    big5 = {
        'personality-E': para_tpl['personality-E'][0],
        'personality-A': para_tpl['personality-A'][0],
        'personality-N': para_tpl['personality-N'][0],
        'personality-O': para_tpl['personality-O'][0],
        'personality-C': para_tpl['personality-C'][0],
    }
    pt_para = encode_demographics(demo, big5, enc).unsqueeze(0).to(device)

    lapis_input = {
        'nationality': lapis_tpl['nationality'][0],
        'demo_gender': lapis_tpl['demo_gender'][0],
        'demo_edu': lapis_tpl['demo_edu'][0],
        'demo_colorblind': lapis_tpl['demo_colorblind'][0],
        'age': lapis_tpl['age'][0],
    }
    for k in [f'VAIAK{i}' for i in range(1,8)] + [f'2VAIAK{i}' for i in range(1,5)]:
        lapis_input[k] = 3
    lapis_enc = {k: {v:i for i,v in enumerate(lapis_tpl[k])} for k in ['nationality','demo_gender','demo_edu','demo_colorblind','age']}
    pt_lapis = encode_lapis_inputs(lapis_input, lapis_enc).unsqueeze(0).to(device)

    pipe = UnifiedIAA.from_pretrained(REPO, device=device)

    print('dataset,model,backbone,checkpoint,num_pt,pip,direct,abs_diff')

    for model, backbone, ckpt in PARA_MODELS:
        dm, num_pt = load_direct(model, backbone, str(BASE / 'models_pth' / ckpt), device)
        with torch.no_grad():
            direct = float(dm(x, pt_para[:, :num_pt]).squeeze().item())
        pip = float(pipe.predict_piaa(IMG, demo, big5, task='PIAA', model=model, backbone=backbone))
        print(f"PARA,{model},{backbone},{ckpt},{num_pt},{pip:.12f},{direct:.12f},{abs(pip-direct):.6e}")

    for model, backbone, ckpt in LAPIS_MODELS:
        dm, num_pt = load_direct(model, backbone, str(BASE / 'models_pth' / ckpt), device)
        with torch.no_grad():
            direct = float(dm(x, pt_lapis[:, :num_pt]).squeeze().item())
        pip = float(pipe.predict_lapis(IMG, lapis_input, task='PIAA', model=model, backbone=backbone))
        print(f"LAPIS,{model},{backbone},{ckpt},{num_pt},{pip:.12f},{direct:.12f},{abs(pip-direct):.6e}")

if __name__ == '__main__':
    main()

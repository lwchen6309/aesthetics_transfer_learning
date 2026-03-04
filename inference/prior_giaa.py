import argparse
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from train_piaa_mir import PIAA_MIR
from train_piaa_ici import PIAA_ICI
from inference.demographics_encoder import build_encoders, encode_demographics, PERSONAL_TRAITS, BIG5


def build_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(task, backbone, checkpoint, device):
    model = PIAA_MIR(9, 8, 70, dropout=0.0, backbone=backbone) if task == "mir" else PIAA_ICI(9, 8, 70, dropout=0.0, backbone=backbone)
    sd = torch.load(checkpoint, map_location=device)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def compute_prior_pt(para_root: str) -> torch.Tensor:
    images = pd.read_csv(f"{para_root}/annotation/PARA-Images.csv")
    users = pd.read_csv(f"{para_root}/annotation/PARA-UserInfo.csv")
    df = images.merge(users, on="userId", how="inner")

    enc = build_encoders(f"{para_root}/annotation/PARA-UserInfo.csv")

    # annotation-weighted mean prior (same spirit as evaluate_with_prior)
    vectors = []
    for _, r in df.iterrows():
        demo = {k: str(r[k]) for k in PERSONAL_TRAITS}
        big5 = {k: float(r[k]) for k in BIG5}
        vectors.append(encode_demographics(demo, big5, enc))
    pt = torch.stack(vectors, dim=0).mean(dim=0)
    return pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mir", "ici"], required=True)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--para_root", default="/mnt/d/datasets/PARA")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.task, args.backbone, args.checkpoint, device)

    tfm = build_transform()
    img = tfm(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    pt = compute_prior_pt(args.para_root).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img, pt)

    print(float(pred.squeeze().item()))


if __name__ == "__main__":
    main()

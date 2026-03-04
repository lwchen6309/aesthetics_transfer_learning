import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train_piaa_mir import PIAA_MIR
from train_piaa_ici import PIAA_ICI
from inference.demographics_encoder import load_encoders, encode_demographics, PERSONAL_TRAITS, BIG5


def build_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(task, backbone, checkpoint, device):
    if task == "mir":
        model = PIAA_MIR(9, 8, 70, dropout=0.0, backbone=backbone)
    else:
        model = PIAA_ICI(9, 8, 70, dropout=0.0, backbone=backbone)
    sd = torch.load(checkpoint, map_location=device)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mir", "ici"], required=True)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--encoder_json", default="inference/demographics_encoder.json")
    ap.add_argument("--demographics_json", required=True, help="JSON with age/gender/EducationalLevel/artExperience/photographyExperience + big5")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.task, args.backbone, args.checkpoint, device)

    enc = load_encoders(args.encoder_json)
    user = json.loads(Path(args.demographics_json).read_text(encoding="utf-8"))

    demo = {k: user[k] for k in PERSONAL_TRAITS}
    big5 = {k: user[k] for k in BIG5}
    pt = encode_demographics(demo, big5, enc).unsqueeze(0).to(device)

    tfm = build_transform()
    img = tfm(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img, pt)

    print(float(pred.squeeze().item()))


if __name__ == "__main__":
    main()

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F

PERSONAL_TRAITS = ["age", "gender", "EducationalLevel", "artExperience", "photographyExperience"]
BIG5 = ["personality-E", "personality-A", "personality-N", "personality-O", "personality-C"]


def build_encoders(userinfo_csv: str) -> Dict[str, Dict[str, int]]:
    df = pd.read_csv(userinfo_csv)
    enc = {}
    for col in PERSONAL_TRAITS:
        uniq = df[col].dropna().astype(str).unique().tolist()
        enc[col] = {v: i for i, v in enumerate(uniq)}
    return enc


def save_encoders(userinfo_csv: str, out_json: str) -> None:
    enc = build_encoders(userinfo_csv)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(enc, f, ensure_ascii=False, indent=2)


def load_encoders(path: str) -> Dict[str, Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_demographics(demo: Dict[str, str], big5: Dict[str, float], encoders: Dict[str, Dict[str, int]]) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for k in PERSONAL_TRAITS:
        v = str(demo[k])
        if v not in encoders[k]:
            raise ValueError(f"Unknown category for {k}: {v}")
        idx = encoders[k][v]
        onehot = F.one_hot(torch.tensor(idx), num_classes=len(encoders[k])).float()
        parts.append(onehot)

    b = torch.tensor([float(big5[k]) for k in BIG5], dtype=torch.float32)
    parts.append(b)
    return torch.cat(parts, dim=0)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--userinfo_csv", required=True)
    ap.add_argument("--out_json", default="inference/demographics_encoder.json")
    args = ap.parse_args()

    save_encoders(args.userinfo_csv, args.out_json)
    print(f"saved: {args.out_json}")

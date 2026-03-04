import json
from typing import Dict, List

import torch
import torch.nn.functional as F

PERSONAL_TRAITS = ["age", "gender", "EducationalLevel", "artExperience", "photographyExperience"]
BIG5 = ["personality-E", "personality-A", "personality-N", "personality-O", "personality-C"]
BIG5_BINS = 10


def load_encoders(path: str) -> Dict[str, Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _big5_to_onehot(v: float) -> torch.Tensor:
    iv = int(round(float(v)))
    if iv < 1 or iv > BIG5_BINS:
        raise ValueError(f"Big5 value out of range [1,10]: {v}")
    return F.one_hot(torch.tensor(iv - 1), num_classes=BIG5_BINS).float()


def encode_demographics(demo: Dict[str, str], big5: Dict[str, float], encoders: Dict[str, Dict[str, int]]) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for k in PERSONAL_TRAITS:
        v = str(demo[k])
        if v not in encoders[k]:
            raise ValueError(f"Unknown category for {k}: {v}")
        idx = encoders[k][v]
        parts.append(F.one_hot(torch.tensor(idx), num_classes=len(encoders[k])).float())

    parts.append(torch.cat([_big5_to_onehot(big5[k]) for k in BIG5], dim=0))
    return torch.cat(parts, dim=0)

import json
from typing import Dict, List

import torch
import torch.nn.functional as F

PERSONAL_TRAITS = ["age", "gender", "EducationalLevel", "artExperience", "photographyExperience"]
BIG5 = ["personality-E", "personality-A", "personality-N", "personality-O", "personality-C"]
BIG5_BINS = 10

# LAPIS onehot(137) layout used by LAPIS PIAA training:
# [nationality_onehot, demo_gender_onehot, demo_edu_onehot, demo_colorblind_onehot, age_onehot,
#  VAIAK1..7 onehot(7 each), 2VAIAK1..4 onehot(7 each)]
LAPIS_CATEGORICAL = ["nationality", "demo_gender", "demo_edu", "demo_colorblind", "age"]
LAPIS_VAIAK = [f"VAIAK{i}" for i in range(1, 8)] + [f"2VAIAK{i}" for i in range(1, 5)]


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


def encode_lapis_inputs(
    lapis_input: Dict[str, object],
    lapis_trait_encoders: Dict[str, Dict[str, int]],
) -> torch.Tensor:
    parts: List[torch.Tensor] = []

    for k in LAPIS_CATEGORICAL:
        if k not in lapis_input:
            raise ValueError(f"Missing LAPIS field: {k}")
        v = str(lapis_input[k])
        if k not in lapis_trait_encoders or v not in lapis_trait_encoders[k]:
            raise ValueError(f"Unknown category for {k}: {v}")
        idx = lapis_trait_encoders[k][v]
        parts.append(F.one_hot(torch.tensor(idx), num_classes=len(lapis_trait_encoders[k])).float())

    for k in LAPIS_VAIAK:
        if k not in lapis_input:
            raise ValueError(f"Missing VAIAK field: {k}")
        iv = int(lapis_input[k])
        if iv < 0 or iv > 6:
            raise ValueError(f"{k} out of range [0,6]: {iv}")
        parts.append(F.one_hot(torch.tensor(iv), num_classes=7).float())

    return torch.cat(parts, dim=0)

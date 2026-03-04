---
library_name: pytorch
tags:
  - image-aesthetics
  - personalization
  - vit
  - swin
  - pt70
---

# Unified IAA (PARA pt=70 + LAPIS pt=137)

This release includes:

- **PARA PIAA-MIR (pt=70)**
  - `best_model_vit_small_patch16_224_piaamir_super-yogurt-742.pth`
  - `best_model_swin_tiny_patch4_window7_224_piaamir_fanciful-blaze-742.pth`
- **PARA PIAA-ICI (pt=70)**
  - `best_model_swin_tiny_patch4_window7_224_piaaici_ethereal-cherry-741.pth`
  - `best_model_vit_small_patch16_224_piaaici_laced-bird-742.pth`
- **LAPIS PIAA-MIR (pt=137)**
  - `lapis_best_model_resnet50_piaamir_azure-gorge-1153.pth`
- **LAPIS PIAA-ICI (pt=137)**
  - `lapis_best_model_resnet50_piaaici_dutiful-serenity-1076.pth`
- **PARA GIAA prior vector** (`prior_mean_vector.pt`, shape `[70]`)

## Scope and compatibility

- ✅ PARA supported: `num_pt=70`, `disable_onehot=false`
- ✅ LAPIS supported: `num_pt=137` (use direct trait vector input)
- ✅ Supports both:
  - **PARA GIAA prior inference** (using uploaded `prior_mean_vector.pt`)
  - **PARA PIAA personalized inference** (user provides demographics + Big5)
  - **LAPIS inference** (user provides a trait vector)

See `configs/compatibility.json` for exact artifact mapping and hashes.

## Files

- `models/best_model_vit_small_patch16_224_piaamir_super-yogurt-742.pth`
- `models/best_model_swin_tiny_patch4_window7_224_piaamir_fanciful-blaze-742.pth`
- `models/best_model_swin_tiny_patch4_window7_224_piaaici_ethereal-cherry-741.pth`
- `models/best_model_vit_small_patch16_224_piaaici_laced-bird-742.pth`
- `models/lapis_best_model_resnet50_piaamir_azure-gorge-1153.pth`
- `models/lapis_best_model_resnet50_piaaici_dutiful-serenity-1076.pth`
- `inference/prior_mean_vector.pt`
- `inference/demographics_encoder.py`
- `inference/predict_piaa.py`
- `inference/prior_giaa.py`
- `infer_unified_iaa.sh`
- `run_LAPIS_PIAA.sh`

## Python SDK (pip style)

```python
from unified_iaa import UnifiedIAA

m = UnifiedIAA.from_pretrained("stupidog04/Unified_IAA")

score_piaa = m.predict_piaa(
    image="/path/to/test.jpg",
    demographics={
        "age": "20-29",
        "gender": "female",
        "EducationalLevel": "Bachelor",
        "artExperience": "medium",
        "photographyExperience": "low",
    },
    big5={
        "personality-E": 6,
        "personality-A": 7,
        "personality-N": 4,
        "personality-O": 8,
        "personality-C": 6,
    },
    task="mir",
    backbone="vit_small_patch16_224",
)

score_giaa = m.predict_giaa_prior(
    image="/path/to/test.jpg",
    task="ici",
    backbone="swin_tiny_patch4_window7_224",
)

# LAPIS (pt=137) with direct trait vector
lapis_traits = [0.0] * 137
score_lapis = m.predict_with_traits(
    image="/path/to/test.jpg",
    traits=lapis_traits,
    task="mir",
    backbone="resnet50",
    dataset="lapis",
)
```

## Quick usage

### 1) Build demographics encoder
```bash
python inference/demographics_encoder.py \
  --userinfo_csv /mnt/d/datasets/PARA/annotation/PARA-UserInfo.csv \
  --out_json inference/demographics_encoder.json
```

### 2) Personalized PIAA inference (user demographics input)
```bash
python inference/predict_piaa.py \
  --task mir \
  --backbone vit_small_patch16_224 \
  --checkpoint models/best_model_vit_small_patch16_224_piaamir_super-yogurt-742.pth \
  --image /path/to/image.jpg \
  --encoder_json inference/demographics_encoder.json \
  --demographics_json /path/to/user_demo.json
```

### 3) GIAA prior inference (uploaded prior vector)
```bash
python inference/prior_giaa.py \
  --task ici \
  --backbone swin_tiny_patch4_window7_224 \
  --checkpoint models/best_model_swin_tiny_patch4_window7_224_piaaici_ethereal-cherry-741.pth \
  --image /path/to/image.jpg \
  --prior_vector_path inference/prior_mean_vector.pt
```

### 4) Run LAPIS PIAA eval pipeline
```bash
bash run_LAPIS_PIAA.sh
```

## user_demo.json format

```json
{
  "age": "20-29",
  "gender": "female",
  "EducationalLevel": "Bachelor",
  "artExperience": "medium",
  "photographyExperience": "low",
  "personality-E": 6,
  "personality-A": 7,
  "personality-N": 4,
  "personality-O": 8,
  "personality-C": 6
}
```

> Category strings must match encoder categories produced from your PARA `PARA-UserInfo.csv`.

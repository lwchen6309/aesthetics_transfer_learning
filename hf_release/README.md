---
library_name: pytorch
tags:
  - image-aesthetics
  - personalization
  - vit
  - swin
  - pt70
---

# Unified IAA (pt=70, ViT/Swin)

This release includes **pt=70 onehot** compatible checkpoints for:

- **PIAA-MIR**
  - `best_model_vit_small_patch16_224_piaamir_super-yogurt-742.pth`
  - `best_model_swin_tiny_patch4_window7_224_piaamir_fanciful-blaze-742.pth`
- **PIAA-ICI**
  - `best_model_swin_tiny_patch4_window7_224_piaaici_ethereal-cherry-741.pth`
  - `best_model_vit_small_patch16_224_piaaici_laced-bird-742.pth`
- **GIAA prior vector** (`prior_mean_vector.pt`, shape `[70]`)

## Scope and compatibility

- ✅ Supported: `num_pt=70`, `disable_onehot=false`
- ✅ Supports both:
  - **GIAA prior inference** (using uploaded `prior_mean_vector.pt`)
  - **PIAA personalized inference** (user provides demographics + Big5)
- ℹ️ ResNet checkpoints are excluded from this package by choice.

See `configs/compatibility.json` for exact artifact mapping and hashes.

## Files

- `models/best_model_vit_small_patch16_224_piaamir_super-yogurt-742.pth`
- `models/best_model_swin_tiny_patch4_window7_224_piaamir_fanciful-blaze-742.pth`
- `models/best_model_swin_tiny_patch4_window7_224_piaaici_ethereal-cherry-741.pth`
- `models/best_model_vit_small_patch16_224_piaaici_laced-bird-742.pth`
- `inference/prior_mean_vector.pt`
- `inference/demographics_encoder.py`
- `inference/predict_piaa.py`
- `inference/prior_giaa.py`

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

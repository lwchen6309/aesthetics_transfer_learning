---
library_name: pytorch
tags:
  - image-aesthetics
  - personalization
  - resnet50
  - pt70
---

# PARA PIAA/GIAA (pt=70, resnet50)

This release includes **pt=70 onehot** compatible checkpoints for:

- **PIAA-MIR** (`best_model_resnet50_piaamir_desert-dawn-621.pth`)
- **PIAA-ICI** (`best_model_resnet50_piaaici_crimson-sound-642.pth`)
- **GIAA prior vector** (`prior_mean_vector.pt`, shape `[70]`)

## Scope and compatibility

- ✅ Supported: `num_pt=70`, `disable_onehot=false`
- ✅ Supports both:
  - **GIAA prior inference** (using uploaded `prior_mean_vector.pt`)
  - **PIAA personalized inference** (user provides demographics + Big5)
- ❌ Not in this release: `num_pt=25` / `--disable_onehot` variants

See `configs/compatibility.json` for exact artifact mapping and hashes.

## Files

- `models/best_model_resnet50_piaamir_desert-dawn-621.pth`
- `models/best_model_resnet50_piaaici_crimson-sound-642.pth`
- `inference/prior_mean_vector.pt`
- `inference/demographics_encoder.py`
- `inference/predict_piaa.py`
- `inference/prior_giaa.py`

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
  --backbone resnet50 \
  --checkpoint models/best_model_resnet50_piaamir_desert-dawn-621.pth \
  --image /path/to/image.jpg \
  --encoder_json inference/demographics_encoder.json \
  --demographics_json /path/to/user_demo.json
```

### 3) GIAA prior inference (uploaded prior vector)
```bash
python inference/prior_giaa.py \
  --task ici \
  --backbone resnet50 \
  --checkpoint models/best_model_resnet50_piaaici_crimson-sound-642.pth \
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

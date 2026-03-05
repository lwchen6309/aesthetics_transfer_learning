from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from unified_iaa import UnifiedIAA
from unified_iaa.encoding import encode_demographics, encode_lapis_inputs
from PARA_PIAA_dataloader import load_data as para_load_data
from LAPIS_PIAA_dataloader import load_data as lapis_load_data, collate_fn as lapis_collate_fn
from utils.argflags import parse_arguments_piaa


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / 'models_pth'


def _device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _args():
    parser = parse_arguments_piaa(False)
    args = parser.parse_args([])
    args.trainset = 'PIAA'
    args.is_eval = True
    args.is_log = False
    args.num_workers = 0
    args.batch_size = 1
    return args


def _assert_close(a: float, b: float, tol: float = 1e-5):
    assert abs(float(a) - float(b)) <= tol, f'mismatch: {a} vs {b}'


def _max_samples(default: int = 5) -> int:
    import os
    return int(os.getenv('PIP_UT_MAX_SAMPLES', default))


@pytest.mark.skipif(not MODEL_DIR.exists(), reason='models_pth not found')
def test_pip_para_on_trainer_dataset_matches_direct_forward():
    device = _device()
    m = UnifiedIAA.from_pretrained('stupidog04/Unified_IAA', device=device)

    args = _args()
    _, _, test_dataset = para_load_data(args)
    _ = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    row = test_dataset.data.iloc[0]
    image_path = Path(test_dataset.root_dir) / 'imgs' / str(row['sessionId']) / str(row['imageName'])

    demographics = {
        'age': str(row['age']),
        'gender': str(row['gender']),
        'EducationalLevel': str(row['EducationalLevel']),
        'artExperience': str(row['artExperience']),
        'photographyExperience': str(row['photographyExperience']),
    }
    big5 = {
        'personality-E': float(row['personality-E']),
        'personality-A': float(row['personality-A']),
        'personality-N': float(row['personality-N']),
        'personality-O': float(row['personality-O']),
        'personality-C': float(row['personality-C']),
    }

    pip_score = m.predict_piaa(
        image=image_path,
        demographics=demographics,
        big5=big5,
        task='PIAA',
        model='mir',
        backbone='vit_small_patch16_224',
    )

    model = m._load_model(task='PIAA', model='mir', backbone='vit_small_patch16_224', dataset='para')
    x = m._to_image_tensor(image_path)
    pt = encode_demographics(demographics, big5, m._encoders).unsqueeze(0).to(device)
    with torch.no_grad():
        direct_score = float(model(x, pt).squeeze().item())

    _assert_close(pip_score, direct_score)


@pytest.mark.skipif(not MODEL_DIR.exists(), reason='models_pth not found')
def test_pip_lapis_on_trainer_dataset_matches_direct_forward():
    device = _device()
    m = UnifiedIAA.from_pretrained('stupidog04/Unified_IAA', device=device)

    args = _args()
    _, _, test_dataset = lapis_load_data(args)
    _ = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lapis_collate_fn)

    row = test_dataset.data.iloc[0]
    image_path = Path(test_dataset.image_dir) / str(row['imageName'])

    lapis_input = {
        'nationality': str(row['nationality']),
        'demo_gender': str(row['demo_gender']),
        'demo_edu': str(row['demo_edu']),
        'demo_colorblind': str(row['demo_colorblind']),
        'age': str(row['age']),
        'VAIAK1': int(row['VAIAK1']), 'VAIAK2': int(row['VAIAK2']), 'VAIAK3': int(row['VAIAK3']), 'VAIAK4': int(row['VAIAK4']),
        'VAIAK5': int(row['VAIAK5']), 'VAIAK6': int(row['VAIAK6']), 'VAIAK7': int(row['VAIAK7']),
        '2VAIAK1': int(row['2VAIAK1']), '2VAIAK2': int(row['2VAIAK2']), '2VAIAK3': int(row['2VAIAK3']), '2VAIAK4': int(row['2VAIAK4']),
    }

    pip_score = m.predict_lapis(
        image=image_path,
        lapis_input=lapis_input,
        task='PIAA',
        model='mir',
        backbone='resnet50',
    )

    model = m._load_model(task='PIAA', model='mir', backbone='resnet50', dataset='lapis')
    x = m._to_image_tensor(image_path)
    pt = encode_lapis_inputs(lapis_input, m._lapis_trait_encoders).unsqueeze(0).to(device)
    with torch.no_grad():
        direct_score = float(model(x, pt).squeeze().item())

    _assert_close(pip_score, direct_score)


@pytest.mark.skipif(not MODEL_DIR.exists(), reason='models_pth not found')
def test_pip_para_multisample_consistency_from_trainer_dataset():
    device = _device()
    m = UnifiedIAA.from_pretrained('stupidog04/Unified_IAA', device=device)
    model = m._load_model(task='PIAA', model='mir', backbone='vit_small_patch16_224', dataset='para')

    args = _args()
    _, _, test_dataset = para_load_data(args)

    n = min(_max_samples(5), len(test_dataset.data))
    checked = 0
    for i in range(n):
        row = test_dataset.data.iloc[i]
        image_path = Path(test_dataset.root_dir) / 'imgs' / str(row['sessionId']) / str(row['imageName'])
        demographics = {
            'age': str(row['age']),
            'gender': str(row['gender']),
            'EducationalLevel': str(row['EducationalLevel']),
            'artExperience': str(row['artExperience']),
            'photographyExperience': str(row['photographyExperience']),
        }
        big5 = {
            'personality-E': float(row['personality-E']),
            'personality-A': float(row['personality-A']),
            'personality-N': float(row['personality-N']),
            'personality-O': float(row['personality-O']),
            'personality-C': float(row['personality-C']),
        }
        pip_score = m.predict_piaa(image=image_path, demographics=demographics, big5=big5, task='PIAA', model='mir', backbone='vit_small_patch16_224')
        x = m._to_image_tensor(image_path)
        pt = encode_demographics(demographics, big5, m._encoders).unsqueeze(0).to(device)
        with torch.no_grad():
            direct_score = float(model(x, pt).squeeze().item())
        _assert_close(pip_score, direct_score)
        checked += 1

    assert checked > 0


@pytest.mark.skipif(not MODEL_DIR.exists(), reason='models_pth not found')
def test_pip_lapis_multisample_consistency_from_trainer_dataset():
    device = _device()
    m = UnifiedIAA.from_pretrained('stupidog04/Unified_IAA', device=device)
    model = m._load_model(task='PIAA', model='mir', backbone='resnet50', dataset='lapis')

    args = _args()
    _, _, test_dataset = lapis_load_data(args)

    n = min(_max_samples(5), len(test_dataset.data))
    checked = 0
    for i in range(n):
        row = test_dataset.data.iloc[i]
        image_path = Path(test_dataset.image_dir) / str(row['imageName'])
        lapis_input = {
            'nationality': str(row['nationality']),
            'demo_gender': str(row['demo_gender']),
            'demo_edu': str(row['demo_edu']),
            'demo_colorblind': str(row['demo_colorblind']),
            'age': str(row['age']),
            'VAIAK1': int(row['VAIAK1']), 'VAIAK2': int(row['VAIAK2']), 'VAIAK3': int(row['VAIAK3']), 'VAIAK4': int(row['VAIAK4']),
            'VAIAK5': int(row['VAIAK5']), 'VAIAK6': int(row['VAIAK6']), 'VAIAK7': int(row['VAIAK7']),
            '2VAIAK1': int(row['2VAIAK1']), '2VAIAK2': int(row['2VAIAK2']), '2VAIAK3': int(row['2VAIAK3']), '2VAIAK4': int(row['2VAIAK4']),
        }
        pip_score = m.predict_lapis(image=image_path, lapis_input=lapis_input, task='PIAA', model='mir', backbone='resnet50')
        x = m._to_image_tensor(image_path)
        pt = encode_lapis_inputs(lapis_input, m._lapis_trait_encoders).unsqueeze(0).to(device)
        with torch.no_grad():
            direct_score = float(model(x, pt).squeeze().item())
        _assert_close(pip_score, direct_score)
        checked += 1

    assert checked > 0


@pytest.mark.skipif(not MODEL_DIR.exists(), reason='models_pth not found')
def test_pip_para_ici_multisample_consistency_from_trainer_dataset():
    device = _device()
    m = UnifiedIAA.from_pretrained('stupidog04/Unified_IAA', device=device)
    model = m._load_model(task='PIAA', model='ici', backbone='vit_small_patch16_224', dataset='para')

    args = _args()
    _, _, test_dataset = para_load_data(args)

    n = min(_max_samples(5), len(test_dataset.data))
    checked = 0
    for i in range(n):
        row = test_dataset.data.iloc[i]
        image_path = Path(test_dataset.root_dir) / 'imgs' / str(row['sessionId']) / str(row['imageName'])
        demographics = {
            'age': str(row['age']),
            'gender': str(row['gender']),
            'EducationalLevel': str(row['EducationalLevel']),
            'artExperience': str(row['artExperience']),
            'photographyExperience': str(row['photographyExperience']),
        }
        big5 = {
            'personality-E': float(row['personality-E']),
            'personality-A': float(row['personality-A']),
            'personality-N': float(row['personality-N']),
            'personality-O': float(row['personality-O']),
            'personality-C': float(row['personality-C']),
        }
        pip_score = m.predict_piaa(image=image_path, demographics=demographics, big5=big5, task='PIAA', model='ici', backbone='vit_small_patch16_224')
        x = m._to_image_tensor(image_path)
        pt = encode_demographics(demographics, big5, m._encoders).unsqueeze(0).to(device)
        with torch.no_grad():
            direct_score = float(model(x, pt).squeeze().item())
        _assert_close(pip_score, direct_score)
        checked += 1

    assert checked > 0


@pytest.mark.skipif(not MODEL_DIR.exists(), reason='models_pth not found')
def test_pip_lapis_ici_multisample_consistency_from_trainer_dataset():
    device = _device()
    m = UnifiedIAA.from_pretrained('stupidog04/Unified_IAA', device=device)
    model = m._load_model(task='PIAA', model='ici', backbone='resnet50', dataset='lapis')

    args = _args()
    _, _, test_dataset = lapis_load_data(args)

    n = min(_max_samples(5), len(test_dataset.data))
    checked = 0
    for i in range(n):
        row = test_dataset.data.iloc[i]
        image_path = Path(test_dataset.image_dir) / str(row['imageName'])
        lapis_input = {
            'nationality': str(row['nationality']),
            'demo_gender': str(row['demo_gender']),
            'demo_edu': str(row['demo_edu']),
            'demo_colorblind': str(row['demo_colorblind']),
            'age': str(row['age']),
            'VAIAK1': int(row['VAIAK1']), 'VAIAK2': int(row['VAIAK2']), 'VAIAK3': int(row['VAIAK3']), 'VAIAK4': int(row['VAIAK4']),
            'VAIAK5': int(row['VAIAK5']), 'VAIAK6': int(row['VAIAK6']), 'VAIAK7': int(row['VAIAK7']),
            '2VAIAK1': int(row['2VAIAK1']), '2VAIAK2': int(row['2VAIAK2']), '2VAIAK3': int(row['2VAIAK3']), '2VAIAK4': int(row['2VAIAK4']),
        }
        pip_score = m.predict_lapis(image=image_path, lapis_input=lapis_input, task='PIAA', model='ici', backbone='resnet50')
        x = m._to_image_tensor(image_path)
        pt = encode_lapis_inputs(lapis_input, m._lapis_trait_encoders).unsqueeze(0).to(device)
        with torch.no_grad():
            direct_score = float(model(x, pt).squeeze().item())
        _assert_close(pip_score, direct_score)
        checked += 1

    assert checked > 0

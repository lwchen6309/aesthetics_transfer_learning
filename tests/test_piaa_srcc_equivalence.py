from pathlib import Path

import pytest
import torch

from unified_iaa import UnifiedIAA
from unified_iaa.encoding import encode_demographics, encode_lapis_inputs


SAMPLE_IMAGE = Path('/mnt/d/datasets/PARA/imgs/session1/iaa_pub10_.jpg')


@pytest.fixture(scope='module')
def iaa_model() -> UnifiedIAA:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return UnifiedIAA.from_pretrained('stupidog04/Unified_IAA', device=device)


def _assert_close(a: float, b: float, tol: float = 1e-6):
    assert abs(float(a) - float(b)) <= tol, f'mismatch: {a} vs {b}'


@pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason='sample image not found')
def test_pip_para_predict_matches_direct_forward(iaa_model: UnifiedIAA):
    m = iaa_model

    demographics = {
        'age': '30-34',
        'gender': 'female',
        'EducationalLevel': 'junior_college',
        'artExperience': 'proficient',
        'photographyExperience': 'proficient',
    }
    big5 = {
        'personality-E': 6,
        'personality-A': 7,
        'personality-N': 4,
        'personality-O': 8,
        'personality-C': 6,
    }

    pip_score = m.predict_piaa(
        image=SAMPLE_IMAGE,
        demographics=demographics,
        big5=big5,
        task='PIAA',
        model='mir',
        backbone='vit_small_patch16_224',
    )

    model = m._load_model(task='PIAA', model='mir', backbone='vit_small_patch16_224', dataset='para')
    x = m._to_image_tensor(SAMPLE_IMAGE)
    pt = encode_demographics(demographics, big5, m._encoders).unsqueeze(0).to(m.device)
    with torch.no_grad():
        direct_score = float(model(x, pt).squeeze().item())

    _assert_close(pip_score, direct_score)


@pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason='sample image not found')
def test_pip_lapis_predict_matches_direct_forward(iaa_model: UnifiedIAA):
    m = iaa_model

    lapis_input = {
        'nationality': 'british',
        'demo_gender': 'female',
        'demo_edu': "Bachelor's or equivalent",
        'demo_colorblind': 'No',
        'age': '28-38',
        'VAIAK1': 3, 'VAIAK2': 3, 'VAIAK3': 3, 'VAIAK4': 3,
        'VAIAK5': 3, 'VAIAK6': 3, 'VAIAK7': 3,
        '2VAIAK1': 3, '2VAIAK2': 3, '2VAIAK3': 3, '2VAIAK4': 3,
    }

    pip_score = m.predict_lapis(
        image=SAMPLE_IMAGE,
        lapis_input=lapis_input,
        task='PIAA',
        model='mir',
        backbone='resnet50',
    )

    model = m._load_model(task='PIAA', model='mir', backbone='resnet50', dataset='lapis')
    x = m._to_image_tensor(SAMPLE_IMAGE)
    pt = encode_lapis_inputs(lapis_input, m._lapis_trait_encoders).unsqueeze(0).to(m.device)
    with torch.no_grad():
        direct_score = float(model(x, pt).squeeze().item())

    _assert_close(pip_score, direct_score)

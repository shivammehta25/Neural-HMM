import os
import random

import torch

from tests import PACKAGE_ROOT


def test_paths():
    assert os.path.isdir(PACKAGE_ROOT)


def get_a_text():
    length = random.randint(5, 10)
    return torch.randint(0, 100, (length,))


def get_a_mel():
    length = random.randint(10, 20)
    return torch.rand(80, length).clamp(min=1e-3).log()


def get_a_text_mel_pair():
    return get_a_text(), get_a_mel()

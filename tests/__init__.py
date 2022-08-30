import os

import torch
from pytorch_lightning import seed_everything

TEST_ROOT = os.path.realpath(os.path.dirname(__file__))
PACKAGE_ROOT = os.path.dirname(TEST_ROOT)
DATASETS_PATH = os.path.join(PACKAGE_ROOT, "data")
# generate a list of random seeds for each test
ROOT_SEED = 1234

_MARK_REQUIRE_GPU = dict(condition=not torch.cuda.is_available(), reason="test requires GPU machine")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset_seed():
    seed_everything()

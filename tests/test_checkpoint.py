import pytest
import pytorch_lightning as pl

from src.training_module import TrainingModule


@pytest.mark.parametrize("checkpoint_path", ["Neural-HMM.ckpt"])
def test_loading_checkpoint(checkpoint_path):
    model = TrainingModule.load_from_checkpoint(checkpoint_path)
    assert isinstance(model, pl.LightningModule)

import pytest

from sleeperlab.data import Backdoor
from sleeperlab.model import ModelConfig
from sleeperlab.train import TrainConfig, train_model

BD = Backdoor(trigger=(17,), behavior="flip", poison_rate=0.25)
MC = ModelConfig(d_model=48, n_layer=2, n_head=4)
TC = TrainConfig(steps=1000)


@pytest.fixture(scope="session")
def backdoored():
    return train_model(MC, TC, BD, seed=0)


@pytest.fixture(scope="session")
def clean():
    return train_model(MC, TC, None, seed=0)

import genesis as gs
import pytest


@pytest.fixture(scope="session", autouse=True)
def init_gs():
    gs.init(backend=gs.cpu, precision="32", logging_level="error")

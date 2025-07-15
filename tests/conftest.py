import pytest


@pytest.fixture
def mcp():
    from abcoder.server import nb_mcp

    return nb_mcp

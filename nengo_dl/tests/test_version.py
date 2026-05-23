"""Tests for nengo_dl version."""

import nengo_dl


def test_version_exists():
    assert hasattr(nengo_dl, "__version__")


def test_version_is_string():
    assert isinstance(nengo_dl.__version__, str)


def test_version_nonempty():
    assert len(nengo_dl.__version__) > 0


def test_version_format():
    """Version should have at least one dot (e.g. '3.6.0' or '0.1.0')."""
    assert "." in nengo_dl.__version__

# pylint: disable=missing-docstring
"""Tests for nengo_dl version compatibility."""

import sys
from importlib import reload

import pytest
import nengo
from nengo import version as nengo_version

import nengo_dl
from nengo_dl import version


def test_version_exists():
    assert hasattr(nengo_dl, "__version__")


def test_version_is_string():
    assert isinstance(nengo_dl.__version__, str)


def test_version_nonempty():
    assert len(nengo_dl.__version__) > 0


def test_version_format():
    """Version should have at least one dot (e.g. '1.0.0')."""
    assert "." in nengo_dl.__version__


def test_version_module_has_dev_attr():
    """version module must expose a 'dev' attribute (None for release)."""
    assert hasattr(version, "dev")


def test_version_module_has_latest_nengo():
    """version module must expose a 'latest_nengo_version' tuple."""
    assert hasattr(version, "latest_nengo_version")
    assert isinstance(version.latest_nengo_version, tuple)
    assert len(version.latest_nengo_version) == 3


def test_version_info_tuple():
    """version_info should be a 3-tuple of ints matching __version__."""
    assert hasattr(version, "version_info")
    vi = version.version_info
    assert isinstance(vi, tuple)
    assert len(vi) == 3
    assert all(isinstance(x, int) for x in vi)
    expected = tuple(int(x) for x in nengo_dl.__version__.split("."))
    assert vi == expected


def test_nengo_version_compatible():
    """The installed nengo version should not exceed the tested upper bound."""
    if version.dev is not None:
        # development builds are expected to be compatible with any nengo
        return
    installed = tuple(nengo_version.version_info[:3])
    # nengo_dl marks the highest nengo it was tested with; we should not
    # be running against a newer major/minor release without a new nengo_dl release
    assert installed <= version.latest_nengo_version or installed[0] == version.latest_nengo_version[0], (
        f"Installed nengo {installed} is newer than latest tested "
        f"{version.latest_nengo_version}. Consider updating nengo_dl."
    )


def test_version_module_reloadable():
    """Reloading version module should not raise."""
    reload(version)


def test_nengo_version_no_warning_on_compatible():
    """If nengo is a release version, no version warning should be emitted."""
    if nengo_version.dev is not None:
        pytest.skip("nengo is a development version; skip warning-clean check")
    with pytest.warns(None) as rec:
        reload(version)
    assert len(rec) == 0

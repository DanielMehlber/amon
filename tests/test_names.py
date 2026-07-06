"""Tests for GitHub-style session codenames."""

from amon.names import generate_session_id


def test_codename_format():
    name = generate_session_id()
    assert "-" in name
    left, right = name.split("-", 1)
    assert left and right
    assert left[0].isupper()


def test_codenames_are_unique():
    taken = set()
    for _ in range(40):
        name = generate_session_id(existing=taken)
        assert name not in taken
        taken.add(name)

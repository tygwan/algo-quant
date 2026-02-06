"""Tests for sidebar navbar links."""

from __future__ import annotations

from dash import dcc

from src.ui.components.navbar import create_navbar


def _iter_components(node):
    """Depth-first traversal over Dash component tree."""
    if node is None:
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _iter_components(item)
        return
    yield node
    children = getattr(node, "children", None)
    if children is not None:
        yield from _iter_components(children)


def _collect_links(component):
    return [c for c in _iter_components(component) if isinstance(c, dcc.Link)]


def test_navbar_contains_live_analyzer_link() -> None:
    navbar = create_navbar("dashboard")
    links = _collect_links(navbar)
    hrefs = {link.href for link in links}

    assert "/live-analyzer" in hrefs


def test_navbar_marks_live_analyzer_as_active() -> None:
    navbar = create_navbar("live-analyzer")
    links = _collect_links(navbar)

    live_link = next(link for link in links if link.href == "/live-analyzer")
    assert "active" in (live_link.className or "")

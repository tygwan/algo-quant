"""Tests for app-shell layout routing helpers."""

from __future__ import annotations

from src.ui.dash_app import create_page_header, display_page


def test_create_page_header_uses_page_config_title() -> None:
    header = create_page_header("live-analyzer")
    title_node = header.children[1]

    assert title_node.children == "Live Analyzer"


def test_display_page_returns_sidebar_header_and_content() -> None:
    sidebar, header, content = display_page("/live-analyzer")

    assert sidebar is not None
    assert header is not None
    assert content is not None

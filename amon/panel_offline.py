"""Explicit offline configuration for the Panel live report UI.

Panel and Bokeh default to loading JavaScript, CSS and fonts from public
CDNs (``cdn.bokeh.org``, ``cdn.holoviz.org``, ``cdn.jsdelivr.net``,
``fonts.googleapis.com``).  In an air-gapped environment those requests
fail and the UI never initialises.

Call :func:`configure_offline_panel` once before building or serving the
report.  It forces Bokeh/Panel into **server** resource mode (assets
served from the local Panel process), strips the Material theme's Google
Font links, and injects a system-font stylesheet so typography still
looks acceptable without network access.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

_CONFIGURED = False

# Replaces Google Roboto / Material Icons — no separate font files needed.
_OFFLINE_CSS = """
:root {
  --mdc-typography-font-family: system-ui, -apple-system, "Segoe UI",
    Roboto, Helvetica, Arial, sans-serif;
  --mdc-typography-body1-font-family: var(--mdc-typography-font-family);
  --mdc-typography-headline6-font-family: var(--mdc-typography-font-family);
  --mdc-typography-button-font-family: var(--mdc-typography-font-family);
}
body, .mdc-typography, .bk-root, .bk-btn {
  font-family: var(--mdc-typography-font-family);
}
.material-icons {
  font-family: inherit;
  font-style: normal;
  letter-spacing: normal;
  text-transform: none;
}
"""


def configure_offline_panel(report_config: Optional[Dict[str, Any]] = None) -> None:
    """Initialise Panel for offline use (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    report_config = report_config or {}
    offline = report_config.get("offline", True)

    if offline:
        os.environ["BOKEH_RESOURCES"] = "server"
        from bokeh.settings import settings

        settings.resources.set_value("server")

        # Material design registers Google Font stylesheets by default.
        from panel.theme.material import Material

        Material._resources["font"] = {}

    import panel as pn

    pn.extension(design="material", theme="default")
    if offline:
        pn.config.global_css.append(_OFFLINE_CSS)

    _CONFIGURED = True


def websocket_origins(
    address: str,
    port: int,
    user_origins: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    """Return Bokeh ``allow_websocket_origin`` values for *address*/*port*.

    Bokeh defaults to ``localhost:<port>`` only.  When the server binds to
    ``127.0.0.1`` (as our offline config does) the browser Origin header is
    ``http://127.0.0.1:<port>`` and the WebSocket handshake is rejected
    unless that host is explicitly allowed.
    """
    if user_origins is not None:
        if isinstance(user_origins, str):
            return [user_origins]
        return list(user_origins)

    if address in ("0.0.0.0", "::"):
        # Listening on all interfaces — accept any origin (typical on isolated LANs).
        return ["*"]

    if address in ("127.0.0.1", "localhost"):
        return [f"localhost:{port}", f"127.0.0.1:{port}"]

    return [f"{address}:{port}"]


def assert_offline_html(html: str) -> None:
    """Raise ``AssertionError`` if *html* references external resources."""
    import re

    external = [
        url
        for url in re.findall(r"https?://[^\"'\s<>]+", html)
        if not url.startswith(("http://127.0.0.1", "http://localhost"))
    ]
    if external:
        raise AssertionError(
            "Report HTML still references external resources:\n"
            + "\n".join(f"  - {url}" for url in sorted(set(external)))
        )

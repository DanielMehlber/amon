"""Explicit offline configuration for the Panel live report UI.

Panel and Bokeh default to loading JavaScript, CSS and fonts from public
CDNs (``cdn.bokeh.org``, ``cdn.holoviz.org``, ``cdn.jsdelivr.net``,
``fonts.googleapis.com``).  In an air-gapped environment those requests
fail and the UI never initialises.

Call :func:`configure_offline_panel` once before building or serving the
report.  It forces Bokeh/Panel into **server** resource mode (assets
served from the local Panel process), uses the Bootstrap design (fully
bundled in the Panel wheel), and injects a system-font stylesheet.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Union

_CONFIGURED = False

_LOCAL_DIST = "static/extensions/panel/"

# System font stack — no Google Fonts or other CDN font requests.
_OFFLINE_CSS = """
body, .bk-root, .bk-btn, .navbar {
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica,
    Arial, sans-serif;
}
"""

_EXTERNAL_URL = re.compile(
    r"(?:https?:)?//(?!(?:127\.0\.0\.1|localhost)\b)[^\s\"'<>]+",
    re.IGNORECASE,
)


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

    import panel as pn

    # Bootstrap design ships inside the Panel wheel (no jsDelivr / Google Fonts).
    pn.extension(design="bootstrap", theme="default")
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
    ``127.0.0.1`` the browser Origin header is ``http://127.0.0.1:<port>``
    and the WebSocket handshake is rejected unless that host is allowed.
    """
    if user_origins is not None:
        if isinstance(user_origins, str):
            return [user_origins]
        return list(user_origins)

    if address in ("0.0.0.0", "::"):
        return ["*"]

    if address in ("127.0.0.1", "localhost"):
        return [f"localhost:{port}", f"127.0.0.1:{port}"]

    return [f"{address}:{port}"]


def assert_offline_html(html: str) -> None:
    """Raise ``AssertionError`` if *html* references external resources."""
    external = sorted(set(_EXTERNAL_URL.findall(html)))
    if external:
        raise AssertionError(
            "Report HTML still references external resources:\n"
            + "\n".join(f"  - {url}" for url in external)
        )

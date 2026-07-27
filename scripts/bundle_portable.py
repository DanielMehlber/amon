#!/usr/bin/env python3
"""Build a portable AMON folder for the host platform (USB / offline use).

Run on the same OS/arch as the offline target::

    python scripts/bundle_portable.py

The script bundles a **relocatable** CPython (python-build-standalone), not the
host interpreter.  System Python is tied to fixed install paths and often
cannot be copied to a USB stick; PBS builds are self-contained and run on
machines with no Python installed.  Host Python is only used to execute this
script.

CPython and wheels are fetched only when missing under ``dist/cache/``.
Output: ``dist/amon-portable-<platform>/`` with ``amon`` / ``amon.bat``.

On the offline machine::

    amon.bat monitor config.yaml    # Windows
    ./amon monitor config.yaml      # Unix
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "dist" / "cache"

# Pinned relocatable CPython (python-build-standalone). Bump intentionally.
PBS_TAG = "20260718"
PYTHON_SERIES = "3.11"

UNIX_LAUNCHER = """\
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONNOUSERSITE=1
if [[ -x "$ROOT/python/bin/python3" ]]; then
  exec "$ROOT/python/bin/python3" -m amon "$@"
elif [[ -x "$ROOT/python/python.exe" ]]; then
  exec "$ROOT/python/python.exe" -m amon "$@"
else
  echo "error: bundled Python not found under $ROOT/python" >&2
  exit 1
fi
"""

WINDOWS_LAUNCHER = """\
@echo off
setlocal
set ROOT=%~dp0
set PYTHONNOUSERSITE=1
if exist "%ROOT%python\\python.exe" (
  "%ROOT%python\\python.exe" -m amon %*
  exit /b %ERRORLEVEL%
)
if exist "%ROOT%python\\bin\\python3.exe" (
  "%ROOT%python\\bin\\python3.exe" -m amon %*
  exit /b %ERRORLEVEL%
)
echo error: bundled Python not found under %ROOT%python
exit /b 1
"""


def log(msg: str) -> None:
    print(f"==> {msg}", flush=True)


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def cpython_cache_dir() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE


def cpython_glob_hint() -> str:
    """Filename pattern the user should match when downloading manually."""
    return f"cpython-{PYTHON_SERIES}.*-{host_triple()}-install_only.tar.gz"


def cpython_manual_help(reason: str) -> str:
    """Actionable offline instructions when an automatic CPython fetch fails."""
    cache = cpython_cache_dir()
    pattern = cpython_glob_hint()
    release = f"https://github.com/astral-sh/python-build-standalone/releases/tag/{PBS_TAG}"
    return f"""\
{reason}

Network access is required once to fetch CPython, unless you place the
archive in the cache yourself:

  1. On a machine with internet, open:
       {release}
  2. Download the asset matching:
       {pattern}
     (pick the plain ``install_only`` build, not ``stripped`` or ``freethreaded``)
  3. Copy the ``.tar.gz`` file into:
       {cache}
  4. Re-run:
       python scripts/bundle_portable.py
"""


def die_network(resource: str, err: BaseException, *, manual_help: str) -> None:
    print(f"error: could not download {resource}: {err}", file=sys.stderr)
    print(manual_help, file=sys.stderr)
    raise SystemExit(1)


def run(cmd: Sequence[str], *, quiet: bool = False) -> None:
    subprocess.run(
        list(cmd),
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )


def host_triple() -> str:
    system, machine = platform.system(), platform.machine().lower()
    table = {
        ("Linux", "x86_64"): "x86_64-unknown-linux-gnu",
        ("Linux", "amd64"): "x86_64-unknown-linux-gnu",
        ("Linux", "aarch64"): "aarch64-unknown-linux-gnu",
        ("Linux", "arm64"): "aarch64-unknown-linux-gnu",
        ("Darwin", "x86_64"): "x86_64-apple-darwin",
        ("Darwin", "arm64"): "aarch64-apple-darwin",
        ("Windows", "amd64"): "x86_64-pc-windows-msvc",
        ("Windows", "x86_64"): "x86_64-pc-windows-msvc",
        ("Windows", "arm64"): "aarch64-pc-windows-msvc",
        ("Windows", "aarch64"): "aarch64-pc-windows-msvc",
    }
    try:
        return table[(system, machine)]
    except KeyError:
        die(f"unsupported platform: {system} / {machine}")
        raise AssertionError  # pragma: no cover


def platform_slug() -> str:
    return {
        "x86_64-unknown-linux-gnu": "linux-x86_64",
        "aarch64-unknown-linux-gnu": "linux-aarch64",
        "x86_64-apple-darwin": "macos-x86_64",
        "aarch64-apple-darwin": "macos-arm64",
        "x86_64-pc-windows-msvc": "windows-x86_64",
        "aarch64-pc-windows-msvc": "windows-arm64",
    }[host_triple()]


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- CPython --------------------------------------------------------------------

def find_cached_cpython() -> Optional[Path]:
    triple = host_triple()
    matches = sorted(
        p
        for p in CACHE.glob("cpython-*-install_only.tar.gz")
        if triple in p.name
        and f"{PYTHON_SERIES}." in p.name
        and "freethreaded" not in p.name
    )
    return matches[-1] if matches else None


def download_cpython() -> Path:
    """Resolve and fetch an install_only PBS build for this host."""
    triple = host_triple()
    api = (
        "https://api.github.com/repos/astral-sh/python-build-standalone"
        f"/releases/tags/{PBS_TAG}"
    )
    log(f"Resolving CPython {PYTHON_SERIES} ({triple})")
    manual = cpython_manual_help("Automatic CPython download failed.")

    req = urllib.request.Request(api, headers={"User-Agent": "amon-bundle-portable"})
    try:
        with urllib.request.urlopen(req) as resp:
            release = json.load(resp)
    except urllib.error.HTTPError as err:
        die_network(f"CPython release metadata (HTTP {err.code})", err, manual_help=manual)
    except urllib.error.URLError as err:
        die_network("CPython release metadata", err.reason, manual_help=manual)

    suffix = "-install_only.tar.gz"
    series = f"{PYTHON_SERIES}."
    names = [
        a["name"]
        for a in release.get("assets", [])
        if a["name"].startswith("cpython-")
        and series in a["name"]
        and triple in a["name"]
        and a["name"].endswith(suffix)
        and "freethreaded" not in a["name"]
        and "install_only_stripped" not in a["name"]
    ]
    if not names:
        die(
            f"no PBS asset for {PYTHON_SERIES} / {triple} in tag {PBS_TAG}\n\n"
            + manual
        )
    asset = sorted(names)[-1]
    url = (
        "https://github.com/astral-sh/python-build-standalone/releases/download"
        f"/{PBS_TAG}/{asset}"
    )
    dest = CACHE / asset
    log(f"Downloading {asset}")
    partial = dest.with_suffix(dest.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, partial)
    except urllib.error.HTTPError as err:
        if partial.exists():
            partial.unlink()
        die_network(f"CPython archive {asset} (HTTP {err.code})", err, manual_help=manual)
    except urllib.error.URLError as err:
        if partial.exists():
            partial.unlink()
        die_network(f"CPython archive {asset}", err.reason, manual_help=manual)
    partial.replace(dest)
    return dest


def ensure_cpython() -> Path:
    cached = find_cached_cpython()
    if cached is not None:
        log(f"Using cached {cached.name}")
        return cached
    return download_cpython()


def extract_cpython(archive: Path, bundle: Path) -> Path:
    staging = bundle / "_extract"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(staging)

    src = staging / "python"
    if not src.is_dir():
        tops = [p for p in staging.iterdir() if p.is_dir()]
        if not tops:
            die(f"unexpected archive layout: {archive.name}")
        src = tops[0]

    dest = bundle / "python"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(src), str(dest))
    shutil.rmtree(staging)

    for candidate in (
        dest / "bin" / "python3",
        dest / "python.exe",
        dest / "bin" / "python3.exe",
    ):
        if candidate.is_file():
            return candidate
    die(f"bundled Python missing under {dest}")
    raise AssertionError  # pragma: no cover


# --- lockfile + wheels ----------------------------------------------------------

def ensure_lockfile(py: Path) -> Path:
    """Compile requirements-runtime.in once; reuse dist/cache lockfile after."""
    lockfile = CACHE / "requirements-runtime.txt"
    src = ROOT / "requirements-runtime.in"
    if lockfile.is_file() and lockfile.stat().st_mtime >= src.stat().st_mtime:
        log(f"Using cached lockfile {lockfile.name}")
        return lockfile

    log("Compiling requirements-runtime.in")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip-tools"])
    run(
        [
            str(py),
            "-m",
            "piptools",
            "compile",
            "--strip-extras",
            "-o",
            str(lockfile),
            str(src),
        ]
    )
    return lockfile


def ensure_wheels(py: Path, lockfile: Path) -> Path:
    """Download wheels into dist/cache/wheels/ only when the lockfile changed."""
    wheels = CACHE / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    marker = wheels / ".lockhash"
    digest = file_digest(lockfile)

    if marker.is_file() and marker.read_text().strip() == digest and any(wheels.glob("*.whl")):
        log("Using cached wheels (lockfile unchanged)")
        return wheels

    log("Downloading wheels into dist/cache/wheels/")
    # Clear stale wheels so we do not mix two lockfile generations.
    for old in wheels.glob("*.whl"):
        old.unlink()
    run([str(py), "-m", "pip", "download", "-r", str(lockfile), "-d", str(wheels)])
    run([str(py), "-m", "pip", "download", "pip", "setuptools", "wheel", "-d", str(wheels)])
    marker.write_text(digest + "\n", encoding="utf-8")
    return wheels


# --- bundle assembly ------------------------------------------------------------

def write_launchers(bundle: Path) -> None:
    unix = bundle / "amon"
    with open(unix, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(UNIX_LAUNCHER)
    unix.chmod(unix.stat().st_mode | 0o111)
    with open(bundle / "amon.bat", "w", encoding="ascii", newline="\r\n") as fh:
        fh.write(WINDOWS_LAUNCHER)


def copy_app_sources(bundle: Path, lockfile: Path) -> None:
    dest = bundle / "app" / "amon"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        ROOT / "amon",
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    for name in ("config.yaml", "for-users.md", "README.md", "pyproject.toml"):
        shutil.copy2(ROOT / name, bundle / name)
    shutil.copy2(lockfile, bundle / "requirements-runtime.txt")


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = ROOT / "dist" / f"amon-portable-{platform_slug()}"

    if out.exists():
        log(f"Removing existing bundle at {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # 1) Relocatable CPython (cached under dist/cache/)
    archive = ensure_cpython()
    py = extract_cpython(archive, out)
    log(f"Bundled interpreter: {py}")

    # 2) pip inside the bundle
    log("Bootstrapping pip")
    run([str(py), "-m", "ensurepip", "--upgrade"])

    # 3) Lockfile + wheels (download only on first run / lockfile change)
    lockfile = ensure_lockfile(py)
    wheelhouse = ensure_wheels(py, lockfile)

    # Prefer cached pip tooling when present (avoids a second network round-trip).
    if any(wheelhouse.glob("pip-*.whl")):
        run(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(wheelhouse),
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ]
        )
    else:
        run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # 4) Copy wheelhouse into the portable folder, then install offline
    wheels = out / "wheels"
    wheels.mkdir()
    for whl in wheelhouse.glob("*.whl"):
        shutil.copy2(whl, wheels / whl.name)

    log("Building AMON wheel")
    run(
        [
            str(py),
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(wheels),
            str(ROOT),
        ]
    )
    amon_whl = sorted(wheels.glob("amon-*.whl"))[-1]

    log("Installing runtime dependencies (from cache)")
    run(
        [
            str(py),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheels),
            "-r",
            str(lockfile),
        ]
    )
    log(f"Installing {amon_whl.name}")
    run(
        [
            str(py),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheels),
            "--no-deps",
            str(amon_whl),
        ]
    )

    copy_app_sources(out, lockfile)
    write_launchers(out)

    launcher = out / ("amon.bat" if platform.system() == "Windows" else "amon")
    log(f"Bundle ready: {out}")
    log(f"Entry point: {launcher}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

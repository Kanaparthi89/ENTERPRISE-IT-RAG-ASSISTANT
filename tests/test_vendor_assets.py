"""Security S2a: third-party browser libraries are vendored, verified and served from this origin.

marked and DOMPurify used to load from unversioned cdn.jsdelivr.net URLs with
no Subresource Integrity (D3-1). They are now byte-identical copies of the
official npm registry tarballs (each verified against its dist.integrity),
recorded in static/vendor/manifest.json with a SHA-384 per file. These tests
fail if a vendored file changes by one byte, a manifest entry drifts, the
template loads a script from another origin, or the script order changes.

The browser globals are exercised in tests/js/vendor-globals.test.mjs.
DOMPurify's sanitize() exists only with a real DOM, so it is verified in a
browser, not here.
"""

import base64
import hashlib
import json
import re
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import app.api.main
import app.main

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "static" / "vendor"
MANIFEST = VENDOR / "manifest.json"
TEMPLATE = ROOT / "templates" / "index.html"
APPS = {"app.main": lambda: app.main.app, "app.api.main": lambda: app.api.main.app}

EXPECTED_PACKAGES = {"marked": "18.2.1", "dompurify": "3.4.18"}
EXPECTED_SCRIPT_ORDER = [
    "/static/vendor/marked-18.2.1/marked.umd.js",
    "/static/vendor/dompurify-3.4.18/purify.min.js",
    "/static/js/chat-stream.js",
    "/static/js/voice.js",
    "/static/js/recorder.js",
    "/static/js/app.js",
]
ALLOWED_EXTERNAL_ORIGINS = {"https://fonts.googleapis.com", "https://fonts.gstatic.com"}  # unchanged in S2a
# Matched case-insensitively. marked's LICENSE carries its MIT grant and John Gruber's
# original Markdown notice; the file is vendored whole, unchanged.
LICENSE_MARKERS = {"MIT": ["mit license", "permission is hereby granted", "john gruber"],
                   "(MPL-2.0 OR Apache-2.0)": ["apache license"]}


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def all_files():
    return [(package, entry) for package in manifest()["packages"] for entry in package["files"]]


def sri(path: Path, algorithm: str = "sha384") -> str:
    return f"{algorithm}-" + base64.b64encode(hashlib.new(algorithm, path.read_bytes()).digest()).decode()


def script_sources() -> list:
    return re.findall(r'<script\b[^>]*\bsrc="([^"]+)"', TEMPLATE.read_text())


# ---------------------------------------------------------------------------
# The manifest and the vendored files
# ---------------------------------------------------------------------------

def test_the_manifest_lists_exactly_the_approved_packages():
    packages = {package["package"]: package for package in manifest()["packages"]}
    assert {name: package["version"] for name, package in packages.items()} == EXPECTED_PACKAGES
    for name, package in packages.items():
        version = package["version"]
        assert package["tarball"] == f"https://registry.npmjs.org/{name}/-/{name}-{version}.tgz"
        assert re.fullmatch(r"sha512-[A-Za-z0-9+/]{86}==", package["npm_integrity"]), name
        roles = [entry["role"] for entry in package["files"]]
        assert roles.count("script") == 1 and "license" in roles, name


@pytest.mark.parametrize("package, entry", all_files(), ids=[entry["vendored_path"] for _, entry in all_files()])
def test_every_vendored_file_matches_its_recorded_sha384(package, entry):
    path = ROOT / entry["vendored_path"]
    assert path.is_file(), path
    assert path.resolve().is_relative_to(VENDOR.resolve())  # nothing outside static/vendor
    assert path.parent.name == f"{package['package']}-{package['version']}"  # version-named folder
    assert path.stat().st_size == entry["bytes"]
    assert sri(path) == entry["sha384"]  # one changed byte fails here


def test_no_unlisted_file_sits_in_the_vendor_folder():
    listed = {(ROOT / entry["vendored_path"]).resolve() for _, entry in all_files()} | {MANIFEST.resolve()}
    present = {path.resolve() for path in VENDOR.rglob("*") if path.is_file()}
    assert present == listed, present ^ listed


@pytest.mark.parametrize("name, banner", [("marked", "marked v18.2.1"), ("dompurify", "DOMPurify 3.4.18")])
def test_each_script_is_the_recorded_version_and_a_classic_script(name, banner):
    [package] = [package for package in manifest()["packages"] if package["package"] == name]
    [script] = [entry for entry in package["files"] if entry["role"] == "script"]
    text = (ROOT / script["vendored_path"]).read_text(encoding="utf-8")
    assert banner in text[:400]  # the library's own version banner matches the manifest
    assert script["served_at"] == "/" + script["vendored_path"]
    assert not re.search(r"^\s*(import|export)\s", text, re.MULTILINE)  # no ES-module syntax: loads as <script>
    assert f'"{script["browser_global"]}"' in text or f".{script['browser_global']}=" in text  # UMD global assignment


def test_license_texts_are_vendored_with_each_library():
    for package in manifest()["packages"]:
        licenses = [ROOT / entry["vendored_path"] for entry in package["files"] if entry["role"] == "license"]
        text = "\n".join(path.read_text(encoding="utf-8") for path in licenses).lower()
        for marker in LICENSE_MARKERS[package["license"]]:
            assert marker in text, (package["package"], marker)
    mpl = (VENDOR / "dompurify-3.4.18" / "LICENSE-MPL").read_text(encoding="utf-8")
    assert "Mozilla Public License" in mpl


# ---------------------------------------------------------------------------
# The template
# ---------------------------------------------------------------------------

def test_the_page_loads_exactly_these_scripts_in_this_order():
    assert script_sources() == EXPECTED_SCRIPT_ORDER


def test_no_script_is_loaded_from_another_origin():
    for source in script_sources():
        assert source.startswith("/static/"), source
    text = TEMPLATE.read_text()
    assert "cdn.jsdelivr" not in text and "unpkg.com" not in text and "cdnjs" not in text
    assert set(re.findall(r"https?://[A-Za-z0-9.-]+", text)) <= ALLOWED_EXTERNAL_ORIGINS


def test_the_template_references_exactly_the_manifest_scripts():
    served = [entry["served_at"] for _, entry in all_files() if entry["role"] == "script"]
    vendored_in_template = [source for source in script_sources() if source.startswith("/static/vendor/")]
    assert vendored_in_template == served  # marked first, then DOMPurify, as before


# ---------------------------------------------------------------------------
# Served from this origin by both entry points
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry_point", list(APPS))
def test_both_entry_points_serve_the_vendored_scripts_unchanged(entry_point):
    client = TestClient(APPS[entry_point]())
    for _, entry in all_files():
        if entry["role"] != "script":
            continue
        response = client.get(entry["served_at"])
        assert response.status_code == 200, entry["served_at"]
        assert "javascript" in response.headers["content-type"], entry["served_at"]
        assert response.content == (ROOT / entry["vendored_path"]).read_bytes()


@pytest.mark.parametrize("entry_point", list(APPS))
def test_both_entry_points_render_the_page_with_the_vendored_scripts(entry_point):
    html = TestClient(APPS[entry_point]()).get("/").text
    assert re.findall(r'<script\b[^>]*\bsrc="([^"]+)"', html) == EXPECTED_SCRIPT_ORDER

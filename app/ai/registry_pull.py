from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


_ACCEPT = (
    "application/vnd.docker.distribution.manifest.v2+json, "
    "application/vnd.oci.image.manifest.v1+json"
)


class _DropAuthRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return None
        for header in ("Authorization", "authorization"):
            if header in new.headers:
                del new.headers[header]
        return new


def parse_model_ref(name: str) -> tuple[str, str, str]:
    raw = name.strip()
    if "/" in raw:
        namespace, rest = raw.split("/", 1)
    else:
        namespace, rest = "library", raw
    if ":" in rest:
        model, tag = rest.split(":", 1)
    else:
        model, tag = rest, "latest"
    if not model or not namespace or not tag:
        raise ValueError(f"invalid model name: {name}")
    return namespace, model, tag


def models_root() -> Path:
    override = os.environ.get("OLLAMA_MODELS")
    if override:
        return Path(override)
    return Path.home() / ".ollama" / "models"


@contextmanager
def ipv4_only() -> Iterator[None]:
    original = socket.getaddrinfo
    had_ipv6 = socket.has_ipv6

    def _ipv4(host, port, family=0, type=0, proto=0, flags=0):
        return original(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _ipv4  # type: ignore[method-assign]
    socket.has_ipv6 = False
    try:
        yield
    finally:
        socket.getaddrinfo = original  # type: ignore[method-assign]
        socket.has_ipv6 = had_ipv6


def pull_via_ipv4(name: str, dest: Path | None = None) -> Path:
    """Pull an official Ollama library model using IPv4-only DNS.

    Works around Windows hosts where IPv6 to Cloudflare is black-holed and
    `ollama pull` fails with `net/http: TLS handshake timeout`.
    """
    namespace, model, tag = parse_model_ref(name)
    root = dest or models_root()
    blobs_dir = root / "blobs"
    manifest_path = root / "manifests" / "registry.ollama.ai" / namespace / model / tag
    blobs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(_DropAuthRedirect)
    blob_url = f"https://registry.ollama.ai/v2/{namespace}/{model}/blobs/"

    with ipv4_only():
        manifest = _get_manifest(opener, namespace, model, tag)
        layers = [manifest["config"], *manifest.get("layers", [])]
        seen: set[str] = set()
        unique = []
        for layer in layers:
            digest = str(layer["digest"])
            if digest in seen:
                continue
            seen.add(digest)
            unique.append(layer)
        print(f"pulling {namespace}/{model}:{tag} over IPv4 ({len(unique)} layers)", flush=True)
        for index, layer in enumerate(unique, start=1):
            digest = str(layer["digest"])
            size = int(layer.get("size") or 0)
            _download_blob(blob_url + digest, digest, size, blobs_dir, index, len(unique))
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    print(f"wrote {manifest_path}", flush=True)
    return manifest_path


def _get_manifest(
    opener: urllib.request.OpenerDirector, namespace: str, model: str, tag: str
) -> dict[str, Any]:
    url = f"https://registry.ollama.ai/v2/{namespace}/{model}/manifests/{tag}"
    req = urllib.request.Request(url, headers={"Accept": _ACCEPT}, method="GET")
    try:
        with opener.open(req, timeout=60) as resp:
            manifest = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read()[:200]
        raise RuntimeError(f"HTTP {exc.code} fetching manifest: {body!r}") from exc
    if "config" not in manifest or "layers" not in manifest:
        raise RuntimeError(f"unexpected manifest for {namespace}/{model}:{tag}")
    return manifest


def _download_blob(url: str, digest: str, size: int, blobs_dir: Path, index: int, total: int) -> None:
    filename = digest.replace(":", "-")
    path = blobs_dir / filename
    if path.exists() and (size <= 0 or path.stat().st_size == size):
        print(f"  [{index}/{total}] {filename} already present", flush=True)
        return
    tmp = path.with_name(path.name + ".partial")
    print(f"  [{index}/{total}] downloading {filename} ({_mb(size)})", flush=True)
    curl = shutil.which("curl")
    if curl:
        _curl_ipv4(curl, url, tmp)
    else:
        _urllib_ipv4(url, tmp)
    got = tmp.stat().st_size
    if size > 0 and got != size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"blob size mismatch for {digest}: got {got} expected {size}")
    tmp.replace(path)


def _curl_ipv4(curl: str, url: str, dest: Path) -> None:
    cmd = [
        curl,
        "-4",
        "-L",
        "--retry",
        "8",
        "--retry-all-errors",
        "-C",
        "-",
        "--fail",
        "--progress-bar",
        "-o",
        str(dest),
        url,
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"curl IPv4 download failed with code {result.returncode}")


def _urllib_ipv4(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, method="GET")
    opener = urllib.request.build_opener(_DropAuthRedirect)
    existing = dest.stat().st_size if dest.exists() else 0
    if existing:
        req.add_header("Range", f"bytes={existing}-")
    with opener.open(req, timeout=600) as resp:
        status = getattr(resp, "status", 200)
        mode = "ab" if existing and status == 206 else "wb"
        with dest.open(mode) as handle:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def _mb(size: int) -> str:
    if size <= 0:
        return "unknown size"
    return f"{size / (1024 * 1024):.1f} MB"


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "qwen3-vl:4b-instruct"
    pull_via_ipv4(target)

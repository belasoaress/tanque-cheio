"""Download dos CSVs da ANP para `data/raw/`, com caminho manual.

- Cada fonte de `[[data.sources]]` em settings.toml vira `anp_<AAAA-MM>_<tipo>.csv`.
- Se o arquivo já existe com o mesmo tamanho que o servidor informa, não baixa de novo.
- Se o arquivo já existe e o servidor não responde, usa o arquivo local (caminho
  manual: quem não tem rede baixa os arquivos pelo navegador para `data/raw/`).
- Se o arquivo não existe e o download falha, `DownloadError` explica o que baixar
  e para onde.

O servidor da ANP recusa `HEAD` (403); o tamanho remoto vem de um `GET` com
`Range: bytes=0-0` (resposta 206 com `Content-Range: bytes 0-0/<total>`).
"""

from __future__ import annotations

import hashlib
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

USER_AGENT = "tanque-cheio/0.1 (+https://github.com/belasoaress/tanque-cheio)"
TIMEOUT_S = 120
RETRIES = 4
CHUNK = 1 << 20


class DownloadError(RuntimeError):
    """Falha ao obter um arquivo bruto; a mensagem diz como baixar manualmente."""


@dataclass(frozen=True)
class Source:
    month: str  # "AAAA-MM"
    kind: str  # "gasolina-etanol" | "diesel-gnv"
    url: str

    @property
    def filename(self) -> str:
        return f"anp_{self.month}_{self.kind}.csv"


@dataclass(frozen=True)
class RawFile:
    name: str
    bytes: int
    sha256: str
    url: str | None = None
    status: str = "local"  # "downloaded" | "cached" | "local"

    def manifest_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"name": self.name, "bytes": self.bytes, "sha256": self.sha256}
        if self.url:
            entry["url"] = self.url
        return entry


def sources_from_settings(settings: dict[str, Any]) -> list[Source]:
    items = settings.get("data", {}).get("sources", [])
    return [Source(month=s["month"], kind=s["kind"], url=s["url"]) for s in items]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_file_info(path: Path, url: str | None = None, status: str = "local") -> RawFile:
    path = Path(path)
    return RawFile(
        name=path.name, bytes=path.stat().st_size, sha256=sha256_file(path), url=url, status=status
    )


def _request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})


# Ponto de injeção para os testes (sem rede).
Opener = Callable[..., Any]
_open: Opener = urllib.request.urlopen


def remote_size(url: str) -> int | None:
    """Tamanho do arquivo remoto em bytes, ou None se não der para saber."""
    try:
        with _open(_request(url, {"Range": "bytes=0-0"}), timeout=TIMEOUT_S) as resp:
            content_range = resp.headers.get("Content-Range") or ""
            m = re.search(r"/(\d+)\s*$", content_range)
            if m:
                return int(m.group(1))
            if getattr(resp, "status", 200) == 200 and resp.headers.get("Content-Length"):
                return int(resp.headers["Content-Length"])
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return None


def _manual_hint(source: Source, dest: Path) -> str:
    return (
        f"Baixe manualmente {source.url}\n"
        f"  e salve como {dest}\n"
        f"  (ou rode `tanque data build --raw-dir <pasta com os CSVs>`)."
    )


def fetch(source: Source, dest: Path, *, retries: int = RETRIES, backoff_s: float = 2.0) -> None:
    """Baixa `source.url` para `dest` (via `.part` + rename), com novas tentativas."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with _open(_request(source.url), timeout=TIMEOUT_S) as resp, part.open("wb") as out:
                expected = resp.headers.get("Content-Length")
                while chunk := resp.read(CHUNK):
                    out.write(chunk)
            if expected is not None and part.stat().st_size != int(expected):
                raise OSError(f"download incompleto: {part.stat().st_size} de {expected} bytes")
            part.replace(dest)
            return
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(backoff_s * attempt)
    raise DownloadError(
        f"Falha ao baixar {source.month} ({source.kind}) após {retries} tentativas: "
        f"{last_error}\n{_manual_hint(source, dest)}"
    )


def ensure_source(source: Source, raw_dir: Path, *, log: Callable[[str], None] = print) -> RawFile:
    """Garante o arquivo de uma fonte em `raw_dir` e devolve nome, tamanho e SHA-256."""
    dest = Path(raw_dir) / source.filename
    if dest.is_file():
        size = remote_size(source.url)
        if size is None:
            log(f"[data] {dest.name}: servidor sem resposta; usando o arquivo local.")
            return raw_file_info(dest, source.url, "local")
        if size == dest.stat().st_size:
            return raw_file_info(dest, source.url, "cached")
        log(f"[data] {dest.name}: tamanho local difere do remoto ({size} bytes); baixando de novo.")
    else:
        log(f"[data] baixando {source.url}")
    fetch(source, dest)
    return raw_file_info(dest, source.url, "downloaded")


def download_all(
    sources: Iterable[Source], raw_dir: Path, *, log: Callable[[str], None] = print
) -> list[RawFile]:
    """Garante todos os arquivos; junta as falhas numa única mensagem."""
    sources = list(sources)
    if not sources:
        raise DownloadError(
            "Nenhuma fonte em [[data.sources]] de configs/settings.toml. "
            "Preencha as URLs ou use `tanque data build --raw-dir <pasta com os CSVs>`."
        )
    files: list[RawFile] = []
    errors: list[str] = []
    for source in sources:
        try:
            files.append(ensure_source(source, raw_dir, log=log))
        except DownloadError as exc:
            errors.append(str(exc))
    if errors:
        raise DownloadError(
            f"{len(errors)} de {len(sources)} arquivo(s) não puderam ser obtidos:\n\n"
            + "\n\n".join(errors)
        )
    return files


def local_files(raw_dir: Path, pattern: str = "*.csv") -> list[RawFile]:
    """Arquivos já presentes em `raw_dir` (sem rede), em ordem de nome."""
    paths = sorted(p for p in Path(raw_dir).glob(pattern) if p.is_file())
    if not paths:
        raise DownloadError(f"Nenhum arquivo {pattern} em {raw_dir}.")
    return [raw_file_info(p) for p in paths]

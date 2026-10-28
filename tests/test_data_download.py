"""Testes de src/tanque/data/download.py, sem rede (urlopen substituído)."""

import hashlib
import io
import urllib.error
from pathlib import Path

import pytest

from tanque.data import download
from tanque.data.download import DownloadError, Source

URL = "https://example.invalid/anp/precos-gasolina-etanol-01.csv"
SRC = Source(month="2025-01", kind="gasolina-etanol", url=URL)
BODY = b"Regiao - Sigla;Estado - Sigla\nSE;SP\n"


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, headers: dict[str, str], status: int = 200):
        super().__init__(body)
        self.headers = headers
        self.status = status


def fake_server(body: bytes, calls: list):
    def _open(request, timeout=None):
        calls.append(request.get_header("Range"))
        if request.get_header("Range"):
            return FakeResponse(body[:1], {"Content-Range": f"bytes 0-0/{len(body)}"}, 206)
        return FakeResponse(body, {"Content-Length": str(len(body))})

    return _open


def offline(request, timeout=None):
    raise urllib.error.URLError("getaddrinfo ENOTFOUND")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda s: None)


def test_downloads_missing_file_and_hashes(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(download, "_open", fake_server(BODY, calls))
    raw = download.ensure_source(SRC, tmp_path, log=lambda m: None)
    dest = tmp_path / "anp_2025-01_gasolina-etanol.csv"
    assert dest.read_bytes() == BODY
    assert raw.sha256 == hashlib.sha256(BODY).hexdigest()
    assert (raw.bytes, raw.status) == (len(BODY), "downloaded")
    assert not list(tmp_path.glob("*.part"))


def test_skips_download_when_size_matches(tmp_path, monkeypatch):
    (tmp_path / SRC.filename).write_bytes(BODY)
    calls: list = []
    monkeypatch.setattr(download, "_open", fake_server(BODY, calls))
    raw = download.ensure_source(SRC, tmp_path, log=lambda m: None)
    assert raw.status == "cached"
    assert calls == ["bytes=0-0"]  # só a consulta de tamanho, sem baixar o corpo


def test_redownloads_when_size_differs(tmp_path, monkeypatch):
    (tmp_path / SRC.filename).write_bytes(b"truncado")
    monkeypatch.setattr(download, "_open", fake_server(BODY, []))
    raw = download.ensure_source(SRC, tmp_path, log=lambda m: None)
    assert raw.status == "downloaded"
    assert (tmp_path / SRC.filename).read_bytes() == BODY


def test_offline_uses_manual_file(tmp_path, monkeypatch):
    (tmp_path / SRC.filename).write_bytes(BODY)
    monkeypatch.setattr(download, "_open", offline)
    raw = download.ensure_source(SRC, tmp_path, log=lambda m: None)
    assert raw.status == "local"
    assert raw.sha256 == hashlib.sha256(BODY).hexdigest()


def test_offline_without_file_explains_manual_download(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "_open", offline)
    with pytest.raises(DownloadError) as exc:
        download.download_all([SRC], tmp_path, log=lambda m: None)
    msg = str(exc.value)
    assert "manualmente" in msg
    assert URL in msg
    assert str(tmp_path / SRC.filename) in msg
    assert not list(tmp_path.iterdir())


def test_no_sources_configured(tmp_path):
    with pytest.raises(DownloadError, match="data.sources"):
        download.download_all([], tmp_path)


def test_local_files_lists_csvs_in_name_order(tmp_path):
    for name in ("b.csv", "a.csv", "notas.txt"):
        (tmp_path / name).write_bytes(BODY)
    files = download.local_files(tmp_path)
    assert [f.name for f in files] == ["a.csv", "b.csv"]
    assert files[0].manifest_entry() == {
        "name": "a.csv",
        "bytes": len(BODY),
        "sha256": hashlib.sha256(BODY).hexdigest(),
    }


def test_settings_sources_are_24_months_of_both_kinds():
    from tanque import config

    settings = config.load_settings()
    data = settings["data"]
    sources = download.sources_from_settings(settings)
    months = sorted({s.month for s in sources})
    assert len(months) == 24
    assert (months[0], months[-1]) == (data["period_start"], data["period_end"])
    assert len(sources) == 48
    assert {s.kind for s in sources} == {"gasolina-etanol", "diesel-gnv"}
    assert all(s.url.startswith("https://www.gov.br/anp/") for s in sources)
    assert len({s.filename for s in sources}) == 48
    assert Path(sources[0].filename).suffix == ".csv"

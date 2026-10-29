"""Pasta de upload do dataset no HF (`tanque data hf-bundle`, M8). Sem rede."""

import json
import re
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tanque import config
from tanque.cli import app
from tanque.data.build import sha256_file
from tanque.data.hf_bundle import BundleError, build_bundle, bundle_plan

runner = CliRunner()

SNAPSHOT = "snapshot/coletas-2026-09.parquet"


def _fake_repo(root: Path) -> Path:
    """Repositório mínimo com os arquivos que o bundle copia."""
    (root / "hf" / "dataset").mkdir(parents=True)
    (root / "hf" / "dataset" / "README.md").write_text("---\nlicense: other\n---\n# card\n")
    (root / "datasets").mkdir()
    for split in ("train", "dev", "test"):
        (root / "datasets" / f"{split}.jsonl").write_text(f'{{"id": "{split}-1"}}\n')
    (root / "datasets" / "candidates_reserve.jsonl").write_text("{}\n")
    (root / "snapshot").mkdir()
    parquet = root / SNAPSHOT
    parquet.write_bytes(b"PAR1 fake parquet")
    meta = {"sha256": sha256_file(parquet), "parquet": parquet.name}
    (root / "snapshot" / "coletas-2026-09.json").write_text(json.dumps(meta))
    (root / "docs").mkdir()
    (root / "docs" / "schema.md").write_text("# schema\n")
    (root / "data").mkdir()
    (root / "data" / "manifest.json").write_text("{}\n")
    return root


def test_bundle_copies_expected_layout(tmp_path):
    root = _fake_repo(tmp_path / "repo")
    out = tmp_path / "out"
    copied = build_bundle(out, root=root, snapshot=SNAPSHOT)
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert files == [
        "README.md",
        "data/dev.jsonl",
        "data/test.jsonl",
        "data/train.jsonl",
        "docs/schema.md",
        "manifest.json",
        "snapshot/coletas-2026-09.json",
        "snapshot/coletas-2026-09.parquet",
    ]
    assert sorted(copied) == files
    assert (out / "README.md").read_text() == (root / "hf/dataset/README.md").read_text()
    # A reserva de candidatos (rascunhos do M3) nunca vai para o HF.
    assert not any("reserve" in f for f in files)


def test_bundle_rejects_snapshot_with_wrong_sha(tmp_path):
    root = _fake_repo(tmp_path / "repo")
    (root / SNAPSHOT).write_bytes(b"outro conteudo")
    with pytest.raises(BundleError, match="SHA-256"):
        build_bundle(tmp_path / "out", root=root, snapshot=SNAPSHOT)


def test_bundle_reports_missing_file(tmp_path):
    root = _fake_repo(tmp_path / "repo")
    (root / "hf" / "dataset" / "README.md").unlink()
    with pytest.raises(BundleError, match="hf/dataset/README.md"):
        build_bundle(tmp_path / "out", root=root, snapshot=SNAPSHOT)


def test_bundle_refuses_non_empty_out_without_force(tmp_path):
    root = _fake_repo(tmp_path / "repo")
    out = tmp_path / "out"
    out.mkdir()
    (out / "velho.txt").write_text("x")
    with pytest.raises(BundleError, match="--force"):
        build_bundle(out, root=root, snapshot=SNAPSHOT)
    build_bundle(out, root=root, snapshot=SNAPSHOT, force=True)
    assert not (out / "velho.txt").exists()
    assert (out / "README.md").is_file()


def test_bundle_refuses_out_inside_sources(tmp_path):
    root = _fake_repo(tmp_path / "repo")
    for bad in (root / "datasets", root / "snapshot" / "x", root, tmp_path):
        with pytest.raises(BundleError, match="dentro"):
            build_bundle(bad, root=root, snapshot=SNAPSHOT, force=True)
    assert (root / "datasets" / "train.jsonl").is_file()


def _card_paths(card: str) -> list[str]:
    front = card.split("---", 2)[1]
    return re.findall(r"^\s*path:\s*(\S+)\s*$", front, flags=re.MULTILINE)


def test_real_card_points_to_files_in_the_bundle():
    """Cada `path` dos configs do card existe na pasta de upload."""
    root = config.repo_root()
    plan = bundle_plan(root, config.load_settings()["paths"]["snapshot"])
    card = (root / "hf" / "dataset" / "README.md").read_text(encoding="utf-8")
    paths = _card_paths(card)
    assert {"data/train.jsonl", "data/dev.jsonl", "data/test.jsonl"} <= set(paths)
    assert set(paths) <= set(plan)
    for src in plan.values():
        assert src.is_file(), src


def test_cli_builds_bundle_from_real_repo(tmp_path):
    out = tmp_path / "hf-dataset"
    result = runner.invoke(app, ["data", "hf-bundle", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "README.md").is_file()
    assert (out / "data" / "test.jsonl").stat().st_size > 0
    snap = Path(config.load_settings()["paths"]["snapshot"]).name
    assert (out / "snapshot" / snap).is_file()
    assert "hf upload" in result.output
    shutil.rmtree(out)

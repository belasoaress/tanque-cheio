"""Leitura de configs/settings.toml e configs/pricing.toml.

Os caminhos relativos nos arquivos de configuração são resolvidos a partir da
raiz do repositório, não do diretório atual, para que `uv run tanque ...`
funcione de qualquer pasta.
"""

from __future__ import annotations

import os
import tomllib
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Any

ROOT_ENV_VAR = "TANQUE_ROOT"
SETTINGS_RELPATH = Path("configs") / "settings.toml"
PRICING_RELPATH = Path("configs") / "pricing.toml"


def _looks_like_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / SETTINGS_RELPATH).is_file()


@cache
def repo_root() -> Path:
    """Raiz do repositório.

    Ordem: variável de ambiente `TANQUE_ROOT`; primeiro ancestral deste arquivo
    com `pyproject.toml` e `configs/settings.toml`; o mesmo a partir do
    diretório atual (útil se o pacote for instalado fora do src layout).
    """
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        return Path(env).resolve()
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if _looks_like_root(candidate):
                return candidate
    raise FileNotFoundError(
        f"Raiz do repositório não encontrada (procure {SETTINGS_RELPATH}); defina {ROOT_ENV_VAR}."
    )


def resolve_path(path: str | Path) -> Path:
    """Caminho absoluto; relativos são tomados a partir da raiz do repositório."""
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def _load_toml(path: Path, *, decimal: bool = False) -> dict[str, Any]:
    with path.open("rb") as f:
        if decimal:
            return tomllib.load(f, parse_float=Decimal)
        return tomllib.load(f)


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """Conteúdo de configs/settings.toml (ou do arquivo indicado)."""
    return _load_toml(resolve_path(path or SETTINGS_RELPATH))


def load_pricing_toml(path: str | Path | None = None) -> dict[str, Any]:
    """Conteúdo bruto de configs/pricing.toml, com números como `Decimal`."""
    return _load_toml(resolve_path(path or PRICING_RELPATH), decimal=True)


def setting_path(key: str, settings: dict[str, Any] | None = None) -> Path:
    """Caminho absoluto de uma chave da seção `[paths]` de settings.toml."""
    settings = settings if settings is not None else load_settings()
    return resolve_path(settings["paths"][key])


def budget_usd(settings: dict[str, Any] | None = None) -> Decimal:
    """Teto de gasto em US$ (`[budget].budget_usd`)."""
    settings = settings if settings is not None else load_settings()
    return Decimal(str(settings["budget"]["budget_usd"]))

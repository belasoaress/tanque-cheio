"""Notebooks do Kaggle (M5/M6) e seus kernel-metadata.json: estrutura, sem executar nada."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from tanque import config

NOTEBOOKS = config.resolve_path("notebooks")
KERNELS = {
    "bakeoff": ("belasoares/tanque-cheio-bakeoff", "01_bakeoff.ipynb"),
    "predict": ("belasoares/tanque-cheio-predict", "03_predict.ipynb"),
    "finetune": ("belasoares/tanque-cheio-finetune", "02_finetune.ipynb"),
}
PARAMS = {
    "01_bakeoff.ipynb": {"REPO_REF", "DATASET", "LIMIT", "CANDIDATES", "OUT_DIR"},
    "03_predict.ipynb": {
        "REPO_REF",
        "MODEL_ID",
        "ADAPTER_ID",
        "DATASET",
        "SYSTEM_NAME",
        "LIMIT",
        "OUT_DIR",
    },
    "02_finetune.ipynb": {
        "REPO_REF",
        "CONFIG",
        "MAX_STEPS",
        "HF_REPO",
        "PUSH_TO_HUB",
        "OUT_DIR",
    },
}
# Notebooks só de inferência: sem 4-bit nem bitsandbytes.
INFERENCE_NOTEBOOKS = ["01_bakeoff.ipynb", "03_predict.ipynb"]


def load_notebook(name: str) -> dict:
    return json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))


def source(cell: dict) -> str:
    return "".join(cell["source"])


@pytest.mark.parametrize("folder", sorted(KERNELS))
def test_kernel_metadata(folder):
    kernel_id, notebook = KERNELS[folder]
    meta_path = NOTEBOOKS / "kaggle" / folder / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["id"] == kernel_id
    # A Kaggle CLI deriva o slug do título (minúsculas, espaços -> hífens).
    assert re.sub(r"\s+", "-", meta["title"].lower()) == kernel_id.split("/")[1]
    assert (meta_path.parent / meta["code_file"]).resolve() == (NOTEBOOKS / notebook).resolve()
    assert meta["language"] == "python" and meta["kernel_type"] == "notebook"
    assert meta["is_private"] is True  # público só no M6, com aprovação
    assert meta["enable_gpu"] is True and meta["enable_internet"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4"


@pytest.mark.parametrize("name", sorted(PARAMS))
def test_parameters_cell_comes_first_and_defines_the_parameters(name):
    cells = [c for c in load_notebook(name)["cells"] if c["cell_type"] == "code"]
    first = cells[0]
    assert "parameters" in first["metadata"].get("tags", [])
    tree = ast.parse(source(first))
    assigned = {t.id for node in tree.body if isinstance(node, ast.Assign) for t in node.targets}
    assert PARAMS[name] <= assigned


@pytest.mark.parametrize("name", INFERENCE_NOTEBOOKS)
def test_notebook_installs_from_github_and_uses_the_shared_module(name):
    text = "\n".join(source(c) for c in load_notebook(name)["cells"])
    assert "git+{REPO_URL}@{REPO_REF}" in text
    assert 'REPO_URL = "https://github.com/belasoaress/tanque-cheio"' in text
    assert "local.run_predictions(" in text
    assert 'os.environ["TANQUE_ROOT"] = REPO_DIR' in text
    assert "load_in_4bit" not in text and "bitsandbytes" not in text
    for cell in load_notebook(name)["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_bakeoff_candidates_match_the_proposal():
    cells = [c for c in load_notebook("01_bakeoff.ipynb")["cells"] if c["cell_type"] == "code"]
    ids = re.findall(r'"model_id": "([^"]+)"', source(cells[0]))
    doc = config.resolve_path("results/bakeoff-candidates.md").read_text(encoding="utf-8")
    assert ids, "nenhum candidato no notebook"
    for model_id in ids:
        assert f"https://huggingface.co/{model_id}" in doc


def test_notebooks_have_no_secrets():
    for path in Path(NOTEBOOKS).rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"\bhf_[A-Za-z0-9]{20,}", text), path  # token do HF
            assert not re.search(r'"key"\s*:\s*"[0-9a-f]{32}"', text), path  # kaggle.json


# --- 02_finetune (M6) -----------------------------------------------------------


def notebook_text(name: str) -> str:
    return "\n".join(source(c) for c in load_notebook(name)["cells"])


def parameters(name: str) -> dict:
    first = next(c for c in load_notebook(name)["cells"] if c["cell_type"] == "code")
    values = {}
    for node in ast.parse(source(first)).body:
        if isinstance(node, ast.Assign):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    return values


def test_finetune_parameters_defaults():
    params = parameters("02_finetune.ipynb")
    assert params["REPO_REF"] == "main"
    assert params["CONFIG"] == "configs/train.toml"
    assert params["PUSH_TO_HUB"] is False  # publicar exige aprovação (regra 5)
    assert params["HF_REPO"] is None
    assert params["MAX_STEPS"] is None
    assert params["OUT_DIR"] == "/kaggle/working"


def test_finetune_notebook_uses_the_package_and_never_the_test_set():
    text = notebook_text("02_finetune.ipynb")
    assert "git+{REPO_URL}@{REPO_REF}" in text
    assert 'REPO_URL = "https://github.com/belasoaress/tanque-cheio"' in text
    assert 'os.environ["TANQUE_ROOT"] = REPO_DIR' in text
    assert "finetune.run_finetune(" in text
    assert "local.run_predictions(" in text and 'adapter_id=info["adapter_dir"]' in text
    assert '"datasets/dev.jsonl"' in text
    assert "test.jsonl" not in text
    assert 'UserSecretsClient().get_secret("HF_TOKEN")' in text
    assert "if PUSH_TO_HUB:" in text
    for cell in load_notebook("02_finetune.ipynb")["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def locked_versions() -> dict[str, str]:
    lock = config.resolve_path("uv.lock").read_text(encoding="utf-8")
    return dict(re.findall(r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', lock))


def test_finetune_pins_match_the_lock_used_by_the_cpu_smoke_test():
    text = notebook_text("02_finetune.ipynb")
    pins = dict(re.findall(r'"([a-z]+)==([0-9][^"]*)"', text))
    assert {"transformers", "peft", "trl", "accelerate", "datasets", "bitsandbytes"} <= set(pins)
    locked = locked_versions()
    for name, version in pins.items():
        if name in locked:  # bitsandbytes não está no lock (não roda em CPU)
            assert locked[name] == version, name

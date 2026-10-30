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
    # v2 (M15): kernel novo, para não sobrescrever o notebook do v1
    "finetune-v2": ("belasoares/tanque-cheio-finetune-v2", "02_finetune.ipynb"),
    "test-predict": ("belasoares/tanque-cheio-test-predict", "04_test_predict.ipynb"),
    "large": ("belasoares/tanque-cheio-large", "05_large_baselines.ipynb"),
    # v2 (M17): dois kernels do mesmo notebook, escolhidos por PART
    "final-v2a": ("belasoares/tanque-cheio-final-v2a", "06_final_v2.ipynb"),
    "final-v2b": ("belasoares/tanque-cheio-final-v2b", "06_final_v2.ipynb"),
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
    "04_test_predict.ipynb": {
        "REPO_REF",
        "MODEL_ID",
        "MODEL_REVISION",
        "ADAPTER_ID",
        "ADAPTER_REVISION",
        "MERGE_ADAPTER",
        "DATASET",
        "LIMIT",
        "MAX_NEW_TOKENS",
        "OUT_DIR",
    },
    "05_large_baselines.ipynb": {
        "REPO_REF",
        "MODELS",
        "DATASET",
        "LIMIT",
        "MAX_NEW_TOKENS",
        "RETRY",
        "PART",
        "TIME_BUDGET_32B_H",
        "RESUME_DIR",
        "OUT_DIR",
    },
    "06_final_v2.ipynb": {
        "REPO_REF",
        "MODELS",
        "DATASET",
        "LIMIT",
        "MAX_NEW_TOKENS",
        "RETRY",
        "PART",
        "FT2_ADAPTER_REVISION",
        "TIME_BUDGET_32B_H",
        "RESUME_DIR",
        "OUT_DIR",
    },
    "02_finetune.ipynb": {
        "REPO_REF",
        "CONFIG",
        "MAX_STEPS",
        "EVAL_DATASET",
        "EVAL_LIMIT",
        "MAX_NEW_TOKENS",
        "RETRY",
        "HF_REPO",
        "PUSH_TO_HUB",
        "OUT_DIR",
    },
}
# Notebooks só de inferência: sem 4-bit nem bitsandbytes.
INFERENCE_NOTEBOOKS = ["01_bakeoff.ipynb", "03_predict.ipynb", "04_test_predict.ipynb"]


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
    # v2 (M15): train-v2.toml (sistema ft2-...), previsões no dev2 com 384 tokens e retry
    assert params["CONFIG"] == "configs/train-v2.toml"
    assert params["EVAL_DATASET"] == "datasets/dev2.jsonl" and params["EVAL_LIMIT"] is None
    assert params["MAX_NEW_TOKENS"] == 384 and params["RETRY"] is True  # D31, D32
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
    assert "config.resolve_path(EVAL_DATASET)" in text
    assert "test.jsonl" not in text
    assert "max_new_tokens=MAX_NEW_TOKENS" in text and "retry=RETRY" in text
    assert "large.write_run_infos(" in text and "config_file=CONFIG" in text
    cells = [
        source(c) for c in load_notebook("02_finetune.ipynb")["cells"] if c["cell_type"] == "code"
    ]
    build = next(i for i, c in enumerate(cells) if '"--from-parquet"' in c)
    predict = next(i for i, c in enumerate(cells) if "local.run_predictions(" in c)
    assert build < predict and "coletas-2026-09.parquet" in cells[build]
    assert 'UserSecretsClient().get_secret("HF_TOKEN")' in text
    assert "if PUSH_TO_HUB:" in text
    for cell in load_notebook("02_finetune.ipynb")["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


@pytest.mark.parametrize(
    "name", ["02_finetune.ipynb", "04_test_predict.ipynb", "06_final_v2.ipynb"]
)
def test_single_gpu_before_importing_torch(name):
    # O Kaggle pode dar 2 T4; o SFTTrainer recusa DataParallel, e no teste final os dois
    # sistemas precisam rodar na mesma GPU.
    cells = [source(c) for c in load_notebook(name)["cells"] if c["cell_type"] == "code"]
    pin = next(i for i, c in enumerate(cells) if 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in c)
    first_torch = next(i for i, c in enumerate(cells) if "import torch" in c)
    assert pin < first_torch


@pytest.mark.parametrize(
    "name",
    [
        "02_finetune.ipynb",
        "03_predict.ipynb",
        "04_test_predict.ipynb",
        "05_large_baselines.ipynb",
        "06_final_v2.ipynb",
    ],
)
def test_adapter_notebooks_drop_torchao_and_fail_fast(name):
    # A imagem do Kaggle traz torchao 0.10; o peft recusa carregar adapters com ele.
    cells = [source(c) for c in load_notebook(name)["cells"] if c["cell_type"] == "code"]
    uninstall = next(i for i, c in enumerate(cells) if '"uninstall", "-y", "-q", "torchao"' in c)
    check = next(i for i, c in enumerate(cells) if "is_torchao_available()" in c)
    first_torch = next(i for i, c in enumerate(cells) if "import torch" in c)
    assert uninstall < first_torch <= check


def locked_versions() -> dict[str, str]:
    lock = config.resolve_path("uv.lock").read_text(encoding="utf-8")
    return dict(re.findall(r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', lock))


@pytest.mark.parametrize(
    ("name", "required"),
    [
        (
            "02_finetune.ipynb",
            {"transformers", "peft", "trl", "accelerate", "datasets", "bitsandbytes"},
        ),
        ("04_test_predict.ipynb", {"transformers", "peft", "accelerate"}),
        ("05_large_baselines.ipynb", {"transformers", "peft", "accelerate", "bitsandbytes"}),
        ("06_final_v2.ipynb", {"transformers", "peft", "accelerate", "bitsandbytes"}),
    ],
)
def test_pins_match_the_lock_used_by_the_cpu_smoke_test(name, required):
    text = notebook_text(name)
    pins = dict(re.findall(r'"([a-z]+)==([0-9][^"]*)"', text))
    assert required <= set(pins)
    locked = locked_versions()
    for name, version in pins.items():
        if name in locked:  # bitsandbytes não está no lock (não roda em CPU)
            assert locked[name] == version, name


# --- 04_test_predict (M7) -------------------------------------------------------

DEV_RUN_INFO = "results/predictions/run-info__ft-qwen2.5-coder-1.5b-instruct__dev.json"


def test_test_predict_parameters_defaults():
    params = parameters("04_test_predict.ipynb")
    assert params["REPO_REF"] == "main"
    assert params["DATASET"] == "test" and params["LIMIT"] is None
    assert params["MAX_NEW_TOKENS"] == 256 and params["DTYPE"] == "float16"
    assert params["MERGE_ADAPTER"] is True
    assert params["ADAPTER_ID"] == "belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora"
    assert re.fullmatch(r"[0-9a-f]{40}", params["ADAPTER_REVISION"])
    assert params["OUT_DIR"] == "/kaggle/working"
    # mesmo modelo e mesmo commit do checkpoint usados no fine-tuning (M6)
    dev = json.loads(config.resolve_path(DEV_RUN_INFO).read_text(encoding="utf-8"))
    assert params["MODEL_ID"] == dev["model_id"]
    assert params["MODEL_REVISION"] == dev["model_revision"]
    train = config.resolve_path("configs/train.toml").read_text(encoding="utf-8")
    assert f'"{params["MODEL_ID"]}"' in train


def test_test_predict_runs_base_and_ft_in_one_session():
    text = notebook_text("04_test_predict.ipynb")
    assert text.count("local.run_predictions(") == 1  # um laço sobre os dois sistemas
    assert '{"adapter_id": None, "adapter_revision": None}' in text
    assert '{"adapter_id": ADAPTER_ID, "adapter_revision": ADAPTER_REVISION}' in text
    assert "merge_adapter=MERGE_ADAPTER" in text
    assert "max_new_tokens=MAX_NEW_TOKENS" in text
    assert 'f"datasets/{DATASET}.jsonl"' in text
    assert "run-info__" in text and "repo_commit=REPO_COMMIT" in text
    assert "dev.jsonl" not in text and "PUSH_TO_HUB" not in text


# --- 05_large_baselines (M14) -----------------------------------------------------

LARGE = "05_large_baselines.ipynb"


def exec_parameters(name: str) -> dict:
    """Valores da 1ª célula, executada (MODELS usa TIME_BUDGET_32B_H)."""
    first = next(c for c in load_notebook(name)["cells"] if c["cell_type"] == "code")
    values: dict = {}
    exec(source(first), {}, values)
    return values


def test_large_parameters_defaults():
    params = exec_parameters(LARGE)
    assert params["REPO_REF"] == "main"
    assert params["DATASET"] == "dev2" and params["LIMIT"] is None
    assert params["MAX_NEW_TOKENS"] == 384 and params["RETRY"] is True  # D31, D32
    assert params["PART"] == "all" and params["RESUME_DIR"] is None
    assert params["TIME_BUDGET_32B_H"] == 3 and params["DTYPE"] == "float16"
    assert params["OUT_DIR"] == "/kaggle/working"


def test_large_models_are_pinned_and_ordered():
    models = exec_parameters(LARGE)["MODELS"]
    ids = [(m["model_id"], m.get("adapter_id")) for m in models]
    assert ids == [
        ("Qwen/Qwen2.5-Coder-1.5B-Instruct", None),
        ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora"),
        ("Qwen/Qwen2.5-Coder-7B-Instruct", None),
        ("Qwen/Qwen2.5-Coder-14B-Instruct", None),
        ("Qwen/Qwen2.5-Coder-32B-Instruct", None),  # por último
    ]
    for m in models:
        assert re.fullmatch(r"[0-9a-f]{40}", m["revision"]), m["model_id"]
    # o 1,5B e o adapter são os mesmos do teste final do M7
    test_params = parameters("04_test_predict.ipynb")
    assert models[0]["revision"] == models[1]["revision"] == test_params["MODEL_REVISION"]
    assert models[1]["adapter_revision"] == test_params["ADAPTER_REVISION"]
    assert [m["quantization"] for m in models] == [None, None, "nf4", "nf4", "nf4"]  # D27
    assert [m["gpus"] for m in models] == [1, 1, 1, 1, 2]
    assert [m["part"] for m in models] == ["a", "a", "a", "a", "b"]
    big = models[-1]
    assert big["time_budget_h"] == 3 and set(big["max_memory"]) == {0, 1}
    assert re.fullmatch(r"[0-9a-f]{40}", big["fallback"]["revision"])


def test_large_notebook_does_not_pin_a_single_gpu():
    # o 32B precisa das 2 T4: nada de CUDA_VISIBLE_DEVICES = "0" (regra 0.8 adaptada)
    text = notebook_text(LARGE)
    assert 'os.environ["CUDA_VISIBLE_DEVICES"]' not in text


def test_large_notebook_flow():
    text = notebook_text(LARGE)
    assert "git+{REPO_URL}@{REPO_REF}" in text
    assert 'REPO_URL = "https://github.com/belasoaress/tanque-cheio"' in text
    assert 'os.environ["TANQUE_ROOT"] = REPO_DIR' in text
    assert "test.jsonl" not in text  # o teste fica para o M17
    assert 'f"datasets/{DATASET}.jsonl"' in text
    assert "large.run_spec(" in text and "large.write_run_infos(" in text
    assert "large.select_models(MODELS, PART)" in text
    assert "retry=RETRY" in text and "max_new_tokens=MAX_NEW_TOKENS" in text
    assert "torch.cuda.empty_cache()" in text and "gc.collect()" in text
    assert "repo_commit=REPO_COMMIT" in text
    cells = [source(c) for c in load_notebook(LARGE)["cells"] if c["cell_type"] == "code"]
    build = next(i for i, c in enumerate(cells) if '"--from-parquet"' in c)
    run = next(i for i, c in enumerate(cells) if "large.run_spec(" in c)
    assert build < run and "coletas-2026-09.parquet" in cells[build]
    for cell in load_notebook(LARGE)["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_large_part_split_covers_every_model():
    from tanque.infer import large

    models = exec_parameters(LARGE)["MODELS"]
    a, b = large.select_models(models, "a"), large.select_models(models, "b")
    assert a + b == models
    assert [large.system_of(m) for m in b] == ["base-qwen2.5-coder-32b-instruct"]


# --- 06_final_v2 (M17) -------------------------------------------------------------

FINAL = "06_final_v2.ipynb"
FT2_SYSTEM = "ft2-qwen2.5-coder-1.5b-instruct"
# Commit do adapter v2 publicado no HF (belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2).
FT2_REVISION = "beeb19b5b4331ff048c6be0f03b9d53476d509ab"
# Um valor não preenchido, para provar que a guarda do notebook falha com mensagem clara.
FT2_PLACEHOLDER = "PREENCHER_COM_O_COMMIT_DO_ADAPTER_V2"


def code_cells(name: str) -> list[str]:
    return [source(c) for c in load_notebook(name)["cells"] if c["cell_type"] == "code"]


def test_final_parameters_defaults():
    params = exec_parameters(FINAL)
    assert params["REPO_REF"] == "main"
    assert params["DATASET"] == "test" and params["LIMIT"] is None  # M17: o teste
    assert params["MAX_NEW_TOKENS"] == 384 and params["RETRY"] is True  # D31, D32
    assert params["PART"] == "a" and params["RESUME_DIR"] is None
    assert params["DTYPE"] == "float16" and params["OUT_DIR"] == "/kaggle/working"
    revision = params["FT2_ADAPTER_REVISION"]
    assert revision == FT2_REVISION  # o adapter v2 publicado, fixado por commit


def test_final_systems_reuse_the_m14_checkpoints():
    from tanque.infer import large

    params = exec_parameters(FINAL)
    models = params["MODELS"]
    a, b = large.select_models(models, "a"), large.select_models(models, "b")
    assert a + b == models
    assert [large.system_of(m) for m in a] == [
        "base-qwen2.5-coder-1.5b-instruct",
        "ft-qwen2.5-coder-1.5b-instruct",
        FT2_SYSTEM,
        "base-qwen2.5-coder-7b-instruct",
        "base-qwen2.5-coder-14b-instruct",
    ]
    assert [large.system_of(m) for m in b] == ["base-qwen2.5-coder-32b-instruct"]
    assert all(m["gpus"] == 1 for m in a) and b[0]["gpus"] == 2
    assert [m["quantization"] for m in models] == [None, None, None, "nf4", "nf4", "nf4"]
    # mesmos commits do M14 (e do M7) para os 5 sistemas que já rodaram no dev2
    m14 = {large.system_of(m): m for m in exec_parameters(LARGE)["MODELS"]}
    keys = ("model_id", "revision", "adapter_id", "adapter_revision", "quantization")
    for m in models:
        system = large.system_of(m)
        if system == FT2_SYSTEM:
            continue
        ref = m14[system]
        for key in (*keys, "fallback", "max_memory"):
            assert m.get(key) == ref.get(key), (system, key)
    ft2 = next(m for m in models if large.system_of(m) == FT2_SYSTEM)
    assert ft2["adapter_id"] == "belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2"
    assert ft2["adapter_revision"] == params["FT2_ADAPTER_REVISION"]
    assert ft2["revision"] == m14["ft-qwen2.5-coder-1.5b-instruct"]["revision"]
    v2 = config.resolve_path("configs/train-v2.toml").read_text(encoding="utf-8")
    assert f'revision = "{ft2["revision"]}"' in v2  # o base em que o v2 foi treinado
    assert b[0]["time_budget_h"] == params["TIME_BUDGET_32B_H"]


def _guard_cell() -> str:
    return next(c for c in code_cells(FINAL) if "FT2_ADAPTER_REVISION não foi preenchido" in c)


def _run_guard(monkeypatch, **params) -> str | None:
    import os

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    exec(_guard_cell(), dict(params))
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return value


def test_final_guard_fails_clearly_without_the_ft2_commit(monkeypatch):
    placeholder = FT2_PLACEHOLDER
    with pytest.raises(ValueError, match="FT2_ADAPTER_REVISION não foi preenchido"):
        _run_guard(monkeypatch, PART="a", FT2_ADAPTER_REVISION=placeholder)
    with pytest.raises(ValueError, match="FT2_ADAPTER_REVISION"):
        _run_guard(monkeypatch, PART="a", FT2_ADAPTER_REVISION=None)
    with pytest.raises(ValueError, match="PART"):
        _run_guard(monkeypatch, PART="all", FT2_ADAPTER_REVISION="a" * 40)
    # preenchido: o kernel (a) fica com 1 GPU; o (b) não precisa do adapter nem restringe
    assert _run_guard(monkeypatch, PART="a", FT2_ADAPTER_REVISION="a" * 40) == "0"
    assert _run_guard(monkeypatch, PART="b", FT2_ADAPTER_REVISION=placeholder) is None


def test_final_guard_runs_before_install_and_torch():
    cells = code_cells(FINAL)
    guard = cells.index(_guard_cell())
    install = next(i for i, c in enumerate(cells) if '"pip", "install"' in c)
    first_torch = next(i for i, c in enumerate(cells) if "import torch" in c)
    assert 0 < guard < install < first_torch
    assert 'if PART == "a":\n    os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in cells[guard]


def test_final_notebook_flow():
    text = notebook_text(FINAL)
    assert "git+{REPO_URL}@{REPO_REF}" in text
    assert 'REPO_URL = "https://github.com/belasoaress/tanque-cheio"' in text
    assert 'os.environ["TANQUE_ROOT"] = REPO_DIR' in text
    assert 'f"datasets/{DATASET}.jsonl"' in text and "split=DATASET" in text
    assert "large.select_models(MODELS, PART)" in text
    assert "large.require_pinned(selected)" in text
    assert "large.run_spec(" in text and "large.write_run_infos(" in text
    assert "retry=RETRY" in text and "max_new_tokens=MAX_NEW_TOKENS" in text
    assert "repo_commit=REPO_COMMIT" in text
    assert "torch.cuda.empty_cache()" in text and "gc.collect()" in text
    cells = code_cells(FINAL)
    build = next(i for i, c in enumerate(cells) if '"--from-parquet"' in c)
    run = next(i for i, c in enumerate(cells) if "large.run_spec(" in c)
    assert build < run and "coletas-2026-09.parquet" in cells[build]
    assert '["sql"]' not in text  # o log não mostra SQLs do teste
    for cell in load_notebook(FINAL)["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_final_part_line_is_unique_for_the_powershell_edit():
    # notebooks/README.md troca a linha de PART com .Replace; ela aparece 1 vez no JSON
    raw = (NOTEBOOKS / FINAL).read_text(encoding="utf-8")
    assert raw.count('PART = \\"a\\"') == 1
    revision = exec_parameters(FINAL)["FT2_ADAPTER_REVISION"]
    assert raw.count(f'FT2_ADAPTER_REVISION = \\"{revision}\\"') == 1
    readme = (NOTEBOOKS / "README.md").read_text(encoding="utf-8")
    assert FT2_REVISION in readme and "final-v2a" in readme and "final-v2b" in readme

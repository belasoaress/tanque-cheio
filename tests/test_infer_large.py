"""Modelos maiores (M14): 4 bits, várias GPUs, memória, checkpoint por item e limite de tempo.

Sem GPU e sem bitsandbytes: modelo, tokenizer e gerador falsos. O que exige
transformers (o `BitsAndBytesConfig` e os argumentos de `from_pretrained`) é
pulado sem o grupo `infer`.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tanque.eval.score import read_jsonl
from tanque.infer import large, local

GOOD_SQL = "SELECT AVG(valor_venda) FROM coletas WHERE produto = 'GASOLINA'"
BAD_SQL = "SELECT AVG(preco) FROM coletas"
NEW_KEYS = {"quantization", "gpus", "peak_gpu_mem_gib"}


# --- configuração de 4 bits -------------------------------------------------------


def test_nf4_config_is_double_quant_with_fp16_compute():
    assert local.bnb_config_kwargs("nf4") == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "float16",
    }


def test_unknown_quantization_is_refused():
    with pytest.raises(ValueError):
        local.bnb_config_kwargs("int3")


def test_no_quantization_gives_no_config():
    assert local.make_quantization_config(None) is None


def test_bitsandbytes_config_object():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    cfg = local.make_quantization_config("nf4")
    assert cfg.load_in_4bit is True and cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True
    assert cfg.bnb_4bit_compute_dtype == torch.float16


class _Recorder:
    def __init__(self):
        self.calls = []

    def from_pretrained(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(eval=lambda: None)


@pytest.fixture
def from_pretrained(monkeypatch):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model, tok = _Recorder(), _Recorder()
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", model)
    monkeypatch.setattr(transformers, "AutoTokenizer", tok)
    return model


def test_load_model_fp16_path_is_unchanged(from_pretrained):
    local.load_model("org/m", revision="abc", device="cuda")
    (args, kwargs) = from_pretrained.calls[0]
    assert args == ("org/m",)
    assert set(kwargs) == {"revision", "device_map", "dtype"}  # sem quantização nem max_memory
    assert kwargs["device_map"] == "cuda" and kwargs["revision"] == "abc"


def test_load_model_nf4_on_two_gpus(from_pretrained):
    local.load_model(
        "org/m", quantization="nf4", device="auto", max_memory={0: "13GiB", 1: "14GiB"}
    )
    (_, kwargs) = from_pretrained.calls[0]
    assert kwargs["device_map"] == "auto"
    assert kwargs["max_memory"] == {0: "13GiB", 1: "14GiB"}
    q = kwargs["quantization_config"]
    assert q.load_in_4bit and q.bnb_4bit_quant_type == "nf4" and q.bnb_4bit_use_double_quant


def test_load_model_single_gpu_device_map(from_pretrained):
    local.load_model("org/m", quantization="nf4", device={"": 0})
    assert from_pretrained.calls[0][1]["device_map"] == {"": 0}


def test_adapter_over_quantized_model_is_refused(from_pretrained):
    with pytest.raises(ValueError):
        local.load_model("org/m", "org/adapter", quantization="nf4")


# --- GPUs e memória ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0], [0]),
        (["cuda"], [0]),
        (["cuda:1", 0, 0, "cuda:1"], [0, 1]),
        (["cpu", "disk"], []),
        ([1, "cpu"], [1]),
    ],
)
def test_gpu_indices(values, expected):
    assert local.gpu_indices(values) == expected


def test_model_gpus_from_device_map_or_device():
    split = SimpleNamespace(hf_device_map={"model.embed": 0, "model.layers.40": 1, "lm_head": 1})
    assert local.model_gpus(split) == [0, 1]
    assert local.model_gpus(SimpleNamespace(hf_device_map={"": 0})) == [0]
    assert local.model_gpus(SimpleNamespace(device="cuda:0")) == [0]
    assert local.model_gpus(SimpleNamespace(device="cpu")) == []


def test_peak_is_the_max_over_the_used_gpus_in_gib():
    peaks = {0: 9 * 2**30, 1: int(11.5 * 2**30), 2: 99 * 2**30}
    assert local.peak_gpu_mem_gib([0, 1], peaks.__getitem__) == 11.5
    assert local.peak_gpu_mem_gib([0], peaks.__getitem__) == 9.0
    assert local.peak_gpu_mem_gib([], peaks.__getitem__) is None


@pytest.mark.parametrize(
    ("device", "uses"),
    [("cuda", True), ("cuda:0", True), ("auto", True), ({"": 0}, True), ("cpu", False)],
)
def test_uses_cuda(device, uses):
    assert local._uses_cuda(device) is uses


# --- checkpoint por item ----------------------------------------------------------


def test_read_partial_drops_a_cut_last_line(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"id": "a"}\n{"id": "b"}\n{"id": "c", "sq', encoding="utf-8")
    assert [p["id"] for p in local.read_partial(path)] == ["a", "b"]
    assert local.read_partial(tmp_path / "nada.jsonl") == []


def test_done_records_need_both_files_when_retrying():
    items = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    singles = [{"id": "b"}, {"id": "a"}]
    retried = [{"id": "a", "attempts": 1}]
    s, r = local.done_records(items, singles, retried)
    assert [p["id"] for p in s] == ["a"] and r == [{"id": "a", "attempts": 1}]
    s, r = local.done_records(items, singles, None)
    assert [p["id"] for p in s] == ["a", "b"] and r == []


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


class ScriptedGenerator:
    """1ª tentativa: SQL inválida na pergunta "Quantas coletas"; correção: SQL válida."""

    instances: list[ScriptedGenerator] = []

    def __init__(self, model, tokenizer, max_new_tokens):
        self.prompts: list[str] = []
        ScriptedGenerator.instances.append(self)

    def sync(self) -> None:
        pass

    def generate(self, prompt: str) -> local.Generation:
        self.prompts.append(prompt)
        if "A consulta falhou:" in prompt:
            return local.Generation("SELECT produto, COUNT(*) FROM coletas GROUP BY produto", 9, 9)
        if "Quantas coletas" in prompt:
            return local.Generation(BAD_SQL, 5, 5)
        return local.Generation(GOOD_SQL, 5, 5)


@pytest.fixture
def fake_gpu(monkeypatch):
    """Modelo "em 2 GPUs" com pico de 7 GiB na GPU 0 e 12,25 GiB na GPU 1."""
    ScriptedGenerator.instances = []
    loads = []

    def load_model(*args, **kwargs):
        loads.append(kwargs)
        model = SimpleNamespace(hf_device_map={"embed": 0, "layers": 1})
        return model, FakeTokenizer()

    peaks = {0: 7 * 2**30, 1: int(12.25 * 2**30)}
    monkeypatch.setattr(local, "load_model", load_model)
    monkeypatch.setattr(local, "HFGenerator", ScriptedGenerator)
    monkeypatch.setattr(local, "_uses_cuda", lambda device: device != "cpu")
    monkeypatch.setattr(local, "_reset_peak_memory", lambda: None)
    monkeypatch.setattr(
        local, "_device_name", lambda device: None if device == "cpu" else "Tesla T4"
    )
    monkeypatch.setattr(local, "_free_cuda", lambda: None)
    monkeypatch.setattr(local, "_resolved_revision", lambda model_id, rev: (rev or "sha", None))
    real_peak = local.peak_gpu_mem_gib
    monkeypatch.setattr(
        local, "peak_gpu_mem_gib", lambda idx, max_allocated=None: real_peak(idx, peaks.get)
    )
    return loads


def _run(dataset, out_dir, db, **kwargs):
    return local.run_predictions(
        dataset,
        "org/Modelo-X",
        out_dir=out_dir,
        retry=True,
        db_path=db,
        echo=lambda _: None,
        **kwargs,
    )


def test_run_predictions_records_quantization_gpus_and_peak(
    fake_gpu, tmp_path, m2_toy_db, m2_dataset
):
    out = tmp_path / "out"
    info = _run(
        m2_dataset,
        out,
        m2_toy_db,
        device="auto",
        quantization="nf4",
        max_memory={0: "13GiB", 1: "14GiB"},
    )
    assert fake_gpu[0]["quantization"] == "nf4" and fake_gpu[0]["device"] == "auto"
    assert fake_gpu[0]["max_memory"] == {0: "13GiB", 1: "14GiB"}
    assert info["quantization"] == "nf4" and info["gpus"] == 2 and info["gpu_indices"] == [0, 1]
    assert info["peak_gpu_mem_gib"] == 12.25 and info["hardware"] == "kaggle-t4"
    assert info["max_memory"] == {"0": "13GiB", "1": "14GiB"}
    assert info["completed"] == info["n_items"] == len(read_jsonl(m2_dataset))
    assert info["stopped_reason"] is None and info["resumed"] == 0
    for name in ("base-modelo-x__dev.jsonl", "base-modelo-x+retry__dev.jsonl"):
        rows = read_jsonl(out / name)
        assert len(rows) == info["n_items"]
        for r in rows:
            assert NEW_KEYS <= set(r)
            assert (r["quantization"], r["gpus"], r["peak_gpu_mem_gib"]) == ("nf4", 2, 12.25)
    json.dumps(info)  # o run-info é serializável


def test_run_predictions_on_cpu_records_no_gpu(fake_gpu, tmp_path, m2_toy_db, m2_dataset):
    info = _run(m2_dataset, tmp_path / "out", m2_toy_db, device="cpu", limit=1)
    assert info["gpus"] == 0 and info["peak_gpu_mem_gib"] is None
    assert info["quantization"] is None and info["hardware"] == "cpu"
    row = read_jsonl(tmp_path / "out" / "base-modelo-x__dev-sample1.jsonl")[0]
    assert (row["quantization"], row["gpus"], row["peak_gpu_mem_gib"]) == (None, 0, None)


def test_resume_skips_the_items_already_written(fake_gpu, tmp_path, m2_toy_db, m2_dataset):
    out = tmp_path / "out"
    first = _run(m2_dataset, out, m2_toy_db, device="auto", limit=2)
    assert first["completed"] == 2
    # a sessão "caiu": o single-shot tem 2 itens, o +retry só o 1º e uma linha cortada
    retry_path = out / "base-modelo-x+retry__dev-sample2.jsonl"
    lines = retry_path.read_text(encoding="utf-8").splitlines()
    retry_path.write_text(lines[0] + "\n" + lines[1][:20], encoding="utf-8")
    ScriptedGenerator.instances = []
    info = _run(m2_dataset, out, m2_toy_db, device="auto", limit=2, resume=True)
    assert info["resumed"] == 1 and info["completed"] == 2
    assert len(ScriptedGenerator.instances) == 1
    regenerated = ScriptedGenerator.instances[0].prompts
    assert regenerated and all("Quantas coletas" in p for p in regenerated)  # só o 2º item
    singles = read_jsonl(out / "base-modelo-x__dev-sample2.jsonl")
    retried = read_jsonl(retry_path)
    assert [r["id"] for r in singles] == [r["id"] for r in retried]
    assert len(singles) == 2


def test_resume_with_everything_done_does_not_load_the_model(
    fake_gpu, tmp_path, m2_toy_db, m2_dataset
):
    out = tmp_path / "out"
    _run(m2_dataset, out, m2_toy_db, device="auto", limit=2, quantization="nf4")
    info = _run(m2_dataset, out, m2_toy_db, device="auto", limit=2, resume=True, quantization="nf4")
    assert len(fake_gpu) == 1  # carregou só na 1ª vez
    assert info["completed"] == 2 and info["resumed"] == 2
    assert info["gpus"] == 2 and info["peak_gpu_mem_gib"] == 12.25


def test_without_resume_the_files_start_over(fake_gpu, tmp_path, m2_toy_db, m2_dataset):
    out = tmp_path / "out"
    _run(m2_dataset, out, m2_toy_db, device="auto", limit=2)
    info = _run(m2_dataset, out, m2_toy_db, device="auto", limit=2)
    assert info["resumed"] == 0
    assert len(read_jsonl(out / "base-modelo-x__dev-sample2.jsonl")) == 2


def test_time_budget_stops_and_is_recorded(fake_gpu, tmp_path, m2_toy_db, m2_dataset, monkeypatch):
    clock = iter([0.0, 10.0, 20.0, 5000.0] + [9000.0] * 50)
    monkeypatch.setattr(local.time, "time", lambda: next(clock))
    info = _run(m2_dataset, tmp_path / "out", m2_toy_db, device="auto", time_budget_s=3600)
    assert info["stopped_reason"] == "time_budget"
    assert 0 < info["completed"] < info["n_items"]
    assert info["time_budget_s"] == 3600
    singles = read_jsonl(tmp_path / "out" / "base-modelo-x__dev.jsonl")
    retried = read_jsonl(tmp_path / "out" / "base-modelo-x+retry__dev.jsonl")
    assert len(singles) == len(retried) == info["completed"]


# --- large.py: a lista de modelos do notebook ------------------------------------

SPEC_32B = {
    "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "revision": "r32",
    "quantization": "nf4",
    "gpus": 2,
    "part": "b",
    "time_budget_h": 3,
    "max_memory": {0: "13GiB", 1: "14GiB"},
    "fallback": {"model_id": "org/Qwen2.5-Coder-32B-Instruct-bnb-4bit", "revision": "f32"},
}
SPEC_7B = {
    "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "revision": "r7",
    "quantization": "nf4",
    "gpus": 1,
    "part": "a",
}


def test_device_map_for():
    assert large.device_map_for(SPEC_7B) == {"": 0}
    assert large.device_map_for(SPEC_32B) == "auto"


def test_select_models_by_part():
    models = [SPEC_7B, SPEC_32B]
    assert large.select_models(models, "all") == models
    assert large.select_models(models, "a") == [SPEC_7B]
    assert large.select_models(models, "b") == [SPEC_32B]
    with pytest.raises(ValueError):
        large.select_models(models, "c")


def test_system_name_comes_from_the_requested_model():
    assert large.system_of(SPEC_32B) == "base-qwen2.5-coder-32b-instruct"


def test_choose_source_prefers_the_official_checkpoint_when_it_fits():
    sizes = {
        "Qwen/Qwen2.5-Coder-32B-Instruct": 61 * large.GIB,
        SPEC_32B["fallback"]["model_id"]: 18 * large.GIB,
    }
    pick = large.choose_source(SPEC_32B, 100 * large.GIB, lambda m, r: sizes[m])
    assert pick == (
        {"model_id": SPEC_32B["model_id"], "revision": "r32", "source": "official"},
        None,
    )
    pick, _ = large.choose_source(SPEC_32B, 40 * large.GIB, lambda m, r: sizes[m])
    assert pick == {
        "model_id": SPEC_32B["fallback"]["model_id"],
        "revision": "f32",
        "source": "fallback",
    }
    pick, reason = large.choose_source(SPEC_32B, 10 * large.GIB, lambda m, r: sizes[m])
    assert pick is None and "disco insuficiente" in reason
    pick, _ = large.choose_source(SPEC_7B, 1, lambda m, r: None)  # tamanho desconhecido: tenta
    assert pick["source"] == "official"


def test_checkpoint_bytes_sums_the_safetensors():
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="model-00001.safetensors", size=10),
            SimpleNamespace(rfilename="model-00002.safetensors", size=5),
            SimpleNamespace(rfilename="tokenizer.json", size=99),
        ]
    )
    assert large.checkpoint_bytes("m", "r", lambda *a, **k: info) == 15

    def boom(*a, **k):
        raise OSError("sem rede")

    assert large.checkpoint_bytes("m", "r", boom) is None


def test_clear_hf_cache_deletes_only_the_given_repos():
    deleted = []

    class Strategy:
        expected_freed_size = 123

        def execute(self):
            pass

    class Cache:
        repos = [
            SimpleNamespace(repo_id="a/x", revisions=[SimpleNamespace(commit_hash="h1")]),
            SimpleNamespace(repo_id="b/y", revisions=[SimpleNamespace(commit_hash="h2")]),
        ]

        def delete_revisions(self, *hashes):
            deleted.extend(hashes)
            return Strategy()

    assert large.clear_hf_cache(["a/x"], scan=Cache) == 123
    assert deleted == ["h1"]
    assert large.clear_hf_cache(["z/z"], scan=Cache) == 0


def _run_spec(spec, tmp_path, run, **kwargs):
    defaults = {
        "gpu_count": lambda: 2,
        "free_disk": lambda: 100 * large.GIB,
        "size_of": lambda m, r: 20 * large.GIB,
        "clear_cache": lambda repos: 0,
    }
    defaults.update(kwargs)
    return large.run_spec(
        spec,
        dataset_path=tmp_path / "dev2.jsonl",
        split="dev2",
        out_dir=tmp_path,
        run=run,
        echo=lambda _: None,
        **defaults,
    )


def test_run_spec_passes_the_right_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "_free_cuda", lambda: None)
    calls = []

    def run(dataset, model_id, **kwargs):
        calls.append((model_id, kwargs))
        return {"system": kwargs["system"], "path": str(tmp_path / "x__dev2.jsonl")}

    cleared = []
    info = _run_spec(SPEC_32B, tmp_path, run, clear_cache=lambda repos: cleared.extend(repos) or 0)
    model_id, kwargs = calls[0]
    assert model_id == SPEC_32B["model_id"]
    assert kwargs["system"] == "base-qwen2.5-coder-32b-instruct"
    assert kwargs["device"] == "auto" and kwargs["quantization"] == "nf4"
    assert kwargs["max_memory"] == {0: "13GiB", 1: "14GiB"}
    assert kwargs["time_budget_s"] == 3 * 3600 and kwargs["resume"] is True
    assert kwargs["retry"] is True and kwargs["max_new_tokens"] == 384
    assert kwargs["revision"] == "r32"
    assert info["skipped"] is False and info["model_source"] == "official"
    assert cleared == [SPEC_32B["model_id"]]


def test_run_spec_uses_the_fallback_when_the_disk_is_short(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "_free_cuda", lambda: None)
    sizes = {SPEC_32B["model_id"]: 61 * large.GIB, SPEC_32B["fallback"]["model_id"]: 18 * large.GIB}
    calls = []

    def run(dataset, model_id, **kwargs):
        calls.append((model_id, kwargs["revision"], kwargs["system"]))
        return {"system": kwargs["system"], "path": "p"}

    info = _run_spec(
        SPEC_32B, tmp_path, run, free_disk=lambda: 50 * large.GIB, size_of=lambda m, r: sizes[m]
    )
    assert calls == [(SPEC_32B["fallback"]["model_id"], "f32", "base-qwen2.5-coder-32b-instruct")]
    assert info["model_source"] == "fallback"
    assert info["requested_model_id"] == SPEC_32B["model_id"]


def test_run_spec_skips_without_enough_gpus(tmp_path):
    def run(*a, **k):
        raise AssertionError("não deveria rodar")

    info = _run_spec(SPEC_32B, tmp_path, run, gpu_count=lambda: 1)
    assert info["skipped"] is True and "2 GPU" in info["skip_reason"]
    assert info["path"].endswith("base-qwen2.5-coder-32b-instruct__dev2.jsonl")


def test_run_spec_records_an_oom_and_moves_on(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "_free_cuda", lambda: None)

    def run(*a, **k):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    info = _run_spec(SPEC_32B, tmp_path, run)
    assert info["skipped"] is True and "out of memory" in info["skip_reason"]
    paths = large.write_run_infos(info, tmp_path, repo_commit="abc")
    names = sorted(p.name for p in paths)
    assert names == [
        "run-info__base-qwen2.5-coder-32b-instruct+retry__dev2.json",
        "run-info__base-qwen2.5-coder-32b-instruct__dev2.json",
    ]
    saved = json.loads(paths[0].read_text(encoding="utf-8"))
    assert saved["skipped"] is True and saved["repo_commit"] == "abc"


def test_write_run_infos_for_a_finished_run(tmp_path):
    info = {
        "system": "base-m",
        "path": str(tmp_path / "base-m__dev2.jsonl"),
        "summary": {"n": 2},
        "retry_system": "base-m+retry",
        "retry_path": str(tmp_path / "base-m+retry__dev2.jsonl"),
        "retry_summary": {"n": 2, "retried": 1},
    }
    paths = large.write_run_infos(info, tmp_path)
    retry = json.loads(paths[1].read_text(encoding="utf-8"))
    assert paths[0].name == "run-info__base-m__dev2.json"
    assert retry["system"] == "base-m+retry" and retry["summary"]["retried"] == 1


def test_spec_can_override_device_and_dtype_for_the_cpu_rehearsal(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "_free_cuda", lambda: None)
    calls = []

    def run(dataset, model_id, **kwargs):
        calls.append(kwargs)
        return {"system": kwargs["system"], "path": "p"}

    spec = {"model_id": "org/tiny", "gpus": 0, "device": "cpu", "dtype": "float32"}
    spec["time_budget_h"] = 0  # zero é um limite (já estourado), não "sem limite"
    _run_spec(spec, tmp_path, run, gpu_count=lambda: 0)
    assert calls[0]["device"] == "cpu" and calls[0]["dtype"] == "float32"
    assert calls[0]["time_budget_s"] == 0.0


# --- M17: teste final v2 (06_final_v2.ipynb) ---------------------------------------

SPEC_FT2 = {
    "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
    "adapter_id": "belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2",
    "adapter_revision": "a" * 40,
    "system": "ft2-qwen2.5-coder-1.5b-instruct",
    "gpus": 1,
    "part": "a",
}


def test_spec_can_name_its_system_ft2(tmp_path, monkeypatch):
    # o adapter v2 daria `ft-...` pelo nome; o sistema da v2 é `ft2-...` (5.3 v2)
    assert large.system_of(SPEC_FT2) == "ft2-qwen2.5-coder-1.5b-instruct"
    monkeypatch.setattr(local, "_free_cuda", lambda: None)
    calls = []

    def run(dataset, model_id, **kwargs):
        calls.append(kwargs)
        return {"system": kwargs["system"], "path": "p"}

    _run_spec(SPEC_FT2, tmp_path, run)
    assert calls[0]["system"] == "ft2-qwen2.5-coder-1.5b-instruct"
    assert calls[0]["adapter_revision"] == "a" * 40
    skipped = _run_spec(SPEC_FT2, tmp_path, run, gpu_count=lambda: 0)
    assert skipped["system"] == "ft2-qwen2.5-coder-1.5b-instruct"
    assert skipped["path"].endswith("ft2-qwen2.5-coder-1.5b-instruct__dev2.jsonl")


def test_unpinned_lists_placeholders_and_missing_commits():
    assert large.unpinned([SPEC_7B | {"revision": "c" * 40}, SPEC_FT2]) == []
    placeholder = SPEC_FT2 | {"adapter_revision": "PREENCHER"}
    problems = large.unpinned([placeholder, SPEC_7B | {"revision": None}])
    assert len(problems) == 2
    assert "ft2-qwen2.5-coder-1.5b-instruct" in problems[0] and "adapter_revision" in problems[0]
    assert "'PREENCHER'" in problems[0]
    assert "base-qwen2.5-coder-7b-instruct" in problems[1] and "revision" in problems[1]
    with pytest.raises(ValueError, match="commit fixado"):
        large.require_pinned([placeholder])
    large.require_pinned([SPEC_FT2])  # tudo fixado: não levanta

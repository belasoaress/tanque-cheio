"""Space Gradio do tanque-cheio (M19, D36): pergunta em português -> SQL -> tabela.

Ao iniciar:

1. baixa do GitHub, no commit fixado em `tanque_demo.TANQUE_COMMIT`, a configuração,
   o manifest e o snapshot Parquet (2,4 MB, SHA-256 conferido) e monta o DuckDB com
   o mesmo código de `tanque data build --from-parquet`;
2. baixa o GGUF Q4_K_M do HF (revisão em `GGUF_REVISION`) e o carrega com o
   llama-cpp-python (contexto 2048, 2 threads);
3. aquece o cache do prompt (o prefixo do system é igual para todas as perguntas).

Cada pergunta passa pelo prompt único, pelo guardrail, pela execução só leitura e,
se a SQL falhar, por 1 nova tentativa (D32). Fila com 1 worker. Nenhuma pergunta é
gravada.

Variáveis de ambiente (opcionais): `GGUF_REVISION`, `GGUF_REPO`, `GGUF_FILE`,
`GGUF_PATH` (um arquivo local, para ensaio) e `TANQUE_DATA_DIR`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gradio as gr
import tanque_demo as td

HERE = Path(__file__).resolve().parent

INTRO = """
# tanque-cheio · pergunte sobre o preço dos combustíveis

Escreva uma pergunta em português sobre os preços de combustíveis coletados pela ANP.
Um modelo de 1,5B parâmetros (Qwen2.5-Coder-1.5B-Instruct ajustado com QLoRA, aqui
quantizado em 4 bits e rodando em CPU) escreve a SQL, que roda num banco DuckDB só de
leitura. No teste do projeto, ele acertou 80,7% das perguntas, empatado com o
Qwen2.5-Coder-32B sem ajuste.
"""

ABOUT = """
- Modelo: [`belasoares/tanque-cheio-qwen2.5-coder-1.5b-gguf`](https://huggingface.co/belasoares/tanque-cheio-qwen2.5-coder-1.5b-gguf)
  (adapter v2 [`belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2`](https://huggingface.co/belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2)
  fundido no Qwen2.5-Coder-1.5B-Instruct e convertido para GGUF Q4_K_M).
- Mesmo prompt, mesmo guardrail e mesma nova tentativa da avaliação do projeto. A
  primeira resposta pode levar mais tempo; o CPU grátis atende uma pergunta por vez.
- A SQL pode estar errada: confira a consulta antes de usar o número.
- Dados: série histórica de preços de combustíveis da ANP, com atribuição à ANP.
  O banco não tem endereço dos postos. As perguntas não são gravadas.
- Código e avaliação: <https://github.com/belasoaress/tanque-cheio>
"""


def load_runtime() -> tuple[Callable[[list[dict]], Any], Callable[[str], Any]]:
    """(gerador, executor) reais: banco do snapshot e GGUF com o llama-cpp-python."""
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    root = td.fetch_root(Path(os.environ.get("TANQUE_DATA_DIR", HERE / "runtime")))
    db_path = td.build_db(root)

    from tanque import prompting
    from tanque.sql import execute

    model_path = os.environ.get("GGUF_PATH") or hf_hub_download(
        repo_id=os.environ.get("GGUF_REPO", td.GGUF_REPO),
        filename=os.environ.get("GGUF_FILE", td.GGUF_FILE),
        revision=os.environ.get("GGUF_REVISION", td.DEFAULT_GGUF_REVISION),
    )
    llm = Llama(
        model_path=model_path,
        n_ctx=td.N_CTX,
        n_threads=td.N_THREADS,
        n_threads_batch=td.N_THREADS,
        seed=0,
        verbose=False,
    )
    # Aquecimento: avalia o system (igual para todas as perguntas) e deixa no cache.
    td.llama_generator(llm, max_tokens=1)(prompting.build_prompt(td.EXAMPLES[0]))

    def run(sql: str) -> Any:
        return execute.run(sql, timeout_s=td.TIMEOUT_S, max_rows=td.EXEC_MAX_ROWS, db_path=db_path)

    return td.llama_generator(llm), run


def build_ui(generate: Callable[[list[dict]], Any], execute: Callable[[str], Any]) -> gr.Blocks:
    """Interface: pergunta, exemplos, SQL, tabela, status e aviso da data de referência."""

    def on_submit(question: str):
        return td.respond(question, generate, execute)

    with gr.Blocks(title="tanque-cheio", analytics_enabled=False) as ui:
        gr.Markdown(INTRO)
        gr.Markdown(f"**Aviso:** {td.DISCLAIMER}")
        with gr.Row():
            question = gr.Textbox(
                label="Pergunta",
                placeholder=td.EXAMPLES[0],
                max_length=td.MAX_QUESTION_CHARS,
                lines=1,
                scale=4,
            )
            ask = gr.Button("Perguntar", variant="primary", scale=1)
        # Saídas criadas antes dos exemplos (que as preenchem) e desenhadas depois deles.
        status = gr.Markdown(render=False)
        sql = gr.Code(label="SQL gerada", language="sql", interactive=False, render=False)
        table = gr.Dataframe(
            label=f"Resultado (até {td.MAX_TABLE_ROWS} linhas)",
            value=td.table_value(td.Answer(question="")),
            interactive=False,
            wrap=True,
            render=False,
        )
        first = gr.Markdown(render=False)
        outputs = [sql, table, status, first]
        gr.Examples(
            examples=[[q] for q in td.EXAMPLES],
            inputs=[question],
            outputs=outputs,
            fn=on_submit,
            run_on_click=True,  # um clique no exemplo já responde
            cache_examples=False,
            label="Exemplos (clique para perguntar)",
        )
        for component in (status, sql, table, first):
            component.render()
        with gr.Accordion("Sobre a demo", open=False):
            gr.Markdown(ABOUT)

        gr.on(
            triggers=[ask.click, question.submit],
            fn=on_submit,
            inputs=[question],
            outputs=outputs,
            api_name="perguntar",
            concurrency_limit=1,
        )
    # 1 worker: o llama.cpp usa as 2 vCPUs numa pergunta por vez.
    ui.queue(default_concurrency_limit=1, max_size=20)
    return ui


if __name__ == "__main__":
    build_ui(*load_runtime()).launch()

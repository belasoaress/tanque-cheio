<!--
NOTA PARA A ISABELA (não faz parte do artigo; apague antes de publicar)

Sugestões de título:
1. Um modelo de 1,5B empatou com um de 32B. Custou US$ 0.
2. Fine-tuning de graça: 1,5B contra 32B em text-to-SQL em português
3. O que aprendi ajustando um modelo pequeno para responder perguntas sobre combustível

Imagem: results/ex-vs-cost.png (EX no teste contra custo por 1.000 perguntas, com IC 95%).
Mostra numa imagem só o 1,5B ajustado no canto "bom e barato" e o 32B no "bom e caro".

Frases para conferir ou reescrever com a sua voz:
- O gancho ("Ajustei um modelo de 1,5 bilhão de parâmetros..."): é a frase que mais
  aparece na prévia do LinkedIn; vale deixar do seu jeito.
- "revisadas à mão, uma a uma": o rascunho das perguntas teve ajuda de um agente de
  código e a revisão foi sua (README, Limitações). Ajuste se quiser detalhar.
- A lição do dev2 ("Eu caí nisso..."): confira se é assim que você quer contar o erro.
- "Leio como empate, não como vitória": é a leitura que o README faz; mantenha o tom
  que você prefere para não soar como exagero.
- O fechamento (o que vem depois): troque pelo próximo passo real, se for outro.

Todos os números foram conferidos contra results/report.md (o teste
tests/test_readme.py confere os percentuais deste arquivo). Preencha o link da demo.
-->

# Um modelo de 1,5B empatou com um de 32B. Custou US$ 0.

Ajustei um modelo de 1,5 bilhão de parâmetros para transformar perguntas em português em
SQL. No meu conjunto de teste, ele acerta 80,7%. O Qwen2.5-Coder-32B, 21 vezes maior e
sem ajuste, acerta 78,0%. A diferença não é significativa (p = 0,63): é um empate.

Só que o pequeno faz isso com 1 GPU em vez de 2, cerca de 7× menos latência (4,3 s
contra 29,1 s de mediana) e cerca de 15× menos custo por pergunta. E o treino rodou na
GPU gratuita do Kaggle. Gasto com API: zero.

## O problema

A ANP publica os preços de combustíveis coletados em postos do Brasil inteiro. Os dados
são abertos, mas responder "em quais capitais o etanol compensou em todos os meses de
2025?" exige saber SQL e conhecer as tabelas.

O tanque-cheio recebe a pergunta em português, gera a SQL, executa e devolve a tabela,
com a SQL visível para quem quiser conferir.

Por que um modelo pequeno? Porque um modelo grande custa caro por pergunta, demora e
pede GPU de verdade. Um de 1,5B roda numa GPU barata e, quantizado, até em CPU. A
pergunta do projeto foi: até onde um modelo pequeno chega se eu ajustar para uma tarefa
só?

## Como fiz

**Dados.** 24 meses de coletas da ANP (outubro de 2024 a setembro de 2026), cerca de
1,65 milhão de registros, num banco DuckDB com três tabelas. O snapshot fica no
repositório, então qualquer pessoa reconstrói o mesmo banco.

**Um prompt só.** Todos os modelos recebem o mesmo prompt: schema, data de referência e
as convenções de resposta. Assim a comparação é justa.

**Execução segura.** Toda SQL passa por um guardrail antes de rodar: só SELECT, só as
tabelas conhecidas, banco somente leitura, timeout e limite de linhas.

**Avaliação separada do treino.** As perguntas de desenvolvimento e as 150 de teste
foram revisadas à mão, uma a uma. Um filtro no código tira do treino tudo que se pareça
com elas. A métrica é a acurácia de execução: a SQL gerada roda, e o resultado tem de
bater com o da SQL de referência.

**Treino sintético, sem API.** Em vez de pagar um LLM para gerar exemplos, escrevi
templates: cada tipo de pergunta tem uma SQL e várias redações, preenchidas com valores
reais do banco. Cada par é validado por execução. Foram 10.448 pares, a US$ 0.

**QLoRA na T4 do Kaggle.** O Qwen2.5-Coder-1.5B, ajustado em 4 bits, numa GPU gratuita.
Cerca de 5,3 horas de treino na versão final.

## O resultado

Sem ajuste, o 1,5B acerta 12,0% do teste. Ajustado, 80,7%.

Comparei com os irmãos maiores da mesma família, sem ajuste e com o mesmo prompt:

- 7B: 40,7%
- 14B: 60,0%
- 32B: 78,0%
- 1,5B ajustado: 80,7%

Contra o 14B, a vantagem é de 20,7 pontos (p < 0,001). Contra o 32B, empate.

O ganho maior está nas perguntas difíceis, as que pedem ranking, comparação entre
períodos ou "todos os meses": o ajustado acerta 62,5%, contra 37,5% do 32B. Nas fáceis,
o 32B vai melhor (96,0% contra 92,0%).

## O erro que mais me ensinou

Fiz o projeto em duas versões. Na segunda, olhei os erros do modelo no conjunto de
desenvolvimento e escrevi templates novos para cobrir exatamente aqueles padrões.

No desenvolvimento, a acurácia foi de 74,3% (versão 1) para 96,5% (versão 2). No teste, ficou em 80,7%.

A queda de 16 pontos é sobreajuste. Não às perguntas, que o filtro barrava, mas ao estilo
dos erros. Eu caí nisso mesmo sabendo do risco, e é exatamente por isso que o teste
ficou trancado até o fim. Se eu tivesse decidido olhando o teste, não teria um número
honesto para mostrar.

A segunda lição veio da métrica. A acurácia de execução exige as mesmas colunas da
referência. Defini antes dos resultados uma versão tolerante, que aceita colunas a mais.
Por ela, o ajustado vai de 80,7% para 85,3%. Ou seja, parte dos "erros" eram respostas
certas no formato errado: o valor estava lá, com uma coluna extra. Dos 29 erros do
modelo final, 10 são de forma do resultado.

## Limitações

Os modelos grandes rodaram em 4 bits, porque é o que cabe nas GPUs gratuitas; em
precisão cheia, o 32B talvez passasse do ajustado. Com 150 perguntas, o intervalo de
confiança é de cerca de ±7 pontos, então diferenças pequenas não querem dizer muito. E o
treino é sintético: perguntas fora do estilo dos templates tendem a sair piores.

Mesmo assim, a lição para mim é clara. Para uma tarefa bem delimitada, um modelo pequeno
ajustado com dados baratos chega perto de um modelo muito maior, por uma fração do custo.

## Links

- Código, dados e relatório completo: https://github.com/belasoaress/tanque-cheio
- Adapter v2 no Hugging Face: https://huggingface.co/belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2
- Notebook do fine-tuning (v1) no Kaggle: https://www.kaggle.com/code/belasoares/tanque-cheio-finetune
- Demo: [link da demo]

Todos os números saem das previsões salvas no repositório e são recalculados pelo CI.

#MachineLearning #LLM #TextToSQL #DadosAbertos

# Plano de Ação: Pré-processamento Diabetes (Equipe de 3 pessoas)

Para que o trabalho flua de forma eficiente entre 3 pessoas, a melhor estratégia é dividir o trabalho em etapas dependentes (onde um precisa do trabalho do outro) e etapas paralelas (onde vocês podem trabalhar ao mesmo tempo).

Uma ótima forma de fazer isso é utilizar o **Google Colab** ou um repositório no **GitHub** para compartilhamento de código.

## 1. Divisão de Tarefas

### 👤 Pessoa 1: Seleção e Limpeza Básica (Foundation)
Esta pessoa é responsável por entregar a base de dados "sem buracos".
*   **O que foi feito:**
    *   Foram removidas as colunas inviáveis e com excesso de zeros: `Insulin`, `SkinThickness` e `BloodPressure`.
    *   Foi aplicada a imputação (preenchimento) dos zeros nas colunas vitais `Glucose` e `BMI`, utilizando valores aleatórios para preservar a variância original.
    *   O código foi refatorado utilizando uma função centralizadora `pre_processar(df)` para garantir que a mesma regra seja aplicada no treino e no teste.
*   **Status:** ✅ **Concluído**. A base de dados limpa está pronta e acoplada no pipeline.

### 👤 Pessoa 2: Enriquecimento e Análise de Outliers (Feature Engineering)
Esta pessoa vai focar na inteligência dos dados, criando novas variáveis a partir da base limpa pela Pessoa 1.
*   **O que precisa fazer agora:**
    *   Trabalhar escrevendo o seu código dentro do bloco `--- [PESSOA 2] ---` na função `pre_processar()` no script `diabetes_csv.py`.
    *   Criar novas colunas (Enriquecimento). Por exemplo: transformar a Idade (`Age`) em faixas etárias categóricas (Jovem, Adulto, Idoso) ou o `BMI` em níveis de obesidade.
    *   Avaliar se essas novas colunas melhoram o resultado do KNN e se vale a pena mantê-las.
*   **Status:** 🚧 **Em andamento**. O código desta pessoa vai se beneficiar da arquitetura já criada e enriquecer o modelo para a Pessoa 3.

### 👤 Pessoa 3: Transformação e Pipeline Final (Integração)
Esta pessoa é responsável por "juntar tudo", adequar os dados para o algoritmo KNN e enviar para o servidor.
*   **O que faz:**
    *   Aplica o escalonamento (Scaling) como o `StandardScaler` ou `MinMaxScaler` nos dados numéricos.
    *   Aplica a transformação de texto para número (`One-Hot Encoding` / Variáveis Dummy) nas categorias criadas pela Pessoa 2.
    *   Consolida o script (`diabetes_csv.py`), roda o KNN e envia o resultado com a `DEV_KEY` do grupo.
*   **Status:** *Trabalho Final*. Depende das transformações das Pessoas 1 e 2 para rodar o modelo definitivo, mas já pode ir escrevendo a estrutura do `Pipeline` do *scikit-learn* (ou as funções base) em paralelo.

---

## 2. Arquitetura da Atividade

A arquitetura do projeto é bem simples e linear. O fluxo de dados segue a premissa de um torneio de Machine Learning (estilo Kaggle), onde você treina com um arquivo e é avaliado em outro.

### Os Arquivos
1.  `diabetes_dataset.csv` (ou `.xlsx`): **Dados de Treino**. Este arquivo possui as features (colunas) e a coluna resposta `Outcome` (se a pessoa tem ou não diabetes). Vocês usarão este arquivo para aprender e calibrar o modelo (etapas de pré-processamento entram todas aqui primeiro).
2.  `diabetes_app.csv` (ou `.xlsx`): **Dados de Teste/Aplicação**. Este arquivo tem as exatas mesmas colunas do dataset de treino, **exceto** a coluna `Outcome`. É um "arquivo cego". O objetivo é o modelo prever o diabetes desses pacientes.
3.  `diabetes_csv.py` (ou `_xlsx`): **O Motor Principal**. É o único script Python que você realmente precisa rodar.

### O que precisa ser rodado e em que ordem?
Vocês só precisam rodar **um único arquivo**: o `diabetes_csv.py` (se escolherem usar a versão CSV). 
O fluxo que ocorre dentro dele (que vocês irão modificar) é o seguinte:

1.  **Leitura do Treino:** Lê `diabetes_dataset.csv`.
2.  **Preparação:** Separa o que é característica (`X`) e o que é o alvo (`y`).
    > *🚨 É aqui no meio que entrarão todos os códigos de pré-processamento que vocês vão fazer (seleção, limpeza, enriquecimento, transformação).*
3.  **Treinamento:** Cria o modelo KNN e treina (`neigh.fit(X, y)`).
4.  **Leitura do Teste:** Lê `diabetes_app.csv`.
    > *🚨 Toda a transformação de dados que você aplicou no `X` de treino, precisa obrigatoriamente ser aplicada na base `diabetes_app` antes de fazer a previsão.*
5.  **Previsão:** Pede ao KNN treinado para adivinhar o *Outcome* do arquivo `diabetes_app` (`neigh.predict(data_app)`).
6.  **Submissão:** O script empacota as suas previsões em um arquivo JSON, anexa a sua `DEV_KEY` e faz uma requisição POST enviando os resultados para o servidor do professor (linha 49). O servidor responde com o seu score.

**Resumo da Ópera:** Não há vários arquivos para executar. Todo o trabalho do grupo se concentra em adicionar a inteligência do pré-processamento dentro de `diabetes_csv.py` (entre as linhas 17 e 28), garantir que as mesmas regras sejam aplicadas nos dados de teste (`diabetes_app.csv`), adicionar a sua Chave e executar o script.

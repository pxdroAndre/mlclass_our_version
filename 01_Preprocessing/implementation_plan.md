# Plano de Ação: Pré-processamento Diabetes (Equipe de 3 pessoas)

Para que o trabalho flua de forma eficiente entre 3 pessoas, a melhor estratégia é dividir o trabalho em etapas dependentes (onde um precisa do trabalho do outro) e etapas paralelas (onde vocês podem trabalhar ao mesmo tempo).

Uma ótima forma de fazer isso é utilizar o **Google Colab** ou um repositório no **GitHub** para compartilhamento de código.

## 1. Divisão de Tarefas

### 👤 Pessoa 1: Seleção e Limpeza Básica (Foundation)
Esta pessoa será responsável por entregar a base de dados "sem buracos".
*   **O que faz:**
    *   Carrega os dados e analisa os valores nulos (`NaN`) e zeros.
    *   Toma a decisão de remover colunas inviáveis (como `Insulin` e `SkinThickness`).
    *   Aplica a imputação (preenchimento) de nulos na `Glucose`, `BMI` e `BloodPressure` usando média ou mediana.
*   **Status:** *Trabalho Inicial*. Não depende de ninguém.

### 👤 Pessoa 2: Enriquecimento e Análise de Outliers (Feature Engineering)
Esta pessoa vai focar na inteligência dos dados e lidar com valores estranhos.
*   **O que faz:**
    *   Procura e decide o que fazer com *outliers* (ex: pressão sanguínea muito alta ou zerada).
    *   Cria novas colunas (Enriquecimento), como transformar a Idade (`Age`) e o `BMI` em variáveis categóricas (faixas de idade, níveis de obesidade).
*   **Status:** *Parcialmente Paralelo*. Pode começar a analisar o CSV original em paralelo para descobrir os agrupamentos, mas a integração do seu código depende da base limpa da Pessoa 1.

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

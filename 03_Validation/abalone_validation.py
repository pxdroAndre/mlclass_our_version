#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade 03 - Avaliação de classificadores.

Validação completa do modelo ANTES do envio (o servidor aceita apenas 1 envio a cada 12h).
Este script NUNCA envia nada ao servidor; o envio fica em abalone_submit.py.

Metodologia:
  1. Holdout estratificado 80/20: os 20% ficam guardados até o final (simulam o servidor);
  2. Comparação de modelos com validação cruzada repetida (10 folds x 3 repetições)
     apenas nos 80% de desenvolvimento, todos com os MESMOS folds (comparação pareada);
  3. Ajuste de hiperparâmetros do SVM com GridSearchCV;
  4. Teste t pareado corrigido (Nadeau & Bengio) + correção de Holm para comparações
     múltiplas. Os valores de p são EXPLORATÓRIOS: o modelo de referência foi escolhido
     nesses mesmos folds, o que favorece ele na comparação;
  5. Avaliação no holdout: acurácia, F1 por classe e matriz de confusão.
     Observação: o holdout foi consultado duas vezes durante o desenvolvimento
     (C=3/gamma=0.1 -> 67.3%; C=1/gamma='scale' -> 67.9%). A troca de parâmetros
     veio do GridSearch, não do holdout, mas ele já não é um teste estritamente intocado.

Resultado no servidor (1 envio): 67.18% de acurácia.

@updated_by: Equipe MachineLerdos
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from abalone_model import (FINAL_PARAMS, RANDOM_STATE, carregar_treino,
                           modelo_final, montar_pipeline)

warnings.filterwarnings('ignore')

N_SPLITS, N_REPEATS = 10, 3

# =====================================================================
# 1. DADOS E HOLDOUT
# =====================================================================
print('\n - Lendo dados de treino (com limpeza de height inválido)')
X, y = carregar_treino()
print(f'   {len(X)} exemplos | classes: {y.value_counts().sort_index().to_dict()}')

X_dev, X_hold, y_dev, y_hold = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(f'   Desenvolvimento: {len(X_dev)} | Holdout (reservado): {len(X_hold)}')

cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)


def avaliar(modelo):
    return cross_val_score(modelo, X_dev, y_dev, cv=cv, n_jobs=-1)


# =====================================================================
# 2. COMPARAÇÃO DE MODELOS (mesmos folds para todos)
# =====================================================================
candidatos = {
    'Baseline (classe majoritária)': montar_pipeline(DummyClassifier(strategy='most_frequent'), False),
    'k-NN k=15 (originais)':         montar_pipeline(KNeighborsClassifier(15), False),
    'k-NN k=15 (+features)':         montar_pipeline(KNeighborsClassifier(15)),
    'Reg. Logística (originais)':    montar_pipeline(LogisticRegression(max_iter=5000, C=10), False),
    'Reg. Logística (+features)':    montar_pipeline(LogisticRegression(max_iter=5000, C=10)),
    'Random Forest (+features)':     montar_pipeline(RandomForestClassifier(500, min_samples_leaf=5,
                                                                           random_state=RANDOM_STATE)),
    'SVM RBF (originais)':           montar_pipeline(SVC(random_state=RANDOM_STATE, **FINAL_PARAMS), False),
    'SVM RBF (+features)':           modelo_final(),
    'Voting SVM+LogReg (+features)': VotingClassifier([
        ('svc', montar_pipeline(SVC(probability=True, random_state=RANDOM_STATE, **FINAL_PARAMS))),
        ('lr', montar_pipeline(LogisticRegression(max_iter=5000, C=10))),
    ], voting='soft'),
}

print(f'\n - Validação cruzada repetida ({N_SPLITS} folds x {N_REPEATS} repetições) no desenvolvimento')
scores = {}
for nome, modelo in candidatos.items():
    scores[nome] = avaliar(modelo)
    print(f'   {nome:32s} {scores[nome].mean():.4f} ± {scores[nome].std():.4f}', flush=True)

# =====================================================================
# 3. AJUSTE DE HIPERPARÂMETROS DO SVM
# =====================================================================
print('\n - GridSearchCV do SVM RBF (+features)')
grid = GridSearchCV(
    modelo_final(),
    param_grid={'classifier__C': [1, 3, 10, 30], 'classifier__gamma': ['scale', 0.03, 0.1, 0.3]},
    cv=cv, n_jobs=-1,
)
grid.fit(X_dev, y_dev)
res = pd.DataFrame(grid.cv_results_).sort_values('rank_test_score')
for _, r in res.head(5).iterrows():
    print(f'   C={r.param_classifier__C!s:>3} gamma={r.param_classifier__gamma!s:>5}  '
          f'{r.mean_test_score:.4f} ± {r.std_test_score:.4f}')
melhores = {k.replace('classifier__', ''): v for k, v in grid.best_params_.items()}
print(f'   Melhor: {melhores}  |  FINAL_PARAMS em abalone_model.py: {FINAL_PARAMS}')
if melhores != FINAL_PARAMS:
    print('   AVISO: atualize FINAL_PARAMS em abalone_model.py antes de enviar!')


# =====================================================================
# 4. TESTE ESTATÍSTICO PAREADO
# =====================================================================
def teste_t_corrigido(a, b):
    """
    Teste t pareado corrigido de Nadeau & Bengio (2003). Na validação cruzada os
    conjuntos de treino se sobrepõem, então o teste t comum subestima a variância
    e acha diferenças "significativas" demais. A correção infla a variância pelo
    termo n_teste/n_treino.
    """
    d = a - b
    k = len(d)
    razao = 1 / (N_SPLITS - 1)  # n_teste / n_treino em k-fold
    t = d.mean() / np.sqrt((1 / k + razao) * d.var(ddof=1))
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return d.mean(), p


def holm(pvals):
    """Correção de Holm-Bonferroni: controla o erro ao fazer várias comparações ao mesmo tempo."""
    ordem = np.argsort(pvals)
    m = len(pvals)
    ajustados = np.empty(m)
    atual = 0.0
    for i, idx in enumerate(ordem):
        atual = max(atual, min(1.0, (m - i) * pvals[idx]))
        ajustados[idx] = atual
    return ajustados


print('\n - Teste t pareado corrigido (SVM RBF +features vs. outros) - valores EXPLORATÓRIOS')
ref = scores['SVM RBF (+features)']
outros = [n for n in scores if n != 'SVM RBF (+features)']
resultados = [teste_t_corrigido(ref, scores[n]) for n in outros]
p_holm = holm(np.array([p for _, p in resultados]))
for nome, (diff, p), ph in zip(outros, resultados, p_holm):
    sig = 'diferença detectada' if ph < 0.05 else 'sem evidência de diferença'
    print(f'   vs {nome:32s} diferença {diff:+.4f}  p={p:.3f}  p_holm={ph:.3f}  ({sig})')

# =====================================================================
# 5. AVALIAÇÃO NO HOLDOUT
# =====================================================================
print('\n - Avaliação no holdout (com o modelo já escolhido; ver observação no cabeçalho)')
modelo = modelo_final().fit(X_dev, y_dev)
y_pred = modelo.predict(X_hold)
acc = accuracy_score(y_hold, y_pred)
cv_mean, cv_std = ref.mean(), ref.std()

print(f'   Acurácia CV:      {cv_mean:.4f} ± {cv_std:.4f}')
print(f'   Acurácia holdout: {acc:.4f}')
print('\n' + classification_report(y_hold, y_pred, digits=4))
print('   Matriz de confusão (linhas = real, colunas = previsto):')
print(pd.DataFrame(confusion_matrix(y_hold, y_pred), index=[1, 2, 3], columns=[1, 2, 3]).to_string())

# O desvio padrão é a dispersão entre folds (~250 exemplos cada), não um intervalo
# de confiança calibrado para a acurácia no servidor.
print(f'\n - Estimativa local de acurácia: {cv_mean:.1%} '
      f'(desvio padrão entre folds: {cv_std:.1%})')

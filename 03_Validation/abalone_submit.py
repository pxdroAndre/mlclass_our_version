#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade 03 - Treino final e envio ao servidor.

Treina o modelo validado em abalone_validation.py com TODOS os dados rotulados
e gera as previsões para abalone_app.csv.

ATENÇÃO: o servidor aceita apenas 1 envio a cada 12h.
  python abalone_submit.py            -> treina, confere e salva previsões (NÃO envia)
  python abalone_submit.py --submit   -> idem + envia ao servidor

@author: Aydano Machado <aydano.machado@gmail.com>
@updated_by: Equipe MachineLerdos
"""

import sys

import pandas as pd
import requests

from abalone_model import carregar_treino, modelo_final

URL = "https://aydanomachado.com/mlclass/03_Validation.php"
DEV_KEY = "MachineLerdos"
N_APP = 1045

# =====================================================================
# 1. TREINO FINAL COM TODOS OS DADOS ROTULADOS
# =====================================================================
print('\n - Lendo dados de treino')
X, y = carregar_treino()

print(f' - Treinando modelo final (SVM RBF + features) com {len(X)} exemplos')
modelo = modelo_final().fit(X, y)

# =====================================================================
# 2. PREVISÃO E CONFERÊNCIA
# =====================================================================
print(' - Lendo abalone_app.csv e gerando previsões')
data_app = pd.read_csv('abalone_app.csv')
y_pred = modelo.predict(data_app)

# Conferências antes de gastar o envio das próximas 12h
assert len(y_pred) == N_APP, f'esperado {N_APP} previsões, obtido {len(y_pred)}'
assert set(y_pred) <= {1, 2, 3}, f'classes inesperadas: {set(y_pred)}'
print(f'   OK: {len(y_pred)} previsões | distribuição: '
      f'{pd.Series(y_pred).value_counts().sort_index().to_dict()}')

pd.Series(y_pred, name='type').to_csv('abalone_app_predictions.csv', index=False)
print(' - Previsões salvas em abalone_app_predictions.csv')

# =====================================================================
# 3. ENVIO (somente com --submit)
# =====================================================================
if '--submit' not in sys.argv:
    print('\n - Envio NÃO realizado. Rode com --submit para enviar ao servidor.\n')
    sys.exit(0)

print(' - Enviando previsões para o servidor...')
data_json = {
    'dev_key': DEV_KEY,
    'predictions': pd.Series(y_pred).to_json(orient='values'),
}
r = requests.post(url=URL, data=data_json)
print(" - Resposta do servidor:\n", r.text, "\n")

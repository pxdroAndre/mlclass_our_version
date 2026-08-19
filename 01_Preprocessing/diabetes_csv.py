#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# import requests

# =====================================================================
# 🛠️ PIPELINE DE PRÉ-PROCESSAMENTO (Onde a equipe trabalha junta)
# =====================================================================
def pre_processar(df):
    """
    Função unificada que recebe um dataframe bruto, aplica a limpeza, 
    o enriquecimento e a transformação final, e o devolve pronto para o modelo.
    """
    df_limpo = df.copy()
    
    # --- [PESSOA 1: LIMPEZA BÁSICA] ---
    # Substituindo zeros (dados faltantes) em Glucose e BMI por valores aleatórios.
    for col in ['Glucose', 'BMI']:
        # Encontra min (ignorando zeros) e max
        min_val = df_limpo[df_limpo[col] > 0][col].min()
        max_val = df_limpo[col].max()
        
        # Mascara de zeros e substituição
        zeros_mask = df_limpo[col] == 0
        df_limpo.loc[zeros_mask, col] = np.random.uniform(min_val, max_val, size=zeros_mask.sum())
        
    # --- [PESSOA 2: ENRIQUECIMENTO E OUTLIERS] ---
    # TODO: Colega 2 vai escrever o código de enriquecimento de dados aqui
    
    # --- [PESSOA 3: TRANSFORMAÇÃO FINAL] ---
    # TODO: Colega 3 vai escrever o código de transformação (ex: escalonamento) aqui
    
    return df_limpo

# =====================================================================
# 🚀 FLUXO PRINCIPAL DO PROGRAMA (O Motor)
# =====================================================================

# Colunas escolhidas após análise da Pessoa 1
feature_cols = ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']

print('\n - Lendo e processando dados de TREINO')
data = pd.read_csv('diabetes_dataset.csv')
data_tratado = pre_processar(data) # <--- Mágica do Pipeline

# Criando X and y para o algoritmo de aprendizagem
X = data_tratado[feature_cols]
y = data_tratado.Outcome

# Criando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

# Realizando previsões com o arquivo de teste
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')

# --- REGRA DE OURO ---
# Aplicando a mesma função de limpeza no arquivo cego de teste
data_app_tratado = pre_processar(data_app)
data_app_final = data_app_tratado[feature_cols]

y_pred = neigh.predict(data_app_final)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "MachineLerdos"

# json para ser enviado para o servidor
data_json = {'dev_key':DEV_KEY,
             'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
# r = requests.post(url = URL, data = data_json)

# Extraindo e imprimindo o texto da resposta
# pastebin_url = r.text
# print(" - Resposta do servidor:\n", r.text, "\n")

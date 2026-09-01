#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de performance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
@updated_by: Equipe MachineLerdos
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import requests

# =====================================================================
# 1. FUNÇÃO DE LIMPEZA E TRATAMENTO DOS DADOS FALTANTES
# =====================================================================
def pre_processar(df):
    """
    Tratamento de dados faltantes (zeros biológicos):
    Em variáveis clínicas vitais (Glicose, Pressão e IMC), o valor 0 é
    biologicamente impossível e representa dado ausente (Missing Value).
    Substituímos o valor 0 por NaN para que a imputação pela mediana trate corretamente.
    """
    df_limpo = df.copy()
    cols_com_zeros_invalidos = ['Glucose', 'BloodPressure', 'BMI']
    for col in cols_com_zeros_invalidos:
        if col in df_limpo.columns:
            df_limpo[col] = df_limpo[col].replace(0, np.nan)
            
    return df_limpo

# =====================================================================
# 2. SELEÇÃO DE VARIÁVEIS (FEATURE SELECTION)
# =====================================================================
# Mantemos as 6 variáveis com dados completos e alto sinal preditivo.
# 'Insulin' (65% de dados faltantes no treino) e 'SkinThickness' (40% de dados faltantes)
# foram removidas para evitar distorção no cálculo da distância euclidiana do k-NN.
feature_cols = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

# =====================================================================
# 3. PIPELINE DE PRÉ-PROCESSAMENTO E MODELAGEM
# =====================================================================
# 1) Imputação: SimpleImputer com a mediana (robusta contra valores atípicos)
# 2) Normalização: MinMaxScaler colocando as 6 features no intervalo [0, 1]
# 3) Classificador: k-NN com k=3 (100% fixo conforme a regra do professor)
neigh = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=3))
    ]
)

# =====================================================================
# 4. TREINAMENTO DO MODELO
# =====================================================================
print('\n - Lendo e processando dados de TREINO')
data = pd.read_csv('diabetes_dataset.csv')
data_tratado = pre_processar(data)

X = data_tratado[feature_cols]
y = data_tratado.Outcome

print(' - Treinando o Pipeline (Pré-processamento + k-NN k=3)')
neigh.fit(X, y)

# =====================================================================
# 5. PREVISÃO E SUBMISSÃO AO SERVIDOR
# =====================================================================
print(' - Lendo arquivo de teste cego e aplicando transformações')
data_app = pd.read_csv('diabetes_app.csv')
data_app_tratado = pre_processar(data_app)
data_app_final = data_app_tratado[feature_cols]

y_pred = neigh.predict(data_app_final)

print(' - Enviando previsões para o servidor...')
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"
DEV_KEY = "MachineLerdos"

data_json = {
    'dev_key': DEV_KEY,
    'predictions': pd.Series(y_pred).to_json(orient='values')
}

r = requests.post(url=URL, data=data_json)
print(" - Resposta do servidor:\n", r.text, "\n")

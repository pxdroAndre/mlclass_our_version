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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import requests

# =====================================================================
# PIPELINE DE PRÉ-PROCESSAMENTO
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
    # Faixas fixas para que treino e teste recebam exatamente as mesmas regras.
    df_limpo['AgeGroup'] = pd.cut(
        df_limpo['Age'],
        bins=[0, 25, 40, float('inf')],
        labels=['Jovem', 'Adulto', 'Idoso'],
        include_lowest=True
    )

    df_limpo['BMI_Category'] = pd.cut(
        df_limpo['BMI'],
        bins=[0, 18.5, 25, 30, float('inf')],
        labels=['Abaixo_peso', 'Normal', 'Sobrepeso', 'Obesidade'],
        include_lowest=True
    )

    # Indicadores clínicos simples: não removem nem alteram os registros.
    df_limpo['HighGlucose'] = (df_limpo['Glucose'] >= 140).astype(int)
    df_limpo['HighBMI'] = (df_limpo['BMI'] >= 30).astype(int)

    # Sinaliza valores extremos pelo critério de Tukey (IQR), preservando-os
    # para que a Pessoa 3 possa comparar o desempenho com e sem esses sinais.
    for col in ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']:
        q1 = df_limpo[col].quantile(0.25)
        q3 = df_limpo[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_limpo[f'{col}_Outlier'] = (
            (df_limpo[col] < lower) | (df_limpo[col] > upper)
        ).astype(int)
    
    return df_limpo

# =====================================================================
# FLUXO PRINCIPAL DO PROGRAMA
# =====================================================================

# --- [PESSOA 3: DEFINIÇÃO DAS VARIÁVEIS] ---
# Colunas numéricas 
numeric_features = [
    'Pregnancies', 
    'Glucose', 
    'BMI', 
    'DiabetesPedigreeFunction', 
    'Age',
    'HighGlucose',
    'HighBMI',
    'Glucose_Outlier',
    'BMI_Outlier',
    'Age_Outlier',
    'DiabetesPedigreeFunction_Outlier'
]

# Colunas categóricas
categorical_features = [
    'AgeGroup',
    'BMI_Category'
]

# Todas as colunas que serãao usadas  pelo modelo
feature_cols = numeric_features + categorical_features

# --- [PESSOA 3: PRÉ-PROCESSAMENTO PARA O KNN] ---
# Define as transformações que serão aplicadas aos dados:
# StandardScaler nas variáveis numéricas
# OneHotEncoder nas variáveis categóricas
numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

print('\n - Lendo e processando dados de TREINO')
data = pd.read_csv('diabetes_dataset.csv')
data_tratado = pre_processar(data)

# Criando X and y para o algoritmo de aprendizagem
X = data_tratado[feature_cols]
y = data_tratado.Outcome

# --- [PESSOA 3: PIPELINE E MODELO KNN] ---
# Criando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=3))
    ]
)
neigh.fit(X, y)

# Realizando previsões com o arquivo de teste
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')

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
r = requests.post(url = URL, data = data_json)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")

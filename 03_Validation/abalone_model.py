#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definições compartilhadas da Atividade 03 - Validação.

Aqui ficam a engenharia de atributos, a limpeza do treino e a construção
dos pipelines. Tanto o script de validação quanto o de envio importam daqui,
garantindo que o modelo validado seja exatamente o modelo enviado.

@updated_by: Equipe MachineLerdos
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
TARGET = 'type'

# Atributos originais da base
NUM_COLS = [
    'length', 'diameter', 'height',
    'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight',
]

# Atributos derivados criados em criar_features()
EXTRA_COLS = [
    'shell_ratio', 'shucked_ratio', 'viscera_ratio',
    'volume', 'density',
    'log_whole_weight', 'log_shucked_weight', 'log_viscera_weight', 'log_shell_weight',
]

# Hiperparâmetros do modelo final, escolhidos por GridSearchCV no abalone_validation.py
FINAL_PARAMS = {'C': 1, 'gamma': 'scale'}


# =====================================================================
# 1. LIMPEZA DO TREINO
# =====================================================================
def limpar_treino(df):
    """
    Remove registros fisicamente impossíveis ou claramente errados do TREINO:
    - height == 0: abalone sem altura não existe (erro de medição);
    - height > 0.4: único caso (0.515) muito acima de todo o resto.
    Nunca aplicar no abalone_app.csv: todas as 1045 linhas precisam de previsão.
    """
    mask = (df['height'] > 0) & (df['height'] <= 0.4)
    return df[mask].reset_index(drop=True)


# =====================================================================
# 2. ENGENHARIA DE ATRIBUTOS
# =====================================================================
def criar_features(df):
    """
    Cria atributos derivados. Todos são calculados linha a linha (sem estatísticas
    do conjunto), portanto não há vazamento de informação entre treino e teste.
    - Razões de peso: proporção de concha/carne/víscera no peso total, que muda
      com a idade independente do tamanho do animal;
    - Volume e densidade aproximados a partir das medidas;
    - Log dos pesos: os pesos são muito assimétricos (cauda longa à direita).
    """
    df = df.copy()
    peso = df['whole_weight'].clip(lower=1e-3)
    df['shell_ratio'] = df['shell_weight'] / peso
    df['shucked_ratio'] = df['shucked_weight'] / peso
    df['viscera_ratio'] = df['viscera_weight'] / peso
    df['volume'] = df['length'] * df['diameter'] * df['height']
    df['density'] = df['whole_weight'] / df['volume'].clip(lower=1e-4)
    for col in ['whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']:
        df['log_' + col] = np.log1p(df[col] * 10)
    return df


# =====================================================================
# 3. PIPELINES
# =====================================================================
def preprocessador(com_features=True):
    """
    One-hot em 'sex' (categoria sem ordem) + padronização dos numéricos.
    O StandardScaler fica dentro do Pipeline, então em cada fold da validação
    cruzada ele é ajustado apenas nos dados de treino daquele fold.
    """
    nums = NUM_COLS + (EXTRA_COLS if com_features else [])
    steps = []
    if com_features:
        steps.append(('features', FunctionTransformer(criar_features)))
    steps.append(('colunas', ColumnTransformer([
        ('sex', OneHotEncoder(handle_unknown='ignore'), ['sex']),
        ('num', StandardScaler(), nums),
    ])))
    return steps


def montar_pipeline(classificador, com_features=True):
    return Pipeline(preprocessador(com_features) + [('classifier', classificador)])


def modelo_final():
    """SVM com kernel RBF + atributos derivados: melhor resultado na validação."""
    return montar_pipeline(SVC(kernel='rbf', random_state=RANDOM_STATE, **FINAL_PARAMS))


def carregar_treino(caminho='abalone_dataset.csv'):
    df = limpar_treino(pd.read_csv(caminho))
    return df.drop(columns=TARGET), df[TARGET]

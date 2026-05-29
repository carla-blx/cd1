# utils.py - Versión definitiva corregida
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# =============================================================================
# DEFINICIÓN MANUAL DE COLUMNAS
# =============================================================================

NUM_COLS = [
    'LIMIT_BAL', 'AGE', 
    'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'ratio_pago', 'meses_mora', 'max_mora', 'tendencia_mora', 'log_limit_bal'
]

CAT_COLS = ['SEX', 'EDUCATION', 'MARRIAGE']

# Percentiles para winsorización
P01_RATIO = 0.0
P99_RATIO = 1.0

PIPELINE = None

# =============================================================================
# FUNCIONES DE CREACIÓN DE FEATURES
# =============================================================================

def crear_features_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las 5 features derivadas"""
    df = df.copy()
    
    PAY_COLS = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    BILL_COLS = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    PAY_AMT_COLS = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # ratio_pago
    total_pagado = df[PAY_AMT_COLS].sum(axis=1)
    total_facturado = df[BILL_COLS].abs().sum(axis=1)
    df['ratio_pago'] = total_pagado / (total_facturado + 1)
    df['ratio_pago'] = df['ratio_pago'].clip(lower=P01_RATIO, upper=P99_RATIO)
    
    # meses_mora
    df['meses_mora'] = (df[PAY_COLS] > 0).sum(axis=1)
    
    # max_mora
    df['max_mora'] = df[PAY_COLS].max(axis=1)
    
    # tendencia_mora
    df['tendencia_mora'] = df['PAY_1'] - df['PAY_6']
    
    # log_limit_bal
    df['log_limit_bal'] = np.log1p(df['LIMIT_BAL'])
    
    return df

def aplicar_mapeos_categoricos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica los mapeos de educación y estado civil"""
    df = df.copy()
    
    education_map = {4: 4, 5: 4, 6: 4}
    df['EDUCATION'] = df['EDUCATION'].map(education_map).fillna(df['EDUCATION'])
    
    marriage_map = {0: 3}
    df['MARRIAGE'] = df['MARRIAGE'].map(marriage_map).fillna(df['MARRIAGE'])
    
    return df

# =============================================================================
# CREACIÓN DEL PIPELINE
# =============================================================================

def create_pipeline():
    """Crea el pipeline de preprocesamiento"""
    global PIPELINE
    
    print(f"Columnas numéricas: {len(NUM_COLS)}")
    print(f"Columnas categóricas: {CAT_COLS}")
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    PIPELINE = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUM_COLS),
            ('cat', categorical_transformer, CAT_COLS)
        ]
    )
    
    # Crear datos dummy para entrenar el pipeline
    # Incluir todas las combinaciones posibles de categorías
    dummy_data_list = []
    
    for sex in [1, 2]:
        for edu in [1, 2, 3, 4]:
            for marriage in [1, 2, 3]:
                row = {}
                # Valores numéricos en 0
                for col in NUM_COLS:
                    row[col] = 0
                # Valores categóricos
                row['SEX'] = sex
                row['EDUCATION'] = edu
                row['MARRIAGE'] = marriage
                dummy_data_list.append(row)
    
    dummy_df = pd.DataFrame(dummy_data_list)
    PIPELINE.fit(dummy_df)
    
    # Verificar dimensiones
    test_output = PIPELINE.transform(dummy_df.head(1))
    print(f"✅ Pipeline creado. Output shape: {test_output.shape[1]} features")
    
    return PIPELINE

# =============================================================================
# PREPROCESAMIENTO PRINCIPAL
# =============================================================================

def preprocess_input(data_dict: dict) -> np.ndarray:
    """Preprocesa los datos usando el pipeline"""
    global PIPELINE
    
    if PIPELINE is None:
        create_pipeline()
    
    # 1. Crear DataFrame
    df = pd.DataFrame([data_dict])
    
    # 2. Aplicar mapeos categóricos
    df = aplicar_mapeos_categoricos(df)
    
    # 3. Crear features derivadas
    df = crear_features_derivadas(df)
    
    # 4. Seleccionar columnas en el orden correcto
    columnas_esperadas = NUM_COLS + CAT_COLS
    
    # 5. Agregar columnas faltantes con 0
    for col in columnas_esperadas:
        if col not in df.columns:
            df[col] = 0
    
    # 6. Ordenar columnas
    df_final = df[columnas_esperadas]
    
    # 7. Aplicar pipeline
    try:
        X_processed = PIPELINE.transform(df_final)
        print(f"✅ Preprocesamiento exitoso. Shape: {X_processed.shape}")
        return X_processed.astype(np.float32)
    except Exception as e:
        print(f"Error en transform: {e}")
        raise

# =============================================================================
# LOADERS
# =============================================================================

def load_pipeline():
    global PIPELINE
    if PIPELINE is None:
        create_pipeline()
    return PIPELINE

def load_keras_model():
    return tf.keras.models.load_model("keras_model.keras")

def load_sklearn_model():
    return joblib.load("sklearn_model.pkl")

def get_model_info() -> dict:
    return {
        'Keras': {'trained': True, 'status': '✅ Modelo Keras cargado'},
        'Scikit-Learn': {'trained': True, 'status': '✅ Modelo Sklearn cargado'}
    }

def predict_keras(model, X: np.ndarray) -> float:
    proba = model.predict(X, verbose=0)[0][0]
    return float(proba)

def predict_sklearn(model, X: np.ndarray) -> float:
    proba = model.predict_proba(X)[0][1]
    return float(proba)

def get_pipeline_output_shape():
    global PIPELINE
    if PIPELINE is None:
        create_pipeline()
    
    dummy_input = pd.DataFrame(columns=NUM_COLS + CAT_COLS)
    dummy_input.loc[0] = [0] * len(NUM_COLS + CAT_COLS)
    output = PIPELINE.transform(dummy_input)
    return output.shape[1]

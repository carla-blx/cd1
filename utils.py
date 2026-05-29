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
    
    # Crear datos dummy REALISTAS para entrenar el scaler
    np.random.seed(42)
    n_dummy_samples = 1000
    
    dummy_data = {}
    
    # Datos numéricos con rangos realistas
    dummy_data['LIMIT_BAL'] = np.random.uniform(10000, 1000000, n_dummy_samples)
    dummy_data['AGE'] = np.random.uniform(18, 80, n_dummy_samples)
    
    # PAY_* valores entre -2 y 8
    for i in range(1, 7):
        dummy_data[f'PAY_{i}'] = np.random.randint(-2, 9, n_dummy_samples)
    
    # BILL_AMT* valores entre 0 y 500000
    for i in range(1, 7):
        dummy_data[f'BILL_AMT{i}'] = np.random.uniform(0, 500000, n_dummy_samples)
    
    # PAY_AMT* valores entre 0 y 200000
    for i in range(1, 7):
        dummy_data[f'PAY_AMT{i}'] = np.random.uniform(0, 200000, n_dummy_samples)
    
    # Features derivadas (se calcularán después)
    dummy_data['ratio_pago'] = np.random.uniform(0, 1, n_dummy_samples)
    dummy_data['meses_mora'] = np.random.randint(0, 7, n_dummy_samples)
    dummy_data['max_mora'] = np.random.randint(0, 9, n_dummy_samples)
    dummy_data['tendencia_mora'] = np.random.randint(-8, 8, n_dummy_samples)
    dummy_data['log_limit_bal'] = np.log1p(dummy_data['LIMIT_BAL'])
    
    # Datos categóricos
    dummy_data['SEX'] = np.random.choice([1, 2], n_dummy_samples)
    dummy_data['EDUCATION'] = np.random.choice([1, 2, 3, 4], n_dummy_samples)
    dummy_data['MARRIAGE'] = np.random.choice([1, 2, 3], n_dummy_samples)
    
    dummy_df = pd.DataFrame(dummy_data)
    
    # Asegurar el orden de columnas
    dummy_df = dummy_df[NUM_COLS + CAT_COLS]
    
    # Crear y entrenar el pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    PIPELINE = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUM_COLS),
            ('cat', categorical_transformer, CAT_COLS)
        ]
    )
    
    # Entrenar el pipeline
    PIPELINE.fit(dummy_df)
    
    # Verificar que funciona
    test_output = PIPELINE.transform(dummy_df.head(1))
    print(f"✅ Pipeline entrenado. Output shape: {test_output.shape[1]}")
    print(f"   Rango de salida - min: {test_output.min():.4f}, max: {test_output.max():.4f}")
    
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

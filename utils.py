# utils.py - Versión con definición manual de columnas
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# =============================================================================
# DEFINICIÓN MANUAL DE COLUMNAS (basado en tu entrenamiento)
# =============================================================================

# 20 columnas originales + 5 derivadas = 25 numéricas
NUM_COLS = [
    'LIMIT_BAL', 'AGE', 
    'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'ratio_pago', 'meses_mora', 'max_mora', 'tendencia_mora', 'log_limit_bal'
]

# 3 columnas categóricas
CAT_COLS = ['SEX', 'EDUCATION', 'MARRIAGE']

# Features finales (para referencia)
FEATURES_FINALES = NUM_COLS + CAT_COLS

# Percentiles para winsorización (valores por defecto)
P01_RATIO = 0.0
P99_RATIO = 1.0

# Pipeline (se creará al inicio)
PIPELINE = None

# =============================================================================
# CREACIÓN DEL PIPELINE MANUAL
# =============================================================================
def create_pipeline():
    """Crea el pipeline de preprocesamiento manualmente"""
    global PIPELINE
    
    # Para depuración - ver cuántas categorías tiene cada variable
    print(f"Columnas numéricas: {len(NUM_COLS)}")
    print(f"Columnas categóricas: {CAT_COLS}")
    
    numeric_transformer = StandardScaler()
    # OneHotEncoder con drop='first' crea (n_categorias - 1) columnas por variable
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    PIPELINE = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUM_COLS),
            ('cat', categorical_transformer, CAT_COLS)
        ]
    )
    
    # Crear datos dummy representativos para entrenar el pipeline
    # Esto es importante para que el OneHotEncoder "aprenda" las categorías
    dummy_data = pd.DataFrame(columns=NUM_COLS + CAT_COLS)
    
    # Valores representativos para categóricas
    # SEX: 1 o 2
    # EDUCATION: 1,2,3,4 (4 agrupa otros)
    # MARRIAGE: 1,2,3
    dummy_rows = []
    for sex in [1, 2]:
        for edu in [1, 2, 3, 4]:
            for marriage in [1, 2, 3]:
                row = [0] * len(NUM_COLS + CAT_COLS)
                # Establecer valores categóricos
                row[NUM_COLS + CAT_COLS.index('SEX')] = sex
                row[NUM_COLS + CAT_COLS.index('EDUCATION')] = edu
                row[NUM_COLS + CAT_COLS.index('MARRIAGE')] = marriage
                dummy_rows.append(row)
    
    dummy_data = pd.DataFrame(dummy_rows, columns=NUM_COLS + CAT_COLS)
    dummy_data[NUM_COLS] = 0  # Valores numéricos en cero
    
    PIPELINE.fit(dummy_data)
    
    # Verificar dimensiones después de fit
    dummy_transformed = PIPELINE.transform(dummy_data.head(1))
    print(f"✅ Pipeline creado. Dimensiones de salida: {dummy_transformed.shape[1]} features")
    print(f"   (Esperado: 31 = 25 numéricas + 6 categóricas)")
    
    return PIPELINE

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
    
    # Winsorización
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
# PREPROCESAMIENTO PRINCIPAL
# =============================================================================

def preprocess_input(data_dict: dict) -> np.ndarray:
    """Preprocesa los datos usando el pipeline"""
    global PIPELINE
    
    # Crear pipeline si no existe
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
    
    # 5. Agregar columnas faltantes
    for col in columnas_esperadas:
        if col not in df.columns:
            df[col] = 0
    
    # 6. Ordenar columnas
    df_final = df[columnas_esperadas]
    
    # 7. Aplicar pipeline
    try:
        X_processed = PIPELINE.transform(df_final)
        return X_processed.astype(np.float32)
    except Exception as e:
        print(f"Error en transform: {e}")
        raise

# =============================================================================
# LOADERS
# =============================================================================

def load_pipeline():
    """Carga o crea el pipeline"""
    global PIPELINE
    if PIPELINE is None:
        create_pipeline()
    return PIPELINE

def load_keras_model():
    """Carga el modelo de Keras"""
    return tf.keras.models.load_model("keras_model.keras")

def load_sklearn_model():
    """Carga el modelo de Scikit-Learn"""
    return joblib.load("sklearn_model.pkl")

def get_model_info() -> dict:
    """Información de los modelos"""
    return {
        'Keras': {'trained': True, 'status': '✅ Modelo Keras cargado'},
        'Scikit-Learn': {'trained': True, 'status': '✅ Modelo Sklearn cargado'}
    }

def predict_keras(model, X: np.ndarray) -> float:
    """Predicción con Keras"""
    return float(model.predict(X, verbose=0)[0][0])

def predict_sklearn(model, X: np.ndarray) -> float:
    """Predicción con Scikit-Learn"""
    return float(model.predict_proba(X)[0][1])

# =============================================================================
# PREDICCIONES
# =============================================================================

def predict_keras(model, X: np.ndarray) -> float:
    """Predicción con Keras"""
    proba = model.predict(X, verbose=0)[0][0]
    return float(proba)

def predict_sklearn(model, X: np.ndarray) -> float:
    """Predicción con Scikit-Learn"""
    proba = model.predict_proba(X)[0][1]
    return float(proba)

# =============================================================================
# TEST


if __name__ == "__main__":
    # Prueba con datos de ejemplo
    test_data = {
        'LIMIT_BAL': 200000,
        'AGE': 30,
        'PAY_1': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 0,
        'BILL_AMT1': 5000,
        'BILL_AMT2': 4000,
        'BILL_AMT3': 3000,
        'BILL_AMT4': 2500,
        'BILL_AMT5': 2000,
        'BILL_AMT6': 1500,
        'PAY_AMT1': 1000,
        'PAY_AMT2': 1000,
        'PAY_AMT3': 1000,
        'PAY_AMT4': 1000,
        'PAY_AMT5': 1000,
        'PAY_AMT6': 1000,
        'SEX': 1,
        'EDUCATION': 2,
        'MARRIAGE': 1
    }
    
    X = preprocess_input(test_data)
    print(f"\n✅ Shape final: {X.shape}")
    print(f"¿Es 31? {X.shape[1] == 31}")

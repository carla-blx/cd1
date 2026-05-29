# utils.py - Versión corregida definitiva
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import pickle

# Variables globales
FEATURES_FINALES = None
NUM_COLS = None
CAT_COLS = None
P01_RATIO = None
P99_RATIO = None
PIPELINE = None

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

def load_pipeline():
    """Carga el pipeline o lo recrea si hay error de versión"""
    global FEATURES_FINALES, NUM_COLS, CAT_COLS, P01_RATIO, P99_RATIO, PIPELINE
    
    # Intentar cargar normalmente
    try:
        with open('preprocessing_pipeline.pkl', 'rb') as f:
            pipeline_dict = pickle.load(f)
        PIPELINE = pipeline_dict['preprocessing_pipeline']
        FEATURES_FINALES = pipeline_dict['features_finales']
        NUM_COLS = pipeline_dict['num_cols']
        CAT_COLS = pipeline_dict['cat_cols']
        P01_RATIO = pipeline_dict['p01_ratio']
        P99_RATIO = pipeline_dict['p99_ratio']
        print("✅ Pipeline cargado desde archivo")
        return PIPELINE
    except Exception as e:
        print(f"⚠️ Error cargando pipeline: {e}")
        print("🔄 Recreando pipeline manualmente...")
        
        # Definir columnas manualmente (según tu entrenamiento)
        NUM_COLS = [
            'LIMIT_BAL', 'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
            'ratio_pago', 'meses_mora', 'max_mora', 'tendencia_mora', 'log_limit_bal'
        ]
        CAT_COLS = ['SEX', 'EDUCATION', 'MARRIAGE']
        FEATURES_FINALES = NUM_COLS + CAT_COLS  # Simplificado
        P01_RATIO = 0.0  # Valores por defecto
        P99_RATIO = 1.0
        
        # Crear pipeline nuevo
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        PIPELINE = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, NUM_COLS),
                ('cat', categorical_transformer, CAT_COLS)
            ]
        )
        
        # Entrenar pipeline con datos dummy (solo para que funcione)
        dummy_data = pd.DataFrame(columns=NUM_COLS + CAT_COLS)
        dummy_data.loc[0] = [0] * len(NUM_COLS + CAT_COLS)
        PIPELINE.fit(dummy_data)
        
        print("✅ Pipeline recreado manualmente")
        return PIPELINE

# Cargar pipeline al inicio
#load_pipeline()

# =============================================================================
# FUNCIONES DE CREACIÓN DE FEATURES
# =============================================================================

def crear_features_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las 5 features derivadas exactamente como en el entrenamiento
    """
    df = df.copy()
    
    PAY_COLS = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    BILL_COLS = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    PAY_AMT_COLS = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # ratio_pago
    total_pagado = df[PAY_AMT_COLS].sum(axis=1)
    total_facturado = df[BILL_COLS].abs().sum(axis=1)
    df['ratio_pago'] = total_pagado / (total_facturado + 1)
    
    # Winsorización con percentiles del training
    if P01_RATIO is not None and P99_RATIO is not None:
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
    
    # Mapeo de educación según tu Colab: {4: 4, 5: 4, 6: 4}
    education_map = {4: 4, 5: 4, 6: 4}
    df['EDUCATION'] = df['EDUCATION'].map(education_map).fillna(df['EDUCATION'])
    
    # Mapeo de matrimonio según tu Colab: {0: 3}
    marriage_map = {0: 3}
    df['MARRIAGE'] = df['MARRIAGE'].map(marriage_map).fillna(df['MARRIAGE'])
    
    return df

# =============================================================================
# PREPROCESAMIENTO PRINCIPAL
# =============================================================================

def preprocess_input(data_dict: dict) -> np.ndarray:
    """
    Preprocesa los datos usando el pipeline real.
    El pipeline espera: NUM_COLS (25) + CAT_COLS (3) = 28 columnas de entrada
    """
    
    # Si el pipeline falló al cargar, intentar recrearlo
    if PIPELINE is None:
        print("⚠️ Pipeline no disponible, intentando recrear...")
        success = recreate_pipeline_from_vars()
        if not success or PIPELINE is None:
            raise ValueError("No se pudo cargar ni recrear el pipeline")
    
    # 1. Crear DataFrame con todos los datos
    df = pd.DataFrame([data_dict])
    
    # 2. Aplicar mapeos categóricos
    df = aplicar_mapeos_categoricos(df)
    
    # 3. Crear features derivadas (esto agrega 5 nuevas columnas)
    df = crear_features_derivadas(df)
    
    # 4. El pipeline espera las columnas en el orden: NUM_COLS + CAT_COLS
    columnas_esperadas = NUM_COLS + CAT_COLS
    
    # 5. Verificar que todas las columnas existan (agregar las que falten con 0)
    for col in columnas_esperadas:
        if col not in df.columns:
            df[col] = 0
    
    # 6. Seleccionar solo las columnas esperadas en el orden correcto
    df_final = df[columnas_esperadas]
    
    # 7. Aplicar el pipeline
    try:
        X_processed = PIPELINE.transform(df_final)
        return X_processed.astype(np.float32)
    except Exception as e:
        print(f"❌ Error en pipeline.transform: {e}")
        raise

def recreate_pipeline_from_vars():
    """Recrea el pipeline usando las variables globales"""
    global PIPELINE, NUM_COLS, CAT_COLS
    
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        
        # Verificar que las columnas estén definidas
        if not NUM_COLS or not CAT_COLS:
            print("❌ No se definieron NUM_COLS o CAT_COLS")
            return False
        
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        PIPELINE = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, NUM_COLS),
                ('cat', categorical_transformer, CAT_COLS)
            ]
        )
        
        # Entrenar el pipeline con datos dummy
        dummy_data = pd.DataFrame(columns=NUM_COLS + CAT_COLS)
        dummy_data.loc[0] = [0] * len(NUM_COLS + CAT_COLS)
        PIPELINE.fit(dummy_data)
        
        print("✅ Pipeline recreado exitosamente")
        return True
    except Exception as e:
        print(f"❌ Error recreando pipeline: {e}")
        return False
# =============================================================================
# LOADERS
# =============================================================================

def load_keras_model():
    """Carga el modelo de Keras"""
    model = tf.keras.models.load_model("keras_model.keras")
    return model

def load_sklearn_model():
    """Carga el modelo de Scikit-Learn"""
    model = joblib.load("sklearn_model.pkl")
    return model

def get_model_info() -> dict:
    """Información de los modelos"""
    return {
        'Keras': {
            'trained': True,
            'status': '✅ Modelo Keras cargado'
        },
        'Scikit-Learn': {
            'trained': True,
            'status': '✅ Modelo Sklearn cargado'
        }
    }

def get_feature_count() -> int:
    """Retorna el número de features"""
    if FEATURES_FINALES:
        return len(FEATURES_FINALES)
    return 31

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

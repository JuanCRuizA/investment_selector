# ================================================
# Data Loader Module
# ================================================
"""
Funciones para carga y preprocesamiento de datos desde SQLite.
Incluye funciones de split train/test e imputación de datos.
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime


def connect_database(db_path: str) -> sqlite3.Connection:
    """
    Conectar a base de datos SQLite.
    
    Args:
        db_path: Ruta al archivo .db
        
    Returns:
        Conexión a la base de datos
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Base de datos no encontrada: {db_path}")
    return sqlite3.connect(db_path)


def list_tables(conn: sqlite3.Connection) -> List[str]:
    """
    Listar todas las tablas disponibles en la base de datos.
    
    Args:
        conn: Conexión a la base de datos
        
    Returns:
        Lista de nombres de tablas
    """
    query = "SELECT name FROM sqlite_master WHERE type='table'"
    tables = pd.read_sql(query, conn)
    return tables['name'].tolist()


def load_prices(conn: sqlite3.Connection, 
                table_name: str = 'prices',
                start_date: Optional[str] = None,
                tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Cargar datos de precios desde la base de datos.
    
    Args:
        conn: Conexión a la base de datos
        table_name: Nombre de la tabla de precios
        start_date: Fecha de inicio (formato 'YYYY-MM-DD')
        tickers: Lista de tickers a cargar (None = todos)
        
    Returns:
        DataFrame con precios
    """
    query = f"SELECT * FROM {table_name}"
    conditions = []
    
    if start_date:
        conditions.append(f"date >= '{start_date}'")
    
    if tickers:
        ticker_list = ','.join([f"'{t}'" for t in tickers])
        conditions.append(f"ticker IN ({ticker_list})")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    df = pd.read_sql(query, conn, parse_dates=['date'])
    return df


def get_valid_tickers(df: pd.DataFrame, 
                      min_observations: int = 1260,
                      date_col: str = 'date',
                      ticker_col: str = 'ticker') -> List[str]:
    """
    Filtrar tickers con suficientes observaciones.
    
    Args:
        df: DataFrame con datos de precios
        min_observations: Mínimo de observaciones requeridas (default: 5 años)
        date_col: Nombre de columna de fecha
        ticker_col: Nombre de columna de ticker
        
    Returns:
        Lista de tickers válidos
    """
    ticker_counts = df.groupby(ticker_col)[date_col].count()
    valid_tickers = ticker_counts[ticker_counts >= min_observations].index.tolist()
    return valid_tickers


def pivot_prices(df: pd.DataFrame,
                 date_col: str = 'date',
                 ticker_col: str = 'ticker',
                 value_col: str = 'adj_close') -> pd.DataFrame:
    """
    Convertir datos de formato largo a ancho (pivot).
    
    Args:
        df: DataFrame en formato largo
        date_col: Columna de fecha
        ticker_col: Columna de ticker
        value_col: Columna de valores (precio)
        
    Returns:
        DataFrame pivotado con fechas como índice y tickers como columnas
    """
    df_pivot = df.pivot(index=date_col, columns=ticker_col, values=value_col)
    df_pivot.index = pd.to_datetime(df_pivot.index)
    return df_pivot.sort_index()


# ================================================
# FUNCIONES NUEVAS PARA PIPELINE PRODUCTIVO
# ================================================

def impute_adj_close(df: pd.DataFrame,
                     adj_close_col: str = 'adj_close',
                     close_col: str = 'close') -> pd.DataFrame:
    """
    Imputar valores nulos de adj_close usando close.
    
    Args:
        df: DataFrame con datos de precios
        adj_close_col: Nombre de la columna adj_close
        close_col: Nombre de la columna close
        
    Returns:
        DataFrame con adj_close imputado
    """
    df = df.copy()
    
    if close_col in df.columns and adj_close_col in df.columns:
        # Contar nulos antes
        nulls_before = df[adj_close_col].isnull().sum()
        
        # Imputar
        df[adj_close_col] = df[adj_close_col].fillna(df[close_col])
        
        # Contar nulos después
        nulls_after = df[adj_close_col].isnull().sum()
        
        if nulls_before > nulls_after:
            print(f"   ℹ️ Imputados {nulls_before - nulls_after} valores de adj_close desde close")
    
    return df


def split_train_test(df_prices: pd.DataFrame,
                     train_end_date: str = '2023-12-31',
                     test_start_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dividir datos de precios en conjuntos de entrenamiento y prueba.
    
    Args:
        df_prices: DataFrame de precios (índice = fecha, columnas = tickers)
        train_end_date: Fecha final del período de entrenamiento
        test_start_date: Fecha inicial del período de prueba (opcional)
        
    Returns:
        Tupla (df_train, df_test)
    """
    train_end = pd.to_datetime(train_end_date)
    
    if test_start_date:
        test_start = pd.to_datetime(test_start_date)
    else:
        # Por defecto, día siguiente al fin de train
        test_start = train_end + pd.Timedelta(days=1)
    
    df_train = df_prices[df_prices.index <= train_end].copy()
    df_test = df_prices[df_prices.index >= test_start].copy()
    
    return df_train, df_test


def validate_benchmark(df_prices: pd.DataFrame,
                       benchmark_ticker: str = 'SPY') -> bool:
    """
    Validar que el benchmark esté presente en los datos.
    
    Args:
        df_prices: DataFrame de precios
        benchmark_ticker: Ticker del benchmark
        
    Returns:
        True si el benchmark está presente, False si no
    """
    if benchmark_ticker not in df_prices.columns:
        print(f"   ⚠️ Benchmark {benchmark_ticker} no encontrado en los datos")
        return False
    
    # Verificar que no tenga demasiados nulos
    null_pct = df_prices[benchmark_ticker].isnull().mean()
    if null_pct > 0.1:
        print(f"   ⚠️ Benchmark {benchmark_ticker} tiene {null_pct:.1%} de valores nulos")
        return False
    
    return True


def fill_missing_prices(df_prices: pd.DataFrame,
                        method: str = 'ffill',
                        max_consecutive_nans: int = 5) -> pd.DataFrame:
    """
    Rellenar valores faltantes en precios.
    
    Args:
        df_prices: DataFrame de precios
        method: Método de relleno ('ffill', 'bfill', 'interpolate')
        max_consecutive_nans: Máximo de NaNs consecutivos permitidos
        
    Returns:
        DataFrame con valores rellenados
    """
    df = df_prices.copy()
    
    if method == 'ffill':
        df = df.ffill(limit=max_consecutive_nans)
    elif method == 'bfill':
        df = df.bfill(limit=max_consecutive_nans)
    elif method == 'interpolate':
        df = df.interpolate(method='linear', limit=max_consecutive_nans)
    
    return df


def get_data_summary(df_prices: pd.DataFrame) -> Dict[str, Any]:
    """
    Obtener resumen de los datos de precios.
    
    Args:
        df_prices: DataFrame de precios
        
    Returns:
        Diccionario con estadísticas resumidas
    """
    return {
        'n_tickers': len(df_prices.columns),
        'n_dates': len(df_prices),
        'date_range': {
            'start': df_prices.index.min().strftime('%Y-%m-%d'),
            'end': df_prices.index.max().strftime('%Y-%m-%d')
        },
        'missing_pct': df_prices.isnull().mean().mean() * 100,
        'tickers': df_prices.columns.tolist()
    }


def load_prices_from_csv(file_path: str,
                         date_col: str = 'date') -> pd.DataFrame:
    """
    Cargar precios desde archivo CSV.
    
    Args:
        file_path: Ruta al archivo CSV
        date_col: Nombre de la columna de fecha
        
    Returns:
        DataFrame de precios con fecha como índice
    """
    df = pd.read_csv(file_path, parse_dates=[date_col], index_col=date_col)
    return df.sort_index()


def save_prices_to_csv(df_prices: pd.DataFrame,
                       file_path: str,
                       date_col: str = 'date') -> None:
    """
    Guardar precios a archivo CSV.
    
    Args:
        df_prices: DataFrame de precios
        file_path: Ruta de destino
        date_col: Nombre de la columna de fecha
    """
    # Asegurar que el directorio existe
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar con índice como columna date
    df_prices.index.name = date_col
    df_prices.to_csv(file_path)


def run_data_ingestion(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Ejecutar proceso completo de ingesta de datos.
    
    Args:
        config: Diccionario de configuración
        
    Returns:
        Tupla (df_train, df_test, valid_tickers)
    """
    from .utils import print_step_header, print_success, print_info, get_path
    
    print_step_header("DATA INGESTION", 1)
    
    # Extraer configuración
    data_config = config.get('data', {})
    params = config.get('data_params', {})
    
    db_path = get_path(data_config.get('database_path', 'data/raw/trading_data.db'))
    table_name = data_config.get('table_name', 'prices')
    min_observations = params.get('min_observations', 1260)
    train_end = params.get('train_end_date', '2023-12-31')
    benchmark = params.get('benchmark_ticker', 'SPY')
    
    # 1. Conectar a base de datos
    print_info(f"Conectando a: {db_path}")
    conn = connect_database(str(db_path))
    
    # 2. Cargar datos
    print_info(f"Cargando tabla: {table_name}")
    df_raw = load_prices(conn, table_name)
    print_success(f"Cargados {len(df_raw):,} registros")
    
    # 3. Imputar adj_close
    df_raw = impute_adj_close(df_raw)
    
    # 4. Filtrar tickers válidos
    valid_tickers = get_valid_tickers(df_raw, min_observations)
    print_success(f"Tickers con ≥{min_observations} observaciones: {len(valid_tickers)}")
    
    # 5. Filtrar y pivotar
    df_filtered = df_raw[df_raw['ticker'].isin(valid_tickers)]
    df_prices = pivot_prices(df_filtered)
    
    # 6. Rellenar valores faltantes
    df_prices = fill_missing_prices(df_prices, method=params.get('fillna_method', 'ffill'))
    
    # 7. Validar benchmark
    validate_benchmark(df_prices, benchmark)
    
    # 8. Split train/test
    df_train, df_test = split_train_test(df_prices, train_end)
    
    print_success(f"Train: {df_train.index.min().strftime('%Y-%m-%d')} a {df_train.index.max().strftime('%Y-%m-%d')} ({len(df_train)} días)")
    print_success(f"Test: {df_test.index.min().strftime('%Y-%m-%d')} a {df_test.index.max().strftime('%Y-%m-%d')} ({len(df_test)} días)")
    
    conn.close()
    
    return df_train, df_test, valid_tickers

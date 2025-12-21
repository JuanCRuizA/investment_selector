# ================================================
# Data Loader Module
# ================================================
"""
Funciones para carga y preprocesamiento de datos desde SQLite.
"""

import pandas as pd
import sqlite3
from pathlib import Path
from typing import List, Optional


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

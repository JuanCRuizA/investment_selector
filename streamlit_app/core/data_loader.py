"""
Data Loader Module - Handles loading and caching of pipeline outputs
"""
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional
import streamlit as st


class DataLoader:
    """
    Centralized data loading with Streamlit caching.
    Loads pre-computed outputs from pipeline.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize DataLoader with path to API outputs.
        
        Args:
            data_path: Path to outputs/api directory. Defaults to relative path.
        """
        if data_path is None:
            # Try multiple possible paths
            possible_paths = [
                Path(__file__).parent.parent.parent / "outputs" / "api",
                Path(__file__).parent.parent / "data",
                Path("outputs/api"),
                Path("../outputs/api"),
            ]
            for p in possible_paths:
                if p.exists():
                    data_path = p
                    break
            else:
                data_path = Path(__file__).parent.parent / "data"
        
        self.data_path = Path(data_path)
    
    @st.cache_data(ttl=3600)
    def load_portfolios(_self) -> pd.DataFrame:
        """
        Load portfolios data with all asset allocations.
        
        Returns:
            DataFrame with columns: perfil, ticker, segmento_nombre, 
            return_annualized, volatility_annual, sharpe_ratio, beta,
            momentum_6m, score_compuesto, peso
        """
        path = _self.data_path / "portfolios.csv"
        if not path.exists():
            st.error(f"❌ No se encontró el archivo: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        return df
    
    @st.cache_data(ttl=3600)
    def load_segments(_self) -> pd.DataFrame:
        """
        Load segment/cluster information.
        
        Returns:
            DataFrame with segment statistics
        """
        path = _self.data_path / "segments.csv"
        if not path.exists():
            st.error(f"❌ No se encontró el archivo: {path}")
            return pd.DataFrame()
        
        return pd.read_csv(path)
    
    @st.cache_data(ttl=3600)
    def load_backtest_summary(_self) -> pd.DataFrame:
        """
        Load backtest summary metrics per profile.
        
        Returns:
            DataFrame with performance metrics vs benchmark
        """
        path = _self.data_path / "backtest_summary.csv"
        if not path.exists():
            st.error(f"❌ No se encontró el archivo: {path}")
            return pd.DataFrame()
        
        return pd.read_csv(path)
    
    @st.cache_data(ttl=3600)
    def load_equity_curves(_self) -> pd.DataFrame:
        """
        Load equity curves for all profiles.
        
        Returns:
            DataFrame with columns: fecha, equity_portafolio, 
            equity_benchmark, perfil
        """
        path = _self.data_path / "equity_curves.csv"
        if not path.exists():
            st.error(f"❌ No se encontró el archivo: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df
    
    @st.cache_data(ttl=3600)
    def load_metadata(_self) -> Dict:
        """
        Load pipeline metadata.
        
        Returns:
            Dictionary with pipeline execution information
        """
        path = _self.data_path / "metadata.json"
        if not path.exists():
            return {
                "project": "Portfolio Construction via Clustering",
                "version": "1.0",
                "benchmark": "SPY",
                "capital_inicial": 10000
            }
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_available_profiles(self) -> list:
        """
        Get list of available investor profiles.
        
        Returns:
            List of profile names
        """
        portfolios = self.load_portfolios()
        if portfolios.empty:
            return ['conservador', 'moderado', 'agresivo', 'especulativo', 'normal']
        return portfolios['perfil'].unique().tolist()
    
    def get_test_period_years(self) -> float:
        """
        Calculate years of data in test period.
        
        Returns:
            Number of years in test period
        """
        equity_curves = self.load_equity_curves()
        if equity_curves.empty:
            return 2.0  # Default
        
        min_date = equity_curves['fecha'].min()
        max_date = equity_curves['fecha'].max()
        days = (max_date - min_date).days
        return days / 365.25
    
    def get_portfolio_for_profile(self, profile: str) -> pd.DataFrame:
        """
        Get portfolio allocation for specific profile.
        
        Args:
            profile: Profile name (conservador, moderado, etc.)
            
        Returns:
            DataFrame with assets for that profile
        """
        portfolios = self.load_portfolios()
        if portfolios.empty:
            return pd.DataFrame()
        
        return portfolios[portfolios['perfil'] == profile].copy()

"""
Data Loader Module - Handles loading and caching of pipeline outputs
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import streamlit as st


class DataLoader:
    """
    Centralized data loading with Streamlit caching.
    Loads pre-computed outputs from pipeline in reports/ directory.
    """
    
    PERFILES = ['conservador', 'moderado', 'normal', 'agresivo', 'especulativo']
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize DataLoader with path to reports directory.
        
        Args:
            data_path: Path to reports directory. Defaults to relative path.
        """
        if data_path is None:
            # Try multiple possible paths - looking for reports/
            # Order: Streamlit Cloud paths first, then local development paths
            possible_paths = [
                # Streamlit Cloud deployment paths
                Path("/mount/src/stocks_portfolio_selector/reports"),
                Path("/mount/src/riskmanagement2025/reports"),
                # Working directory (useful for both local and cloud)
                Path.cwd() / "reports",
                # Relative to this file
                Path(__file__).parent.parent.parent / "reports",
                Path(__file__).parent.parent / "reports",
                # Simple relative paths
                Path("reports"),
                Path("../reports"),
            ]
            for p in possible_paths:
                if p.exists():
                    data_path = str(p)
                    break
            else:
                # Default fallback - will show clear error if not found
                data_path = str(Path.cwd() / "reports")
        
        self.data_path = Path(data_path)
    
    @st.cache_data(ttl=3600)
    def load_portfolio(_self, perfil: str) -> pd.DataFrame:
        """Load portfolio for a specific profile."""
        path = _self.data_path / f"portafolio_{perfil}.csv"
        if not path.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        # Normalize column names
        if 'segmento' not in df.columns and 'segmento_nombre' in df.columns:
            df['segmento'] = df['segmento_nombre']
        return df
    
    @st.cache_data(ttl=3600)
    def load_portfolios(_self) -> pd.DataFrame:
        """Load all portfolios combined."""
        all_portfolios = []
        for perfil in _self.PERFILES:
            df = _self.load_portfolio(perfil)
            if not df.empty:
                df['perfil'] = perfil
                all_portfolios.append(df)
        
        if not all_portfolios:
            return pd.DataFrame()
        return pd.concat(all_portfolios, ignore_index=True)
    
    @st.cache_data(ttl=3600)
    def load_segments(_self) -> pd.DataFrame:
        """Load segments information."""
        path = _self.data_path / "reporte_final_segmentos.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
    
    @st.cache_data(ttl=3600)
    def load_backtest_metrics(_self, perfil: str) -> pd.DataFrame:
        """Load backtest metrics for a profile."""
        path = _self.data_path / f"backtest_metricas_{perfil}.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
    
    @st.cache_data(ttl=3600)
    def load_backtest_summary(_self) -> pd.DataFrame:
        """Load combined backtest metrics for all profiles."""
        # Try loading from backtest_summary.csv first (has all metrics)
        # Check in outputs/api/ directory
        api_path = _self.data_path.parent / "outputs" / "api" / "backtest_summary.csv"
        if api_path.exists():
            return pd.read_csv(api_path)

        # Fallback to reporte_final_metricas.csv
        path = _self.data_path / "reporte_final_metricas.csv"
        if path.exists():
            return pd.read_csv(path)
        
        # Fallback: combine individual files
        all_metrics = []
        for perfil in _self.PERFILES:
            df = _self.load_backtest_metrics(perfil)
            if not df.empty:
                df['perfil'] = perfil
                all_metrics.append(df)
        
        if not all_metrics:
            return pd.DataFrame()
        return pd.concat(all_metrics, ignore_index=True)
    
    @st.cache_data(ttl=3600)
    def load_equity_curves(_self, perfil: str = None) -> pd.DataFrame:
        """Load equity curves for a profile or all."""
        if perfil:
            path = _self.data_path / f"backtest_equity_curves_{perfil}.csv"
            if not path.exists():
                return pd.DataFrame()
            df = pd.read_csv(path)
        else:
            # Load all profiles
            all_curves = []
            for p in _self.PERFILES:
                df = _self.load_equity_curves(p)
                if not df.empty:
                    df['perfil'] = p
                    all_curves.append(df)
            if not all_curves:
                return pd.DataFrame()
            df = pd.concat(all_curves, ignore_index=True)
        
        # Set date as index
        date_cols = ['date', 'Date', 'fecha', 'Fecha', 'Unnamed: 0']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
        return df
    
    @st.cache_data(ttl=3600)
    def load_prices(_self) -> pd.DataFrame:
        """Load price matrix."""
        path = _self.data_path / "prices_matrix.csv"
        if not path.exists():
            alt_path = _self.data_path.parent / "data" / "prices_train.csv"
            if alt_path.exists():
                path = alt_path
            else:
                return pd.DataFrame()
        
        df = pd.read_csv(path)
        date_cols = ['date', 'Date', 'fecha', 'Fecha', 'Unnamed: 0']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
        return df
    
    @st.cache_data(ttl=3600)
    def load_metadata(_self) -> Dict:
        """Load pipeline metadata."""
        return {
            "project": "Portfolio Construction via Clustering",
            "version": "1.0",
            "benchmark": "SPY",
            "capital_inicial": 10000
        }
    
    def get_available_profiles(self) -> list:
        """Get list of profiles with available data."""
        available = []
        for perfil in self.PERFILES:
            path = self.data_path / f"portafolio_{perfil}.csv"
            if path.exists():
                available.append(perfil)
        return available if available else self.PERFILES
    
    def get_portfolio_for_profile(self, profile: str) -> pd.DataFrame:
        """Get portfolio for a specific profile."""
        return self.load_portfolio(profile)

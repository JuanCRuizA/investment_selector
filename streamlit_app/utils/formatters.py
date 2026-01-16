"""
Formatters Module - Formatting utilities for display
"""
import pandas as pd
import numpy as np
from typing import Any, Optional


class Formatters:
    """
    Centralized formatting utilities for consistent display.
    """
    
    @staticmethod
    def format_currency(value: float, currency: str = "USD") -> str:
        """Format number as currency."""
        if pd.isna(value):
            return "-"
        if currency == "USD":
            return f"${value:,.2f}"
        return f"{value:,.2f} {currency}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Format number as percentage."""
        if pd.isna(value):
            return "-"
        return f"{value * 100:.{decimals}f}%"
    
    @staticmethod
    def format_number(value: float, decimals: int = 2) -> str:
        """Format number with specified decimals."""
        if pd.isna(value):
            return "-"
        return f"{value:,.{decimals}f}"
    
    @staticmethod
    def format_sharpe(value: float) -> str:
        """Format Sharpe ratio with color indicator."""
        if pd.isna(value):
            return "-"
        return f"{value:.2f}"
    
    @staticmethod
    def get_sharpe_color(value: float) -> str:
        """Get color based on Sharpe ratio value."""
        if value >= 1.0:
            return "green"
        elif value >= 0.5:
            return "orange"
        else:
            return "red"
    
    @staticmethod
    def format_beta(value: float) -> str:
        """Format beta with interpretation."""
        if pd.isna(value):
            return "-"
        return f"{value:.2f}"
    
    @staticmethod
    def get_beta_interpretation(value: float) -> str:
        """Get interpretation of beta value."""
        if value < 0.8:
            return "Defensivo"
        elif value <= 1.2:
            return "Neutral"
        else:
            return "Agresivo"
    
    @staticmethod
    def format_drawdown(value: float) -> str:
        """Format drawdown value."""
        if pd.isna(value):
            return "-"
        return f"{value:.2%}"
    
    @staticmethod
    def format_weight(value: float) -> str:
        """Format portfolio weight."""
        if pd.isna(value):
            return "-"
        return f"{value:.1%}"
    
    @staticmethod
    def style_dataframe(df: pd.DataFrame, metric_columns: list = None):
        """
        Apply consistent styling to a dataframe.
        
        Args:
            df: DataFrame to style
            metric_columns: Columns containing metrics to format
            
        Returns:
            Styled DataFrame
        """
        styled = df.style
        
        # Format percentage columns
        pct_cols = [c for c in df.columns if any(x in c.lower() for x in 
                   ['return', 'volatility', 'drawdown', 'peso', 'weight'])]
        
        for col in pct_cols:
            if col in df.columns:
                styled = styled.format({col: '{:.2%}'})
        
        # Format ratio columns
        ratio_cols = [c for c in df.columns if any(x in c.lower() for x in 
                     ['sharpe', 'sortino', 'beta', 'alpha'])]
        
        for col in ratio_cols:
            if col in df.columns:
                styled = styled.format({col: '{:.2f}'})
        
        return styled
    
    @staticmethod
    def create_metrics_summary(portfolio_df: pd.DataFrame) -> dict:
        """
        Create summary metrics from portfolio.
        
        Args:
            portfolio_df: Portfolio DataFrame
            
        Returns:
            Dictionary of formatted summary metrics
        """
        return {
            'Total Activos': len(portfolio_df),
            'Retorno Esperado': Formatters.format_percentage(
                (portfolio_df['return_annualized'] * portfolio_df['peso']).sum()
            ),
            'Volatilidad Esperada': Formatters.format_percentage(
                (portfolio_df['volatility_annual'] * portfolio_df['peso']).sum()
            ),
            'Sharpe Ponderado': Formatters.format_sharpe(
                (portfolio_df['sharpe_ratio'] * portfolio_df['peso']).sum()
            ),
            'Beta Ponderado': Formatters.format_beta(
                (portfolio_df['beta'] * portfolio_df['peso']).sum()
            ),
        }


class ColorPalette:
    """Color palette for consistent styling."""
    
    # Profile colors
    CONSERVADOR = '#2E7D32'  # Green
    MODERADO = '#1565C0'     # Blue
    AGRESIVO = '#F57C00'     # Orange
    ESPECULATIVO = '#C62828' # Red
    NORMAL = '#7B1FA2'       # Purple
    
    # Segment colors
    OUTLIERS = '#E91E63'
    ALTO_RENDIMIENTO = '#FF9800'
    MODERADO_SEG = '#2196F3'
    CONSERVADOR_SEG = '#4CAF50'
    ESTABLE = '#9E9E9E'
    
    # Chart colors
    PORTFOLIO = '#1E88E5'
    BENCHMARK = '#757575'
    POSITIVE = '#4CAF50'
    NEGATIVE = '#F44336'
    NEUTRAL = '#9E9E9E'
    
    @classmethod
    def get_profile_color(cls, profile: str) -> str:
        """Get color for profile."""
        colors = {
            'conservador': cls.CONSERVADOR,
            'moderado': cls.MODERADO,
            'agresivo': cls.AGRESIVO,
            'especulativo': cls.ESPECULATIVO,
            'normal': cls.NORMAL,
        }
        return colors.get(profile.lower(), cls.NEUTRAL)
    
    @classmethod
    def get_segment_color(cls, segment: str) -> str:
        """Get color for segment."""
        colors = {
            'outliers': cls.OUTLIERS,
            'alto rendimiento': cls.ALTO_RENDIMIENTO,
            'moderado': cls.MODERADO_SEG,
            'conservador': cls.CONSERVADOR_SEG,
            'estable': cls.ESTABLE,
        }
        return colors.get(segment.lower(), cls.NEUTRAL)
    
    @classmethod
    def get_segment_colors_map(cls) -> dict:
        """Get full segment color mapping."""
        return {
            'Outliers': cls.OUTLIERS,
            'Alto Rendimiento': cls.ALTO_RENDIMIENTO,
            'Moderado': cls.MODERADO_SEG,
            'Conservador': cls.CONSERVADOR_SEG,
            'Estable': cls.ESTABLE,
        }

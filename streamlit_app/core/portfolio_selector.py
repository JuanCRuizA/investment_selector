"""
Portfolio Selector Module - Logic for selecting optimal portfolios
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class InvestorProfile:
    """Configuration for each investor profile"""
    name: str
    display_name: str
    description: str
    risk_level: int  # 1-5
    color: str
    icon: str
    cluster_distribution: Dict[str, float]
    expected_return: Tuple[float, float]  # (min, max)
    expected_volatility: Tuple[float, float]


# Profile definitions
PROFILE_CONFIGS = {
    'conservador': InvestorProfile(
        name='conservador',
        display_name='Conservador',
        description='Prioriza la preservación de capital con volatilidad mínima',
        risk_level=1,
        color='#2E7D32',
        icon='',
        cluster_distribution={'Estable': 0.50, 'Conservador': 0.30, 'Moderado': 0.20},
        expected_return=(0.02, 0.08),
        expected_volatility=(0.05, 0.15)
    ),
    'moderado': InvestorProfile(
        name='moderado',
        display_name='Moderado',
        description='Balance entre crecimiento y estabilidad',
        risk_level=2,
        color='#1565C0',
        icon='',
        cluster_distribution={'Alto Rendimiento': 0.40, 'Moderado': 0.30, 'Estable': 0.30},
        expected_return=(0.05, 0.12),
        expected_volatility=(0.10, 0.20)
    ),
    'agresivo': InvestorProfile(
        name='agresivo',
        display_name='Agresivo',
        description='Busca rendimientos superiores aceptando mayor volatilidad',
        risk_level=3,
        color='#F57C00',
        icon='',
        cluster_distribution={'Alto Rendimiento': 0.70, 'Moderado': 0.20, 'Outliers': 0.10},
        expected_return=(0.10, 0.25),
        expected_volatility=(0.15, 0.30)
    ),
    'especulativo': InvestorProfile(
        name='especulativo',
        display_name='Especulativo',
        description='Máximo potencial de rendimiento con alta volatilidad',
        risk_level=4,
        color='#C62828',
        icon='',
        cluster_distribution={'Alto Rendimiento': 0.50, 'Outliers': 0.30, 'Moderado': 0.20},
        expected_return=(0.15, 0.50),
        expected_volatility=(0.25, 0.50)
    ),
    'normal': InvestorProfile(
        name='normal',
        display_name='Normal (Balanceado)',
        description='Distribución equilibrada entre todos los segmentos',
        risk_level=2,
        color='#7B1FA2',
        icon='',
        cluster_distribution={'Outliers': 0.20, 'Conservador': 0.20, 'Alto Rendimiento': 0.20, 
                             'Moderado': 0.20, 'Estable': 0.20},
        expected_return=(0.08, 0.18),
        expected_volatility=(0.12, 0.25)
    ),
}


class PortfolioSelector:
    """
    Handles portfolio selection logic based on investor profile.
    Uses pre-computed portfolios and adjusts based on investment parameters.
    """
    
    def __init__(self, portfolios_df: pd.DataFrame, segments_df: pd.DataFrame):
        """
        Initialize selector with data.
        
        Args:
            portfolios_df: Pre-computed portfolios from pipeline
            segments_df: Segment information
        """
        self.portfolios = portfolios_df
        self.segments = segments_df
    
    def get_profile_config(self, profile: str) -> InvestorProfile:
        """Get configuration for a profile."""
        return PROFILE_CONFIGS.get(profile, PROFILE_CONFIGS['moderado'])
    
    def seleccionar_portafolio(self, perfil: str, n_activos: int = 10) -> pd.DataFrame:
        """
        Selecciona el portafolio para un perfil de inversor.
        
        Args:
            perfil: Nombre del perfil ('conservador', 'moderado', 'agresivo', 
                   'especulativo', 'normal')
            n_activos: Número de activos a incluir (default=10)
            
        Returns:
            DataFrame con columnas: ticker, segmento, peso
        """
        if self.portfolios is None or self.portfolios.empty:
            return pd.DataFrame(columns=['ticker', 'segmento', 'peso'])
        
        # Filtrar por perfil
        portfolio_base = self.portfolios[self.portfolios['perfil'] == perfil].copy()
        
        if portfolio_base.empty:
            # Intentar con lowercase
            portfolio_base = self.portfolios[self.portfolios['perfil'].str.lower() == perfil.lower()].copy()
        
        if portfolio_base.empty:
            return pd.DataFrame(columns=['ticker', 'segmento', 'peso'])
        
        # Seleccionar top N activos
        if 'score_compuesto' in portfolio_base.columns:
            portfolio_final = portfolio_base.nlargest(n_activos, 'score_compuesto')
        else:
            portfolio_final = portfolio_base.head(n_activos)
        
        # Preparar resultado con columnas esperadas
        result = pd.DataFrame({
            'ticker': portfolio_final['ticker'],
            'segmento': portfolio_final.get('segmento_nombre', portfolio_final.get('segmento', 'N/A')),
            'peso': portfolio_final['peso']
        })
        
        return result.reset_index(drop=True)
    
    def get_available_profiles(self) -> List[str]:
        """Retorna lista de perfiles disponibles."""
        if self.portfolios is None or self.portfolios.empty:
            return list(PROFILE_CONFIGS.keys())
        return self.portfolios['perfil'].unique().tolist()

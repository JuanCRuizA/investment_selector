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
        icon='🛡️',
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
        icon='⚖️',
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
        icon='🚀',
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
        icon='🎯',
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
        icon='📊',
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
    
    def seleccionar_portafolio(
        self,
        perfil: str,
        monto_inversion: float,
        horizonte_anos: Optional[float] = None,
        n_activos: int = 10
    ) -> pd.DataFrame:
        """
        Selecciona y configura el portafolio óptimo para un perfil de inversor.
        
        Esta función implementa la lógica de selección basada en:
        1. Perfil de riesgo del inversor
        2. Distribución de clusters según perfil
        3. Score compuesto (Sharpe + momentum)
        4. Horizonte de inversión (ajusta exposición a volatilidad)
        
        Args:
            perfil: Nombre del perfil ('conservador', 'moderado', 'agresivo', 
                   'especulativo', 'normal')
            monto_inversion: Monto total a invertir en USD
            horizonte_anos: Horizonte de inversión en años (opcional)
            n_activos: Número de activos a incluir (default=10)
            
        Returns:
            DataFrame con columnas:
            - ticker: Símbolo del activo
            - segmento_nombre: Cluster/segmento del activo
            - peso: Peso en el portafolio (0-1)
            - monto_invertido: Monto en USD por activo
            - return_annualized: Retorno anualizado esperado
            - volatility_annual: Volatilidad anualizada
            - sharpe_ratio: Ratio de Sharpe
            - beta: Beta vs benchmark
            - momentum_6m: Momentum 6 meses
            - score_compuesto: Score de selección
        """
        # 1. Filtrar portafolio pre-computado para el perfil
        portfolio_base = self.portfolios[self.portfolios['perfil'] == perfil].copy()
        
        if portfolio_base.empty:
            raise ValueError(f"No se encontró portafolio para perfil: {perfil}")
        
        # 2. Obtener configuración del perfil
        config = self.get_profile_config(perfil)
        
        # 3. Si se especifica horizonte, ajustar selección
        if horizonte_anos is not None:
            portfolio_base = self._ajustar_por_horizonte(
                portfolio_base, horizonte_anos, config
            )
        
        # 4. Seleccionar top N activos por score
        portfolio_final = portfolio_base.nlargest(n_activos, 'score_compuesto')
        
        # 5. Recalcular pesos (equal weight por defecto)
        n_selected = len(portfolio_final)
        portfolio_final['peso'] = 1.0 / n_selected
        
        # 6. Calcular montos de inversión
        portfolio_final['monto_invertido'] = portfolio_final['peso'] * monto_inversion
        
        # 7. Ordenar por peso/monto
        portfolio_final = portfolio_final.sort_values('monto_invertido', ascending=False)
        
        # 8. Seleccionar y ordenar columnas para output
        columns_output = [
            'ticker', 'segmento_nombre', 'peso', 'monto_invertido',
            'return_annualized', 'volatility_annual', 'sharpe_ratio',
            'beta', 'momentum_6m', 'score_compuesto'
        ]
        
        return portfolio_final[columns_output].reset_index(drop=True)
    
    def _ajustar_por_horizonte(
        self,
        portfolio: pd.DataFrame,
        horizonte_anos: float,
        config: InvestorProfile
    ) -> pd.DataFrame:
        """
        Ajusta la selección según horizonte de inversión.
        
        - Horizonte corto (<2 años): Penaliza alta volatilidad
        - Horizonte medio (2-5 años): Balance normal
        - Horizonte largo (>5 años): Favorece alto rendimiento
        
        Args:
            portfolio: DataFrame del portafolio
            horizonte_anos: Años de inversión
            config: Configuración del perfil
            
        Returns:
            DataFrame con scores ajustados
        """
        portfolio = portfolio.copy()
        
        if horizonte_anos < 2:
            # Horizonte corto: penalizar volatilidad
            volatility_penalty = portfolio['volatility_annual'] * 0.3
            portfolio['score_compuesto'] = portfolio['score_compuesto'] - volatility_penalty
        elif horizonte_anos > 5:
            # Horizonte largo: bonus a alto rendimiento
            return_bonus = portfolio['return_annualized'] * 0.2
            portfolio['score_compuesto'] = portfolio['score_compuesto'] + return_bonus
        
        return portfolio
    
    def get_portfolio_summary(self, portfolio: pd.DataFrame) -> Dict:
        """
        Calcula métricas resumen del portafolio seleccionado.
        
        Args:
            portfolio: DataFrame del portafolio
            
        Returns:
            Dictionary con métricas agregadas
        """
        return {
            'n_activos': len(portfolio),
            'monto_total': portfolio['monto_invertido'].sum(),
            'return_esperado': (portfolio['return_annualized'] * portfolio['peso']).sum(),
            'volatility_esperada': self._calc_portfolio_vol(portfolio),
            'sharpe_ponderado': (portfolio['sharpe_ratio'] * portfolio['peso']).sum(),
            'beta_ponderado': (portfolio['beta'] * portfolio['peso']).sum(),
            'segmentos': portfolio['segmento_nombre'].value_counts().to_dict()
        }
    
    def _calc_portfolio_vol(self, portfolio: pd.DataFrame) -> float:
        """
        Calcula volatilidad aproximada del portafolio.
        Usa aproximación simple (promedio ponderado).
        """
        return (portfolio['volatility_annual'] * portfolio['peso']).sum()
    
    def compare_profiles(
        self,
        monto_inversion: float,
        horizonte_anos: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Compara todos los perfiles para un monto dado.
        
        Args:
            monto_inversion: Monto a invertir
            horizonte_anos: Horizonte opcional
            
        Returns:
            DataFrame comparativo de todos los perfiles
        """
        comparisons = []
        
        for profile_name in PROFILE_CONFIGS.keys():
            try:
                portfolio = self.seleccionar_portafolio(
                    profile_name, monto_inversion, horizonte_anos
                )
                summary = self.get_portfolio_summary(portfolio)
                
                config = self.get_profile_config(profile_name)
                
                comparisons.append({
                    'Perfil': config.display_name,
                    'Nivel Riesgo': '⭐' * config.risk_level,
                    'N° Activos': summary['n_activos'],
                    'Retorno Esperado': f"{summary['return_esperado']:.2%}",
                    'Volatilidad': f"{summary['volatility_esperada']:.2%}",
                    'Sharpe': f"{summary['sharpe_ponderado']:.2f}",
                    'Beta': f"{summary['beta_ponderado']:.2f}",
                })
            except Exception as e:
                continue
        
        return pd.DataFrame(comparisons)

"""
Componente Backtest View - Visualización de resultados de backtesting.

Muestra:
- Curva de equity del portafolio vs benchmark
- Métricas de rendimiento
- Drawdown
- Retornos acumulados
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Literal

from core.data_loader import DataLoader
from core.calculations import PortfolioCalculations
from utils.formatters import Formatters, ColorPalette
from utils.charts import ChartFactory


def _render_metricas_backtest(metricas: dict, perfil: str):
    """Renderiza las métricas principales del backtest."""
    st.subheader("📈 Métricas de Rendimiento")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        retorno = metricas.get('retorno_total', 0)
        st.metric(
            label="Retorno Total",
            value=Formatters.format_percentage(retorno),
            delta=None,
            help="Retorno acumulado en el período"
        )
    
    with col2:
        cagr = metricas.get('cagr', 0)
        st.metric(
            label="CAGR",
            value=Formatters.format_percentage(cagr),
            help="Tasa de crecimiento anual compuesta"
        )
    
    with col3:
        volatilidad = metricas.get('volatilidad', 0)
        st.metric(
            label="Volatilidad",
            value=Formatters.format_percentage(volatilidad),
            help="Volatilidad anualizada"
        )
    
    with col4:
        sharpe = metricas.get('sharpe', 0)
        st.metric(
            label="Sharpe Ratio",
            value=Formatters.format_sharpe(sharpe),
            help="Retorno ajustado por riesgo"
        )
    
    # Segunda fila de métricas
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        max_dd = metricas.get('max_drawdown', 0)
        st.metric(
            label="Max Drawdown",
            value=Formatters.format_percentage(max_dd),
            help="Máxima caída desde el pico"
        )
    
    with col6:
        sortino = metricas.get('sortino', 0)
        st.metric(
            label="Sortino Ratio",
            value=Formatters.format_sharpe(sortino),
            help="Retorno ajustado por riesgo a la baja"
        )
    
    with col7:
        calmar = metricas.get('calmar', 0)
        st.metric(
            label="Calmar Ratio",
            value=Formatters.format_sharpe(calmar),
            help="CAGR / Max Drawdown"
        )
    
    with col8:
        win_rate = metricas.get('win_rate', 0)
        st.metric(
            label="Win Rate",
            value=Formatters.format_percentage(win_rate),
            help="Porcentaje de meses positivos"
        )


def _render_equity_curve(
    df_equity: pd.DataFrame,
    perfil: str,
    mostrar_benchmark: bool,
    tipo_grafico: Literal['linea', 'velas', 'ambos'],
    monto_inicial: float
):
    """Renderiza la curva de equity."""
    st.subheader("📊 Curva de Equity")
    
    # Preparar datos
    df = df_equity.copy()
    
    # Verificar columnas disponibles
    col_portafolio = 'portafolio' if 'portafolio' in df.columns else 'equity'
    col_benchmark = 'benchmark' if 'benchmark' in df.columns else 'spy'
    
    # Escalar a monto inicial si es necesario
    if df[col_portafolio].iloc[0] != monto_inicial:
        factor = monto_inicial / df[col_portafolio].iloc[0]
        df[col_portafolio] = df[col_portafolio] * factor
        if col_benchmark in df.columns and mostrar_benchmark:
            df[col_benchmark] = df[col_benchmark] * factor
    
    # Crear gráfico según tipo seleccionado
    if tipo_grafico == 'linea' or tipo_grafico == 'ambos':
        # Gráfico de línea
        series_dict = {
            f'Portafolio {perfil.title()}': df[col_portafolio]
        }
        
        if mostrar_benchmark and col_benchmark in df.columns:
            series_dict['SPY (Benchmark)'] = df[col_benchmark]
        
        colors = [ColorPalette.get_profile_color(perfil)]
        if mostrar_benchmark:
            colors.append('#666666')
        
        fig = ChartFactory.create_equity_curve(
            df_equity=pd.DataFrame(series_dict, index=df.index),
            title="Evolución del Portafolio",
            colors=colors
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if tipo_grafico == 'velas' or tipo_grafico == 'ambos':
        # Para velas, necesitamos datos OHLC
        # Si no hay datos OHLC, mostramos mensaje
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            fig_candle = ChartFactory.create_candlestick(
                df_ohlc=df,
                title="Velas del Portafolio"
            )
            st.plotly_chart(fig_candle, use_container_width=True)
        elif tipo_grafico == 'velas':
            st.info("📊 Datos OHLC no disponibles. Mostrando gráfico de línea.")
            # Fallback a línea
            series_dict = {f'Portafolio {perfil.title()}': df[col_portafolio]}
            fig = ChartFactory.create_equity_curve(
                df_equity=pd.DataFrame(series_dict, index=df.index),
                title="Evolución del Portafolio"
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_drawdown(df_equity: pd.DataFrame, perfil: str):
    """Renderiza el gráfico de drawdown."""
    st.subheader("📉 Drawdown")
    
    # Calcular drawdown
    col_portafolio = 'portafolio' if 'portafolio' in df_equity.columns else 'equity'
    
    # Calcular drawdown manualmente
    equity = df_equity[col_portafolio]
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    
    df_dd = pd.DataFrame({
        'Drawdown': drawdown
    }, index=df_equity.index)
    
    fig = ChartFactory.create_drawdown_chart(
        df_drawdown=df_dd,
        title="",
        color=ColorPalette.get_profile_color(perfil)
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_retornos_periodo(df_equity: pd.DataFrame, perfil: str):
    """Renderiza análisis de retornos por período."""
    col1, col2 = st.columns(2)
    
    col_portafolio = 'portafolio' if 'portafolio' in df_equity.columns else 'equity'
    
    with col1:
        st.subheader("📅 Retornos Mensuales")
        
        # Calcular retornos mensuales
        retornos = df_equity[col_portafolio].pct_change()
        df_monthly = retornos.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        ).dropna()
        
        if len(df_monthly) > 0:
            # Crear heatmap si hay suficientes datos
            fig = ChartFactory.create_monthly_returns_heatmap(
                df_returns=df_monthly.to_frame('retorno'),
                title=""
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay suficientes datos para el heatmap mensual")
    
    with col2:
        st.subheader("📊 Retornos Anuales")
        
        # Calcular retornos anuales
        retornos = df_equity[col_portafolio].pct_change()
        df_yearly = retornos.resample('Y').apply(
            lambda x: (1 + x).prod() - 1
        ).dropna()
        
        if len(df_yearly) > 0:
            fig = ChartFactory.create_annual_returns_bar(
                df_returns=df_yearly,
                title="",
                color=ColorPalette.get_profile_color(perfil)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay suficientes datos para retornos anuales")


def _extraer_metricas_de_summary(df_summary: pd.DataFrame, perfil: str) -> dict:
    """Extrae métricas del DataFrame de resumen."""
    try:
        # Buscar la fila del perfil
        df_perfil = df_summary[df_summary['perfil'].str.lower() == perfil.lower()]
        
        if df_perfil.empty:
            return {}
        
        row = df_perfil.iloc[0]
        
        # Mapear columnas según estructura de backtest_summary.csv
        metricas = {
            'retorno_total': row.get('retorno_total', row.get('total_return', 0)),
            'cagr': row.get('cagr', row.get('annual_return', 0)),
            'volatilidad': row.get('volatilidad', row.get('volatility', 0)),
            'sharpe': row.get('sharpe', row.get('sharpe_ratio', 0)),
            'max_drawdown': row.get('max_drawdown', 0),
            'sortino': row.get('sortino', row.get('sortino_ratio', 0)),
            'calmar': row.get('calmar', row.get('calmar_ratio', 0)),
            'win_rate': row.get('win_rate', 0),
        }
        
        return metricas
        
    except Exception as e:
        st.warning(f"Error extrayendo métricas: {e}")
        return {}


def render_backtest_view(
    perfil: str,
    monto_inversion: float,
    mostrar_benchmark: bool,
    tipo_grafico: Literal['linea', 'velas', 'ambos'],
    data_loader: DataLoader
) -> Optional[pd.DataFrame]:
    """
    Renderiza la vista completa de backtesting.
    
    Args:
        perfil: Nombre del perfil
        monto_inversion: Monto inicial
        mostrar_benchmark: Si mostrar SPY
        tipo_grafico: Tipo de gráfico a mostrar
        data_loader: Instancia de DataLoader
        
    Returns:
        DataFrame con equity curve o None si hay error
    """
    st.header(f"🔬 Backtesting - {perfil.title()}")
    
    try:
        # Cargar datos de backtest
        df_summary = data_loader.load_backtest_summary()
        df_equity = data_loader.load_equity_curves(perfil)
        
        if df_summary is None or df_summary.empty:
            st.warning("No hay datos de resumen de backtesting disponibles.")
            return None
        
        # Extraer métricas
        metricas = _extraer_metricas_de_summary(df_summary, perfil)
        
        # Renderizar métricas
        if metricas:
            _render_metricas_backtest(metricas, perfil)
        else:
            st.info("Métricas de rendimiento no disponibles")
        
        st.divider()
        
        # Curva de equity
        if df_equity is not None and not df_equity.empty:
            _render_equity_curve(
                df_equity, 
                perfil, 
                mostrar_benchmark, 
                tipo_grafico,
                monto_inversion
            )
            
            st.divider()
            
            # Drawdown
            _render_drawdown(df_equity, perfil)
            
            st.divider()
            
            # Retornos por período
            with st.expander("📈 Análisis de Retornos por Período", expanded=False):
                _render_retornos_periodo(df_equity, perfil)
            
            return df_equity
        else:
            st.warning(f"No hay datos de equity curve para el perfil: {perfil}")
            return None
        
    except Exception as e:
        st.error(f"Error en backtesting: {str(e)}")
        st.exception(e)
        return None


def render_backtest_comparison(
    perfil1: str,
    perfil2: str,
    monto_inversion: float,
    data_loader: DataLoader
):
    """
    Renderiza comparación de backtest entre dos perfiles.
    
    Args:
        perfil1: Primer perfil
        perfil2: Segundo perfil
        monto_inversion: Monto inicial
        data_loader: Instancia de DataLoader
    """
    st.subheader("📊 Comparación de Rendimiento")
    
    try:
        # Cargar equity curves de ambos perfiles
        df_eq1 = data_loader.load_equity_curves(perfil1)
        df_eq2 = data_loader.load_equity_curves(perfil2)
        
        if df_eq1 is None or df_eq2 is None:
            st.warning("No hay datos suficientes para la comparación")
            return
        
        # Determinar columna de equity
        col1 = 'portafolio' if 'portafolio' in df_eq1.columns else 'equity'
        col2 = 'portafolio' if 'portafolio' in df_eq2.columns else 'equity'
        
        # Normalizar a monto inicial
        eq1 = df_eq1[col1] / df_eq1[col1].iloc[0] * monto_inversion
        eq2 = df_eq2[col2] / df_eq2[col2].iloc[0] * monto_inversion
        
        # Crear DataFrame combinado
        df_combined = pd.DataFrame({
            perfil1.title(): eq1,
            perfil2.title(): eq2
        })
        
        # Gráfico comparativo
        colors = [
            ColorPalette.get_profile_color(perfil1),
            ColorPalette.get_profile_color(perfil2)
        ]
        
        fig = ChartFactory.create_equity_curve(
            df_equity=df_combined,
            title="Comparación de Equity Curves",
            colors=colors
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla comparativa de métricas
        df_summary = data_loader.load_backtest_summary()
        if df_summary is not None:
            metricas1 = _extraer_metricas_de_summary(df_summary, perfil1)
            metricas2 = _extraer_metricas_de_summary(df_summary, perfil2)
            
            if metricas1 and metricas2:
                st.subheader("📋 Comparación de Métricas")
                
                df_comp = pd.DataFrame({
                    'Métrica': ['Retorno Total', 'CAGR', 'Volatilidad', 'Sharpe', 'Max Drawdown'],
                    perfil1.title(): [
                        Formatters.format_percentage(metricas1.get('retorno_total', 0)),
                        Formatters.format_percentage(metricas1.get('cagr', 0)),
                        Formatters.format_percentage(metricas1.get('volatilidad', 0)),
                        Formatters.format_sharpe(metricas1.get('sharpe', 0)),
                        Formatters.format_percentage(metricas1.get('max_drawdown', 0)),
                    ],
                    perfil2.title(): [
                        Formatters.format_percentage(metricas2.get('retorno_total', 0)),
                        Formatters.format_percentage(metricas2.get('cagr', 0)),
                        Formatters.format_percentage(metricas2.get('volatilidad', 0)),
                        Formatters.format_sharpe(metricas2.get('sharpe', 0)),
                        Formatters.format_percentage(metricas2.get('max_drawdown', 0)),
                    ],
                })
                
                st.dataframe(df_comp, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error en comparación: {str(e)}")

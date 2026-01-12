"""
Componente Comparison View - Comparación entre perfiles de inversión.

Muestra:
- Comparación lado a lado de dos perfiles
- Métricas comparativas
- Gráficos superpuestos
- Análisis de diferencias
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Tuple

from core.data_loader import DataLoader
from core.portfolio_selector import PortfolioSelector
from utils.formatters import Formatters, ColorPalette
from utils.charts import ChartFactory
from components.sidebar import PERFILES_DISPONIBLES


def _render_header_comparacion(perfil1: str, perfil2: str):
    """Renderiza el header de la comparación."""
    col1, col2, col3 = st.columns([2, 1, 2])
    
    info1 = PERFILES_DISPONIBLES.get(perfil1, {})
    info2 = PERFILES_DISPONIBLES.get(perfil2, {})
    
    with col1:
        st.markdown(f"### {info1.get('nombre', perfil1.title())}")
        st.caption(info1.get('descripcion', ''))
    
    with col2:
        st.markdown("### ⚔️")
        st.caption("vs")
    
    with col3:
        st.markdown(f"### {info2.get('nombre', perfil2.title())}")
        st.caption(info2.get('descripcion', ''))


def _render_comparacion_composicion(
    df_port1: pd.DataFrame,
    df_port2: pd.DataFrame,
    perfil1: str,
    perfil2: str,
    monto_inversion: float
):
    """Renderiza comparación de composición de portafolios."""
    st.subheader("📋 Comparación de Composición")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{perfil1.title()}**")
        df1 = df_port1.copy()
        df1['monto'] = df1['peso'] * monto_inversion
        df_display1 = pd.DataFrame({
            'Ticker': df1['ticker'],
            'Segmento': df1['segmento'].apply(lambda x: f"Seg. {x}"),
            'Peso': df1['peso'].apply(lambda x: f"{x*100:.1f}%"),
            'Monto': df1['monto'].apply(lambda x: f"${x:,.0f}"),
        })
        st.dataframe(df_display1, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown(f"**{perfil2.title()}**")
        df2 = df_port2.copy()
        df2['monto'] = df2['peso'] * monto_inversion
        df_display2 = pd.DataFrame({
            'Ticker': df2['ticker'],
            'Segmento': df2['segmento'].apply(lambda x: f"Seg. {x}"),
            'Peso': df2['peso'].apply(lambda x: f"{x*100:.1f}%"),
            'Monto': df2['monto'].apply(lambda x: f"${x:,.0f}"),
        })
        st.dataframe(df_display2, use_container_width=True, hide_index=True)
    
    # Análisis de diferencias
    st.divider()
    
    tickers1 = set(df_port1['ticker'].tolist())
    tickers2 = set(df_port2['ticker'].tolist())
    
    comunes = tickers1.intersection(tickers2)
    solo_p1 = tickers1 - tickers2
    solo_p2 = tickers2 - tickers1
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Activos en Común", len(comunes))
        if comunes:
            st.caption(", ".join(sorted(comunes)[:5]) + ("..." if len(comunes) > 5 else ""))
    
    with col2:
        st.metric(f"Solo en {perfil1.title()}", len(solo_p1))
        if solo_p1:
            st.caption(", ".join(sorted(solo_p1)))
    
    with col3:
        st.metric(f"Solo en {perfil2.title()}", len(solo_p2))
        if solo_p2:
            st.caption(", ".join(sorted(solo_p2)))


def _render_comparacion_metricas(
    df_summary: pd.DataFrame,
    perfil1: str,
    perfil2: str
):
    """Renderiza tabla comparativa de métricas."""
    st.subheader("📊 Comparación de Métricas")
    
    def get_metricas(perfil):
        df_p = df_summary[df_summary['perfil'].str.lower() == perfil.lower()]
        if df_p.empty:
            return {}
        row = df_p.iloc[0]
        return {
            'retorno_total': row.get('retorno_total', row.get('total_return', 0)),
            'cagr': row.get('cagr', row.get('annual_return', 0)),
            'volatilidad': row.get('volatilidad', row.get('volatility', 0)),
            'sharpe': row.get('sharpe', row.get('sharpe_ratio', 0)),
            'max_drawdown': row.get('max_drawdown', 0),
            'sortino': row.get('sortino', row.get('sortino_ratio', 0)),
        }
    
    m1 = get_metricas(perfil1)
    m2 = get_metricas(perfil2)
    
    if not m1 or not m2:
        st.warning("Métricas no disponibles para comparación")
        return
    
    # Crear tabla comparativa con indicadores
    metricas_nombres = [
        ('retorno_total', 'Retorno Total', 'higher'),
        ('cagr', 'CAGR', 'higher'),
        ('volatilidad', 'Volatilidad', 'lower'),
        ('sharpe', 'Sharpe Ratio', 'higher'),
        ('max_drawdown', 'Max Drawdown', 'lower'),
        ('sortino', 'Sortino Ratio', 'higher'),
    ]
    
    data = []
    for key, nombre, mejor in metricas_nombres:
        v1 = m1.get(key, 0)
        v2 = m2.get(key, 0)
        
        # Determinar quién es mejor
        if mejor == 'higher':
            ganador = perfil1 if v1 > v2 else (perfil2 if v2 > v1 else 'empate')
        else:
            ganador = perfil1 if v1 < v2 else (perfil2 if v2 < v1 else 'empate')
        
        # Formatear valores
        if key in ['sharpe', 'sortino']:
            v1_fmt = Formatters.format_sharpe(v1)
            v2_fmt = Formatters.format_sharpe(v2)
        else:
            v1_fmt = Formatters.format_percentage(v1)
            v2_fmt = Formatters.format_percentage(v2)
        
        # Agregar indicador
        ind1 = "✅" if ganador == perfil1 else ""
        ind2 = "✅" if ganador == perfil2 else ""
        
        data.append({
            'Métrica': nombre,
            perfil1.title(): f"{v1_fmt} {ind1}",
            perfil2.title(): f"{v2_fmt} {ind2}",
        })
    
    df_comp = pd.DataFrame(data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)


def _render_comparacion_equity(
    df_eq1: pd.DataFrame,
    df_eq2: pd.DataFrame,
    perfil1: str,
    perfil2: str,
    monto_inversion: float
):
    """Renderiza comparación de equity curves."""
    st.subheader("📈 Comparación de Rendimiento")
    
    # Determinar columnas
    col1 = 'portafolio' if 'portafolio' in df_eq1.columns else 'equity'
    col2 = 'portafolio' if 'portafolio' in df_eq2.columns else 'equity'
    
    # Normalizar a monto inicial
    eq1 = df_eq1[col1] / df_eq1[col1].iloc[0] * monto_inversion
    eq2 = df_eq2[col2] / df_eq2[col2].iloc[0] * monto_inversion
    
    # Combinar
    df_combined = pd.DataFrame({
        perfil1.title(): eq1,
        perfil2.title(): eq2
    })
    
    colors = [
        ColorPalette.get_profile_color(perfil1),
        ColorPalette.get_profile_color(perfil2)
    ]
    
    fig = ChartFactory.create_equity_curve(
        df_equity=df_combined,
        title="Evolución Comparativa del Portafolio",
        colors=colors
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Métricas finales
    col1, col2 = st.columns(2)
    
    valor_final1 = eq1.iloc[-1]
    valor_final2 = eq2.iloc[-1]
    ganancia1 = valor_final1 - monto_inversion
    ganancia2 = valor_final2 - monto_inversion
    
    with col1:
        st.metric(
            f"Valor Final - {perfil1.title()}",
            Formatters.format_currency(valor_final1),
            Formatters.format_currency(ganancia1)
        )
    
    with col2:
        st.metric(
            f"Valor Final - {perfil2.title()}",
            Formatters.format_currency(valor_final2),
            Formatters.format_currency(ganancia2)
        )


def _render_comparacion_riesgo(
    df_eq1: pd.DataFrame,
    df_eq2: pd.DataFrame,
    perfil1: str,
    perfil2: str
):
    """Renderiza comparación de métricas de riesgo."""
    st.subheader("📉 Comparación de Riesgo (Drawdown)")
    
    col1 = 'portafolio' if 'portafolio' in df_eq1.columns else 'equity'
    col2 = 'portafolio' if 'portafolio' in df_eq2.columns else 'equity'
    
    # Calcular drawdowns
    eq1 = df_eq1[col1]
    eq2 = df_eq2[col2]
    
    rm1 = eq1.cummax()
    rm2 = eq2.cummax()
    
    dd1 = (eq1 - rm1) / rm1
    dd2 = (eq2 - rm2) / rm2
    
    df_dd = pd.DataFrame({
        perfil1.title(): dd1,
        perfil2.title(): dd2
    })
    
    colors = [
        ColorPalette.get_profile_color(perfil1),
        ColorPalette.get_profile_color(perfil2)
    ]
    
    fig = ChartFactory.create_drawdown_chart(
        df_drawdown=df_dd,
        title="Drawdown Comparativo",
        color=colors[0]  # Se ajustará para múltiples series
    )
    
    # Actualizar para mostrar ambas series
    import plotly.graph_objects as go
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_dd.index,
        y=df_dd[perfil1.title()] * 100,
        fill='tozeroy',
        name=perfil1.title(),
        line=dict(color=colors[0])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_dd.index,
        y=df_dd[perfil2.title()] * 100,
        fill='tozeroy',
        name=perfil2.title(),
        line=dict(color=colors[1])
    ))
    
    fig.update_layout(
        title="Drawdown Comparativo",
        xaxis_title="Fecha",
        yaxis_title="Drawdown (%)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_radar_comparativo(
    df_summary: pd.DataFrame,
    perfil1: str,
    perfil2: str
):
    """Renderiza gráfico radar comparativo."""
    st.subheader("🎯 Perfil de Riesgo-Retorno")
    
    def get_metricas_norm(perfil):
        df_p = df_summary[df_summary['perfil'].str.lower() == perfil.lower()]
        if df_p.empty:
            return None
        row = df_p.iloc[0]
        return {
            'Retorno': min(row.get('retorno_total', row.get('total_return', 0)) * 100, 100),
            'Sharpe': min(row.get('sharpe', row.get('sharpe_ratio', 0)) * 20, 100),
            'Estabilidad': max(100 - abs(row.get('max_drawdown', 0)) * 100, 0),
            'Consistencia': row.get('win_rate', 0.5) * 100,
            'Riesgo Ajustado': min(row.get('sortino', row.get('sortino_ratio', 0)) * 20, 100),
        }
    
    m1 = get_metricas_norm(perfil1)
    m2 = get_metricas_norm(perfil2)
    
    if not m1 or not m2:
        st.info("Datos insuficientes para el gráfico radar")
        return
    
    fig = ChartFactory.create_radar_chart(
        categories=list(m1.keys()),
        values_list=[list(m1.values()), list(m2.values())],
        names=[perfil1.title(), perfil2.title()],
        title=""
    )
    st.plotly_chart(fig, use_container_width=True)


def render_comparison_view(
    perfil1: str,
    perfil2: str,
    monto_inversion: float,
    data_loader: DataLoader,
    portfolio_selector: PortfolioSelector
):
    """
    Renderiza la vista completa de comparación.
    
    Args:
        perfil1: Primer perfil
        perfil2: Segundo perfil
        monto_inversion: Monto a invertir
        data_loader: Instancia de DataLoader
        portfolio_selector: Instancia de PortfolioSelector
    """
    st.header("🔄 Comparación de Perfiles")
    
    try:
        # Header
        _render_header_comparacion(perfil1, perfil2)
        
        st.divider()
        
        # Cargar datos
        df_port1 = portfolio_selector.seleccionar_portafolio(perfil1)
        df_port2 = portfolio_selector.seleccionar_portafolio(perfil2)
        df_summary = data_loader.load_backtest_summary()
        df_eq1 = data_loader.load_equity_curves(perfil1)
        df_eq2 = data_loader.load_equity_curves(perfil2)
        
        # Comparación de composición
        if df_port1 is not None and df_port2 is not None:
            _render_comparacion_composicion(df_port1, df_port2, perfil1, perfil2, monto_inversion)
        
        st.divider()
        
        # Comparación de métricas
        if df_summary is not None:
            _render_comparacion_metricas(df_summary, perfil1, perfil2)
        
        st.divider()
        
        # Comparación de equity
        if df_eq1 is not None and df_eq2 is not None:
            _render_comparacion_equity(df_eq1, df_eq2, perfil1, perfil2, monto_inversion)
            
            st.divider()
            
            # Comparación de riesgo
            _render_comparacion_riesgo(df_eq1, df_eq2, perfil1, perfil2)
        
        st.divider()
        
        # Radar comparativo
        if df_summary is not None:
            _render_radar_comparativo(df_summary, perfil1, perfil2)
        
    except Exception as e:
        st.error(f"Error en comparación: {str(e)}")
        st.exception(e)


def render_all_profiles_comparison(
    monto_inversion: float,
    data_loader: DataLoader
):
    """
    Renderiza comparación de todos los perfiles.
    
    Args:
        monto_inversion: Monto a invertir
        data_loader: Instancia de DataLoader
    """
    st.subheader("📊 Todos los Perfiles")
    
    df_summary = data_loader.load_backtest_summary()
    
    if df_summary is None or df_summary.empty:
        st.warning("No hay datos de comparación disponibles")
        return
    
    # Tabla de todos los perfiles
    df_display = df_summary.copy()
    
    # Formatear columnas
    cols_pct = ['retorno_total', 'cagr', 'volatilidad', 'max_drawdown', 'total_return', 'annual_return', 'volatility']
    cols_ratio = ['sharpe', 'sortino', 'calmar', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
    
    for col in df_display.columns:
        if col in cols_pct:
            df_display[col] = df_display[col].apply(Formatters.format_percentage)
        elif col in cols_ratio:
            df_display[col] = df_display[col].apply(Formatters.format_sharpe)
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

"""
Componente Metrics View - Métricas detalladas de activos individuales.

Muestra:
- Métricas por activo del portafolio
- Retornos mensuales y anuales
- Comparación con benchmark
- Análisis de riesgo individual
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List

from core.data_loader import DataLoader
from utils.formatters import Formatters, ColorPalette
from utils.charts import ChartFactory


def _calcular_metricas_activo(
    df_precios: pd.DataFrame,
    ticker: str,
    benchmark: str = 'SPY'
) -> dict:
    """
    Calcula métricas para un activo individual.
    
    Args:
        df_precios: DataFrame con precios
        ticker: Ticker del activo
        benchmark: Ticker del benchmark
        
    Returns:
        dict con métricas calculadas
    """
    if ticker not in df_precios.columns:
        return {}
    
    precios = df_precios[ticker].dropna()
    retornos = precios.pct_change().dropna()
    
    if len(retornos) < 20:
        return {}
    
    # Métricas básicas
    retorno_total = (precios.iloc[-1] / precios.iloc[0]) - 1
    
    # Anualizar
    n_years = len(retornos) / 252
    cagr = (1 + retorno_total) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    volatilidad = retornos.std() * np.sqrt(252)
    sharpe = cagr / volatilidad if volatilidad > 0 else 0
    
    # Drawdown
    rolling_max = precios.cummax()
    drawdown = (precios - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Sortino (solo volatilidad negativa)
    retornos_negativos = retornos[retornos < 0]
    downside_vol = retornos_negativos.std() * np.sqrt(252) if len(retornos_negativos) > 0 else 0
    sortino = cagr / downside_vol if downside_vol > 0 else 0
    
    # Beta vs benchmark
    beta = 0
    if benchmark in df_precios.columns:
        ret_benchmark = df_precios[benchmark].pct_change().dropna()
        # Alinear índices
        common_idx = retornos.index.intersection(ret_benchmark.index)
        if len(common_idx) > 20:
            ret_a = retornos.loc[common_idx]
            ret_b = ret_benchmark.loc[common_idx]
            cov = np.cov(ret_a, ret_b)[0, 1]
            var_b = np.var(ret_b)
            beta = cov / var_b if var_b > 0 else 0
    
    # Retornos mensuales
    ret_mensual = retornos.resample('M').apply(lambda x: (1 + x).prod() - 1)
    win_rate = (ret_mensual > 0).sum() / len(ret_mensual) if len(ret_mensual) > 0 else 0
    
    return {
        'ticker': ticker,
        'retorno_total': retorno_total,
        'cagr': cagr,
        'volatilidad': volatilidad,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'sortino': sortino,
        'beta': beta,
        'win_rate': win_rate,
        'n_dias': len(retornos),
    }


def _render_tabla_metricas_activos(df_metricas: pd.DataFrame):
    """Renderiza la tabla de métricas por activo."""
    st.subheader("📊 Métricas por Activo")
    
    # Formatear para display
    df_display = pd.DataFrame({
        'Ticker': df_metricas['ticker'],
        'Retorno Total': df_metricas['retorno_total'].apply(Formatters.format_percentage),
        'CAGR': df_metricas['cagr'].apply(Formatters.format_percentage),
        'Volatilidad': df_metricas['volatilidad'].apply(Formatters.format_percentage),
        'Sharpe': df_metricas['sharpe'].apply(Formatters.format_sharpe),
        'Max DD': df_metricas['max_drawdown'].apply(Formatters.format_percentage),
        'Beta': df_metricas['beta'].apply(Formatters.format_beta),
        'Win Rate': df_metricas['win_rate'].apply(Formatters.format_percentage),
    })
    
    # Ordenar por Sharpe descendente
    df_display = df_display.sort_values('Sharpe', ascending=False, key=lambda x: df_metricas['sharpe'])
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Ticker': st.column_config.TextColumn(width="small"),
            'Retorno Total': st.column_config.TextColumn(width="small"),
            'CAGR': st.column_config.TextColumn(width="small"),
            'Volatilidad': st.column_config.TextColumn(width="small"),
            'Sharpe': st.column_config.TextColumn(width="small"),
            'Max DD': st.column_config.TextColumn(width="small"),
            'Beta': st.column_config.TextColumn(width="small"),
            'Win Rate': st.column_config.TextColumn(width="small"),
        }
    )


def _render_selector_activo(tickers: List[str]) -> str:
    """Renderiza selector de activo individual."""
    return st.selectbox(
        "Seleccionar activo para análisis detallado",
        options=tickers,
        key='selector_activo_individual'
    )


def _render_detalle_activo(
    df_precios: pd.DataFrame,
    ticker: str,
    metricas: dict
):
    """Renderiza el detalle de un activo individual."""
    st.subheader(f"📈 Detalle: {ticker}")
    
    # Métricas en cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Retorno Total", Formatters.format_percentage(metricas.get('retorno_total', 0)))
    with col2:
        st.metric("CAGR", Formatters.format_percentage(metricas.get('cagr', 0)))
    with col3:
        st.metric("Volatilidad", Formatters.format_percentage(metricas.get('volatilidad', 0)))
    with col4:
        st.metric("Sharpe", Formatters.format_sharpe(metricas.get('sharpe', 0)))
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Max Drawdown", Formatters.format_percentage(metricas.get('max_drawdown', 0)))
    with col6:
        st.metric("Sortino", Formatters.format_sharpe(metricas.get('sortino', 0)))
    with col7:
        st.metric("Beta", Formatters.format_beta(metricas.get('beta', 0)))
    with col8:
        st.metric("Win Rate", Formatters.format_percentage(metricas.get('win_rate', 0)))
    
    st.divider()
    
    # Gráficos del activo
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**Precio Histórico**")
        if ticker in df_precios.columns:
            precios = df_precios[ticker].dropna()
            # Normalizar a 100
            precios_norm = precios / precios.iloc[0] * 100
            
            fig = ChartFactory.create_line_chart(
                df=pd.DataFrame({ticker: precios_norm}),
                title="",
                colors=['#1E88E5']
            )
            st.plotly_chart(fig, use_container_width=True, key=f'precio_{ticker}')
    
    with col_chart2:
        st.markdown("**Drawdown**")
        if ticker in df_precios.columns:
            precios = df_precios[ticker].dropna()
            rolling_max = precios.cummax()
            drawdown = (precios - rolling_max) / rolling_max

            # Crear gráfico de drawdown manualmente
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=precios.index,
                y=drawdown * 100,
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='#E53935'),
                fillcolor='rgba(229, 57, 53, 0.3)'
            ))
            fig.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Drawdown (%)",
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_retornos_mensuales_activo(df_precios: pd.DataFrame, ticker: str):
    """Renderiza los retornos mensuales de un activo."""
    if ticker not in df_precios.columns:
        return
    
    precios = df_precios[ticker].dropna()
    retornos = precios.pct_change().dropna()
    ret_mensual = retornos.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Retornos Mensuales (Heatmap)**")
        # Crear heatmap de retornos mensuales
        import plotly.graph_objects as go

        df_heat = ret_mensual.to_frame('retorno')
        df_heat['year'] = df_heat.index.year
        df_heat['month'] = df_heat.index.month

        pivot = df_heat.pivot(index='year', columns='month', values='retorno')
        month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                      'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values * 100,
            x=[month_names[m-1] for m in pivot.columns],
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            text=[[f'{v*100:.1f}%' if pd.notna(v) else '' for v in row] for row in pivot.values],
            texttemplate='%{text}',
            textfont={'size': 10},
            hovertemplate='Año: %{y}<br>Mes: %{x}<br>Retorno: %{z:.1f}%<extra></extra>'
        ))
        fig.update_layout(
            xaxis_title='Mes',
            yaxis_title='Año',
            height=300,
            margin=dict(l=50, r=20, t=30, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Distribución de Retornos**")
        # Histograma simple con plotly
        import plotly.express as px
        fig = px.histogram(
            ret_mensual,
            nbins=20,
            title="",
            labels={'value': 'Retorno Mensual', 'count': 'Frecuencia'}
        )
        fig.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_retornos_anuales_activo(df_precios: pd.DataFrame, ticker: str):
    """Renderiza los retornos anuales de un activo."""
    if ticker not in df_precios.columns:
        return
    
    precios = df_precios[ticker].dropna()
    retornos = precios.pct_change().dropna()
    ret_anual = retornos.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    
    st.markdown("**Retornos Anuales**")
    
    # Tabla de retornos anuales
    df_anual = pd.DataFrame({
        'Año': ret_anual.index.year,
        'Retorno': ret_anual.values
    })
    df_anual['Retorno Fmt'] = df_anual['Retorno'].apply(Formatters.format_percentage)
    
    # Crear gráfico de barras de retornos anuales
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ret_anual.index.year,
        y=ret_anual.values * 100,
        marker_color='#1E88E5',
        text=[f'{v*100:.1f}%' for v in ret_anual.values],
        textposition='auto',
        hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
    ))
    fig.update_layout(
        xaxis_title='Año',
        yaxis_title='Retorno (%)',
        height=300,
        margin=dict(l=50, r=20, t=30, b=50)
    )
    st.plotly_chart(fig, use_container_width=True)


def render_metrics_view(
    df_portafolio: pd.DataFrame,
    data_loader: DataLoader,
    perfil: str
) -> Optional[pd.DataFrame]:
    """
    Renderiza la vista de métricas detalladas.
    
    Args:
        df_portafolio: DataFrame con composición del portafolio
        data_loader: Instancia de DataLoader
        perfil: Nombre del perfil
        
    Returns:
        DataFrame con métricas o None si hay error
    """
    st.header(f"📊 Métricas Detalladas - {perfil.title()}")
    
    try:
        # Cargar precios
        df_precios = data_loader.load_prices()
        
        if df_precios is None or df_precios.empty:
            st.warning("No hay datos de precios disponibles")
            return None
        
        # Obtener tickers del portafolio
        tickers = df_portafolio['ticker'].tolist()
        
        # Calcular métricas para cada activo
        metricas_list = []
        for ticker in tickers:
            metricas = _calcular_metricas_activo(df_precios, ticker)
            if metricas:
                metricas_list.append(metricas)
        
        if not metricas_list:
            st.warning("No se pudieron calcular métricas para los activos")
            return None
        
        df_metricas = pd.DataFrame(metricas_list)
        
        # Tabla de métricas
        _render_tabla_metricas_activos(df_metricas)
        
        st.divider()
        
        # Selector de activo individual
        ticker_seleccionado = _render_selector_activo(tickers)
        
        if ticker_seleccionado:
            metricas_activo = df_metricas[df_metricas['ticker'] == ticker_seleccionado].iloc[0].to_dict()
            
            st.divider()
            
            # Detalle del activo
            _render_detalle_activo(df_precios, ticker_seleccionado, metricas_activo)
            
            # Retornos en expanders
            with st.expander("📅 Análisis de Retornos Mensuales", expanded=False):
                _render_retornos_mensuales_activo(df_precios, ticker_seleccionado)
            
            with st.expander("📊 Análisis de Retornos Anuales", expanded=False):
                _render_retornos_anuales_activo(df_precios, ticker_seleccionado)
        
        return df_metricas
        
    except Exception as e:
        st.error(f"Error calculando métricas: {str(e)}")
        st.exception(e)
        return None


def render_metrics_summary(df_metricas: pd.DataFrame, perfil: str):
    """
    Renderiza un resumen compacto de métricas.
    
    Args:
        df_metricas: DataFrame con métricas por activo
        perfil: Nombre del perfil
    """
    st.subheader("📊 Resumen de Métricas del Portafolio")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Mejor Performer**")
        mejor = df_metricas.loc[df_metricas['retorno_total'].idxmax()]
        st.metric(
            mejor['ticker'],
            Formatters.format_percentage(mejor['retorno_total']),
            f"Sharpe: {Formatters.format_sharpe(mejor['sharpe'])}"
        )
    
    with col2:
        st.markdown("**Menor Volatilidad**")
        menor_vol = df_metricas.loc[df_metricas['volatilidad'].idxmin()]
        st.metric(
            menor_vol['ticker'],
            Formatters.format_percentage(menor_vol['volatilidad']),
            f"Retorno: {Formatters.format_percentage(menor_vol['retorno_total'])}"
        )
    
    with col3:
        st.markdown("**Mejor Sharpe**")
        mejor_sharpe = df_metricas.loc[df_metricas['sharpe'].idxmax()]
        st.metric(
            mejor_sharpe['ticker'],
            Formatters.format_sharpe(mejor_sharpe['sharpe']),
            f"Vol: {Formatters.format_percentage(mejor_sharpe['volatilidad'])}"
        )

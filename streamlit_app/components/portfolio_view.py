"""
Componente Portfolio View - Visualización del portafolio seleccionado.

Muestra:
- Tabla de activos con pesos y montos
- Gráfico de distribución por activo
- Distribución por segmento/cluster
"""

import streamlit as st
import pandas as pd
from typing import Optional

from core.data_loader import DataLoader
from core.portfolio_selector import PortfolioSelector
from utils.formatters import Formatters, ColorPalette
from utils.charts import ChartFactory


def _crear_tabla_portafolio(
    df_portafolio: pd.DataFrame, 
    monto_inversion: float
) -> pd.DataFrame:
    """
    Crea la tabla formateada del portafolio.
    
    Args:
        df_portafolio: DataFrame con ticker, peso, segmento
        monto_inversion: Monto total a invertir
        
    Returns:
        DataFrame formateado para mostrar
    """
    # Calcular montos
    df = df_portafolio.copy()
    df['monto'] = df['peso'] * monto_inversion
    
    # Ordenar por peso descendente
    df = df.sort_values('peso', ascending=False).reset_index(drop=True)
    
    # Crear DataFrame formateado para display
    df_display = pd.DataFrame({
        '#': range(1, len(df) + 1),
        'Ticker': df['ticker'],
        'Segmento': df['segmento'].apply(lambda x: f"Seg. {x}"),
        'Peso (%)': df['peso'].apply(lambda x: f"{x*100:.1f}%"),
        'Monto (USD)': df['monto'].apply(lambda x: f"${x:,.2f}"),
    })
    
    return df_display, df


def _render_metricas_resumen(df_portafolio: pd.DataFrame, monto_inversion: float):
    """Renderiza las métricas resumen del portafolio."""
    n_activos = len(df_portafolio)
    n_segmentos = df_portafolio['segmento'].nunique()
    peso_max = df_portafolio['peso'].max() * 100
    peso_min = df_portafolio['peso'].min() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Activos",
            value=n_activos,
            help="Número de activos en el portafolio"
        )
    
    with col2:
        st.metric(
            label="Segmentos",
            value=n_segmentos,
            help="Diversificación por clusters"
        )
    
    with col3:
        st.metric(
            label="Peso Máximo",
            value=f"{peso_max:.1f}%",
            help="Concentración máxima en un activo"
        )
    
    with col4:
        st.metric(
            label="Inversión Total",
            value=Formatters.format_currency(monto_inversion),
            help="Monto total invertido"
        )


def _render_tabla_activos(df_display: pd.DataFrame):
    """Renderiza la tabla de activos."""
    st.subheader("📋 Composición del Portafolio")
    
    # Configurar columnas para mejor visualización
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            '#': st.column_config.NumberColumn(
                width="small"
            ),
            'Ticker': st.column_config.TextColumn(
                width="medium"
            ),
            'Segmento': st.column_config.TextColumn(
                width="small"
            ),
            'Peso (%)': st.column_config.TextColumn(
                width="small"
            ),
            'Monto (USD)': st.column_config.TextColumn(
                width="medium"
            ),
        }
    )


def _render_graficos_distribucion(df_portafolio: pd.DataFrame, perfil: str):
    """Renderiza los gráficos de distribución."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥧 Distribución por Activo")
        
        # Preparar datos para el pie chart
        df_pie = df_portafolio.copy()
        df_pie['peso_pct'] = df_pie['peso'] * 100
        
        fig_activos = ChartFactory.create_pie_chart(
            values=df_pie['peso_pct'].tolist(),
            labels=df_pie['ticker'].tolist(),
            title=None,
            hole=0.4
        )
        st.plotly_chart(fig_activos, use_container_width=True, key='pie_activos')
    
    with col2:
        st.subheader("📊 Distribución por Segmento")
        
        # Agrupar por segmento
        df_segmento = df_portafolio.groupby('segmento')['peso'].sum().reset_index()
        df_segmento['peso_pct'] = df_segmento['peso'] * 100
        df_segmento['label'] = df_segmento['segmento'].apply(lambda x: f"Segmento {x}")
        
        # Colores por segmento
        colors = [ColorPalette.get_segment_color(s) for s in df_segmento['segmento']]
        
        fig_segmentos = ChartFactory.create_pie_chart(
            values=df_segmento['peso_pct'].tolist(),
            labels=df_segmento['label'].tolist(),
            title=None,
            hole=0.4,
            colors=colors
        )
        st.plotly_chart(fig_segmentos, use_container_width=True, key='pie_segmentos')


def _render_detalle_segmentos(df_portafolio: pd.DataFrame):
    """Renderiza el detalle por segmentos en un expander."""
    with st.expander("📑 Detalle por Segmento", expanded=False):
        # Agrupar por segmento
        for segmento in sorted(df_portafolio['segmento'].unique()):
            df_seg = df_portafolio[df_portafolio['segmento'] == segmento]
            peso_total = df_seg['peso'].sum() * 100
            tickers = ', '.join(df_seg['ticker'].tolist())
            
            color = ColorPalette.get_segment_color(segmento)
            
            st.markdown(f"""
            **Segmento {segmento}** ({peso_total:.1f}%)
            - Activos: {tickers}
            - Cantidad: {len(df_seg)} activos
            """)
            st.divider()


def render_portfolio_view(
    perfil: str,
    monto_inversion: float,
    data_loader: DataLoader,
    portfolio_selector: PortfolioSelector
) -> Optional[pd.DataFrame]:
    """
    Renderiza la vista completa del portafolio.
    
    Args:
        perfil: Nombre del perfil de riesgo
        monto_inversion: Monto a invertir
        data_loader: Instancia de DataLoader
        portfolio_selector: Instancia de PortfolioSelector
        
    Returns:
        DataFrame con el portafolio seleccionado o None si hay error
    """
    st.header(f"💼 Portafolio {perfil.title()}")
    
    try:
        # Obtener portafolio
        df_portafolio = portfolio_selector.seleccionar_portafolio(perfil)
        
        if df_portafolio is None or df_portafolio.empty:
            st.error(f"No se encontró portafolio para el perfil: {perfil}")
            return None
        
        # Métricas resumen
        _render_metricas_resumen(df_portafolio, monto_inversion)
        
        st.divider()
        
        # Crear tabla formateada
        df_display, df_completo = _crear_tabla_portafolio(df_portafolio, monto_inversion)
        
        # Tabla de activos
        _render_tabla_activos(df_display)
        
        st.divider()
        
        # Gráficos de distribución
        _render_graficos_distribucion(df_portafolio, perfil)
        
        # Detalle por segmentos
        _render_detalle_segmentos(df_portafolio)
        
        return df_completo
        
    except Exception as e:
        st.error(f"Error al cargar el portafolio: {str(e)}")
        st.exception(e)
        return None


def render_portfolio_comparison(
    perfil1: str,
    perfil2: str,
    monto_inversion: float,
    portfolio_selector: PortfolioSelector
):
    """
    Renderiza comparación lado a lado de dos portafolios.
    
    Args:
        perfil1: Primer perfil
        perfil2: Segundo perfil
        monto_inversion: Monto a invertir
        portfolio_selector: Instancia de PortfolioSelector
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📊 {perfil1.title()}")
        df1 = portfolio_selector.seleccionar_portafolio(perfil1)
        if df1 is not None:
            df_display1, _ = _crear_tabla_portafolio(df1, monto_inversion)
            st.dataframe(df_display1, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader(f"📊 {perfil2.title()}")
        df2 = portfolio_selector.seleccionar_portafolio(perfil2)
        if df2 is not None:
            df_display2, _ = _crear_tabla_portafolio(df2, monto_inversion)
            st.dataframe(df_display2, use_container_width=True, hide_index=True)

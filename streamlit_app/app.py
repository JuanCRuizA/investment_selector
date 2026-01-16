"""
Portfolio Selector - Aplicación Principal Streamlit

Sistema de selección y visualización de portafolios de inversión
basado en clustering y optimización por perfil de riesgo.

Autor: Risk Management 2025
Versión: 1.0.0
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Configuración de página (DEBE ser lo primero)
st.set_page_config(
    page_title="Portfolio Selector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/fantastic1121/stocks_portfolio_selector',
        'Report a bug': 'https://github.com/fantastic1121/stocks_portfolio_selector/issues',
        'About': """
        ## Portfolio Selector v1.0
        
        Sistema de selección de portafolios basado en:
        - Clustering de activos por características de riesgo
        - Optimización de pesos por perfil de inversor
        - Backtesting histórico con benchmark SPY
        
        Desarrollado para el curso de Risk Management 2025.
        """
    }
)

# Agregar el directorio padre al path para imports
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent

# Asegurar que los módulos de streamlit_app estén en el path
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports de módulos propios
from core import DataLoader, PortfolioSelector
from components import (
    render_sidebar,
    SidebarConfig,
    render_portfolio_view,
    render_backtest_view,
    render_metrics_view,
    render_comparison_view,
)
from components.export_utils import render_export_buttons


# =============================================================================
# INICIALIZACIÓN
# =============================================================================

@st.cache_resource
def init_data_loader():
    """Inicializa el DataLoader (cacheado)."""
    # Determinar ruta de datos - buscar reports/
    
    possible_paths = [
        APP_DIR.parent / "reports",           # Principal: ../reports/
        APP_DIR / "reports",                  # Alternativa
        Path("reports"),                      # Relativo
    ]
    
    for data_path in possible_paths:
        if data_path.exists():
            return DataLoader(str(data_path))
    
    # Si no encuentra, usar el primero y dejar que falle con mensaje claro
    return DataLoader()


@st.cache_resource
def init_portfolio_selector(_data_loader):
    """Inicializa el PortfolioSelector (cacheado)."""
    # Cargar los DataFrames necesarios
    portfolios_df = _data_loader.load_portfolios()
    segments_df = _data_loader.load_segments()
    
    if portfolios_df is None:
        portfolios_df = pd.DataFrame()
    if segments_df is None:
        segments_df = pd.DataFrame()
    
    return PortfolioSelector(portfolios_df, segments_df)


def init_session_state():
    """Inicializa variables de estado de sesión."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.df_portafolio = None
        st.session_state.df_metricas = None
        st.session_state.df_equity = None
        st.session_state.metricas_backtest = {}


# =============================================================================
# COMPONENTES DE UI
# =============================================================================

def render_header():
    """Renderiza el header principal."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("🎯 Portfolio Selector")
        st.caption("Sistema de Selección de Portafolios por Perfil de Riesgo")
    
    with col2:
        st.markdown("""
        <div style='text-align: right; padding-top: 20px;'>
            <a href='https://github.com/fantastic1121/stocks_portfolio_selector' target='_blank'>
                📚 Documentación
            </a>
        </div>
        """, unsafe_allow_html=True)


def render_tabs(
    config: SidebarConfig,
    data_loader: DataLoader,
    portfolio_selector: PortfolioSelector
):
    """
    Renderiza las pestañas principales de la aplicación.
    
    Args:
        config: Configuración del sidebar
        data_loader: Instancia de DataLoader
        portfolio_selector: Instancia de PortfolioSelector
    """
    # Determinar pestañas según modo
    if config.modo_comparacion and config.perfil_comparacion:
        tabs = st.tabs([
            "📊 Portafolio",
            "🔬 Backtesting", 
            "📈 Métricas",
            "🔄 Comparación",
            "📥 Exportar"
        ])
    else:
        tabs = st.tabs([
            "📊 Portafolio",
            "🔬 Backtesting",
            "📈 Métricas",
            "📥 Exportar"
        ])
    
    # Tab 1: Portafolio
    with tabs[0]:
        df_portafolio = render_portfolio_view(
            perfil=config.perfil,
            monto_inversion=config.monto_inversion,
            data_loader=data_loader,
            portfolio_selector=portfolio_selector
        )
        st.session_state.df_portafolio = df_portafolio
    
    # Tab 2: Backtesting
    with tabs[1]:
        df_equity = render_backtest_view(
            perfil=config.perfil,
            monto_inversion=config.monto_inversion,
            mostrar_benchmark=config.mostrar_benchmark,
            tipo_grafico=config.tipo_grafico,
            data_loader=data_loader
        )
        st.session_state.df_equity = df_equity
        
        # Extraer métricas para exportación
        df_summary = data_loader.load_backtest_summary()
        if df_summary is not None and not df_summary.empty:
            # Buscar columna perfil (puede ser 'perfil' o 'Perfil')
            perfil_col = 'Perfil' if 'Perfil' in df_summary.columns else 'perfil'
            df_perfil = df_summary[df_summary[perfil_col].str.lower() == config.perfil.lower()]
            if not df_perfil.empty:
                row = df_perfil.iloc[0]
                
                # Función helper para obtener valores con múltiples nombres posibles
                def get_val(cols_list, default=0):
                    for c in cols_list:
                        if c in row.index:
                            return row[c]
                    return default
                
                st.session_state.metricas_backtest = {
                    'retorno_total': get_val(['Retorno Portafolio', 'retorno_total', 'total_return']),
                    'cagr': get_val(['CAGR', 'cagr', 'annual_return']),
                    'volatilidad': get_val(['Volatilidad', 'volatilidad', 'volatility']),
                    'sharpe': get_val(['Sharpe Ratio', 'sharpe', 'sharpe_ratio']),
                    'max_drawdown': get_val(['Max Drawdown', 'max_drawdown']),
                    'sortino': get_val(['Sortino', 'sortino', 'sortino_ratio']),
                    'alpha': get_val(['Alpha', 'alpha']),
                }
    
    # Tab 3: Métricas
    with tabs[2]:
        if st.session_state.df_portafolio is not None:
            df_metricas = render_metrics_view(
                df_portafolio=st.session_state.df_portafolio,
                data_loader=data_loader,
                perfil=config.perfil
            )
            st.session_state.df_metricas = df_metricas
        else:
            st.info("Selecciona un portafolio en la pestaña anterior para ver métricas detalladas.")
    
    # Tab 4: Comparación (condicional)
    if config.modo_comparacion and config.perfil_comparacion:
        with tabs[3]:
            render_comparison_view(
                perfil1=config.perfil,
                perfil2=config.perfil_comparacion,
                monto_inversion=config.monto_inversion,
                data_loader=data_loader,
                portfolio_selector=portfolio_selector
            )
        
        # Tab 5: Exportar
        with tabs[4]:
            render_export_section(config)
    else:
        # Tab 4: Exportar (sin comparación)
        with tabs[3]:
            render_export_section(config)


def render_export_section(config: SidebarConfig):
    """Renderiza la sección de exportación."""
    st.header("📥 Exportar Datos")
    
    if st.session_state.df_portafolio is not None:
        render_export_buttons(
            perfil=config.perfil,
            monto_inversion=config.monto_inversion,
            df_portafolio=st.session_state.df_portafolio,
            metricas=st.session_state.metricas_backtest,
            df_metricas_activos=st.session_state.df_metricas,
            df_equity=st.session_state.df_equity
        )
        
        st.divider()
        
        # Información adicional
        with st.expander("ℹ️ Información sobre los formatos"):
            st.markdown("""
            ### Formatos disponibles:
            
            **📄 CSV**
            - Formato simple y universal
            - Compatible con Excel, Google Sheets, etc.
            - Contiene: Ticker, Segmento, Peso, Monto
            
            **📊 Excel**
            - Múltiples hojas con información detallada
            - Incluye: Resumen, Composición, Métricas, Equity Curve
            - Formato profesional para reportes
            
            **📑 PDF**
            - Reporte formateado para presentación
            - Incluye métricas y composición
            - Ideal para compartir con clientes
            """)
    else:
        st.info("Selecciona un perfil de inversión para habilitar las opciones de exportación.")


def render_error_state(error: Exception):
    """Renderiza estado de error."""
    st.error("❌ Error al cargar la aplicación")
    
    with st.expander("Detalles del error"):
        st.exception(error)
    
    st.markdown("""
    ### Posibles soluciones:
    
    1. **Verificar que existen los datos**:
       - La carpeta `outputs/api/` debe contener los archivos CSV generados por el pipeline
       
    2. **Ejecutar el pipeline primero**:
       ```bash
       python run_pipeline.py
       ```
       
    3. **Verificar dependencias**:
       ```bash
       pip install -r streamlit_app/requirements.txt
       ```
    """)


def render_welcome_message():
    """Renderiza mensaje de bienvenida."""
    st.info("""
    👋 **¡Bienvenido al Portfolio Selector!**
    
    Usa el panel lateral para:
    1. Seleccionar tu **perfil de riesgo**
    2. Ingresar el **monto a invertir**
    3. Definir tu **horizonte temporal**
    
    El sistema te mostrará un portafolio optimizado con 10 activos seleccionados
    mediante clustering y optimización basada en tu perfil de inversor.
    """)


# =============================================================================
# APLICACIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal de la aplicación."""
    try:
        # Inicializar estado
        init_session_state()
        
        # Inicializar componentes
        data_loader = init_data_loader()
        portfolio_selector = init_portfolio_selector(data_loader)
        
        # Renderizar sidebar y obtener configuración
        config = render_sidebar()
        
        # Renderizar header
        render_header()
        
        st.divider()
        
        # Verificar que hay datos disponibles
        df_test = data_loader.load_portfolios()
        
        if df_test is None or df_test.empty:
            render_welcome_message()
            st.warning("""
            ⚠️ **No se encontraron datos del pipeline.**
            
            Asegúrate de haber ejecutado el pipeline de producción:
            ```bash
            python run_pipeline.py
            ```
            
            Esto generará los archivos necesarios en `outputs/api/`.
            """)
            return
        
        # Renderizar pestañas principales
        render_tabs(config, data_loader, portfolio_selector)
        
        # Footer
        st.divider()
        st.caption("""
        🎯 Portfolio Selector v1.0 | Risk Management 2025 | 
        Los rendimientos pasados no garantizan rendimientos futuros.
        """)
        
    except Exception as e:
        render_error_state(e)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    main()

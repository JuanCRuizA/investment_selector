"""
Componente Sidebar - Configuración principal de la aplicación.

Maneja:
- Selección de perfil de riesgo
- Monto de inversión
- Horizonte temporal
- Opciones de visualización
"""

import streamlit as st
from dataclasses import dataclass
from typing import Optional, Literal

# Configuración de perfiles disponibles
PERFILES_DISPONIBLES = {
    'conservador': {
        'nombre': 'Conservador',
        'descripcion': 'Minimo riesgo, retornos estables',
        'color': '#2E7D32'
    },
    'moderado': {
        'nombre': 'Moderado',
        'descripcion': 'Balance entre riesgo y retorno',
        'color': '#1976D2'
    },
    'normal': {
        'nombre': 'Normal',
        'descripcion': 'Diversificacion equilibrada',
        'color': '#7B1FA2'
    },
    'agresivo': {
        'nombre': 'Agresivo',
        'descripcion': 'Mayor riesgo, mayor potencial',
        'color': '#F57C00'
    },
    'especulativo': {
        'nombre': 'Especulativo',
        'descripcion': 'Alto riesgo, alta volatilidad',
        'color': '#D32F2F'
    }
}

HORIZONTES_DISPONIBLES = {
    '1_anio': {'nombre': '1 Año', 'meses': 12},
    '2_anios': {'nombre': '2 Años', 'meses': 24},
    '3_anios': {'nombre': '3 Años', 'meses': 36},
    '5_anios': {'nombre': '5 Años', 'meses': 60},
}


@dataclass
class SidebarConfig:
    """Configuración resultante del sidebar."""
    perfil: str
    monto_inversion: float
    horizonte: str
    horizonte_meses: int
    mostrar_benchmark: bool
    tipo_grafico: Literal['linea', 'velas', 'ambos']
    modo_comparacion: bool
    perfil_comparacion: Optional[str]
    
    def to_dict(self) -> dict:
        """Convierte la configuración a diccionario."""
        return {
            'perfil': self.perfil,
            'monto_inversion': self.monto_inversion,
            'horizonte': self.horizonte,
            'horizonte_meses': self.horizonte_meses,
            'mostrar_benchmark': self.mostrar_benchmark,
            'tipo_grafico': self.tipo_grafico,
            'modo_comparacion': self.modo_comparacion,
            'perfil_comparacion': self.perfil_comparacion,
        }


def _render_perfil_selector() -> str:
    """Renderiza el selector de perfil de riesgo."""
    st.subheader("Perfil de Riesgo")
    
    # Crear opciones para el selectbox
    opciones = list(PERFILES_DISPONIBLES.keys())
    nombres = [PERFILES_DISPONIBLES[p]['nombre'] for p in opciones]
    
    # Selector principal
    indice = st.selectbox(
        "Selecciona tu perfil de inversión",
        range(len(opciones)),
        format_func=lambda x: nombres[x],
        key='perfil_selector',
        help="El perfil determina la composición y nivel de riesgo del portafolio"
    )
    
    perfil_seleccionado = opciones[indice]
    info = PERFILES_DISPONIBLES[perfil_seleccionado]
    
    # Mostrar descripción del perfil
    st.caption(f"*{info['descripcion']}*")
    
    return perfil_seleccionado


def _render_monto_input() -> float:
    """Renderiza el input de monto de inversión."""
    st.subheader("Monto de Inversion")
    
    # Input numérico con formato
    monto = st.number_input(
        "Monto inicial (USD)",
        min_value=1000.0,
        max_value=10_000_000.0,
        value=10_000.0,
        step=1000.0,
        format="%.2f",
        key='monto_input',
        help="Monto mínimo: $1,000 USD"
    )
    
    # Mostrar monto formateado
    st.caption(f"**${monto:,.2f}** USD")
    
    return monto


def _render_horizonte_selector() -> tuple[str, int]:
    """Renderiza el selector de horizonte temporal."""
    st.subheader("Horizonte de Inversion")
    
    opciones = list(HORIZONTES_DISPONIBLES.keys())
    nombres = [HORIZONTES_DISPONIBLES[h]['nombre'] for h in opciones]
    
    indice = st.selectbox(
        "Período de inversión",
        range(len(opciones)),
        format_func=lambda x: nombres[x],
        index=2,  # Default: 3 años
        key='horizonte_selector',
        help="Horizonte recomendado según tu perfil de riesgo"
    )
    
    horizonte = opciones[indice]
    meses = HORIZONTES_DISPONIBLES[horizonte]['meses']
    
    return horizonte, meses


def _render_opciones_visualizacion() -> tuple[bool, str]:
    """Renderiza las opciones de visualización."""
    st.subheader("Visualizacion")
    
    mostrar_benchmark = st.checkbox(
        "Comparar con SPY (S&P 500)",
        value=True,
        key='mostrar_benchmark',
        help="Muestra el benchmark junto al portafolio"
    )
    
    tipo_grafico = st.radio(
        "Tipo de gráfico",
        options=['linea', 'velas', 'ambos'],
        format_func=lambda x: {
            'linea': 'Linea',
            'velas': 'Velas',
            'ambos': 'Ambos'
        }[x],
        index=0,
        key='tipo_grafico',
        horizontal=True,
        help="Selecciona el tipo de gráfico para visualizar"
    )
    
    return mostrar_benchmark, tipo_grafico


def _render_modo_comparacion(perfil_actual: str) -> tuple[bool, Optional[str]]:
    """Renderiza el modo de comparación entre perfiles."""
    st.subheader("Comparacion")
    
    modo_comparacion = st.checkbox(
        "Comparar con otro perfil",
        value=False,
        key='modo_comparacion',
        help="Compara dos perfiles lado a lado"
    )
    
    perfil_comparacion = None
    
    if modo_comparacion:
        # Filtrar el perfil actual de las opciones
        opciones_comparacion = [
            p for p in PERFILES_DISPONIBLES.keys() 
            if p != perfil_actual
        ]
        nombres_comparacion = [
            PERFILES_DISPONIBLES[p]['nombre'] 
            for p in opciones_comparacion
        ]
        
        if opciones_comparacion:
            indice = st.selectbox(
                "Perfil a comparar",
                range(len(opciones_comparacion)),
                format_func=lambda x: nombres_comparacion[x],
                key='perfil_comparacion_selector'
            )
            perfil_comparacion = opciones_comparacion[indice]
    
    return modo_comparacion, perfil_comparacion


def _render_info_footer():
    """Renderiza información adicional en el footer del sidebar."""
    st.divider()
    
    with st.expander("Informacion", expanded=False):
        st.markdown("""
        **Metodologia:**
        - Clustering por caracteristicas de riesgo
        - Optimizacion de portafolios por perfil
        - Backtesting con datos historicos

        **Datos:**
        - Periodo: 2018-2024
        - Activos: ETFs y acciones US
        - Benchmark: SPY (S&P 500)

        **Disclaimer:**
        Resultados pasados no garantizan rendimientos futuros.
        """)


def render_sidebar() -> SidebarConfig:
    """
    Renderiza el sidebar completo y retorna la configuración.
    
    Returns:
        SidebarConfig: Objeto con toda la configuración seleccionada
    """
    with st.sidebar:
        # Logo/Título
        st.title("Portfolio Selector")
        st.caption("Sistema de Seleccion de Portafolios")
        
        st.divider()
        
        # Secciones del sidebar
        perfil = _render_perfil_selector()
        
        st.divider()
        
        monto = _render_monto_input()
        
        st.divider()
        
        horizonte, horizonte_meses = _render_horizonte_selector()
        
        st.divider()
        
        mostrar_benchmark, tipo_grafico = _render_opciones_visualizacion()
        
        st.divider()
        
        modo_comparacion, perfil_comparacion = _render_modo_comparacion(perfil)
        
        # Footer con información
        _render_info_footer()
    
    # Construir y retornar configuración
    return SidebarConfig(
        perfil=perfil,
        monto_inversion=monto,
        horizonte=horizonte,
        horizonte_meses=horizonte_meses,
        mostrar_benchmark=mostrar_benchmark,
        tipo_grafico=tipo_grafico,
        modo_comparacion=modo_comparacion,
        perfil_comparacion=perfil_comparacion
    )


# Función de utilidad para obtener info de perfil
def get_perfil_info(perfil: str) -> dict:
    """
    Obtiene la información de un perfil específico.
    
    Args:
        perfil: Nombre del perfil
        
    Returns:
        dict con nombre, descripcion y color del perfil
    """
    return PERFILES_DISPONIBLES.get(perfil, {
        'nombre': perfil.title(),
        'descripcion': '',
        'color': '#666666'
    })

"""
Charts Module - Plotly chart factory for visualizations
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from .formatters import ColorPalette


class ChartFactory:
    """
    Factory class for creating consistent Plotly charts.
    Supports both line charts and candlestick charts.
    """
    
    # Default layout settings
    DEFAULT_LAYOUT = {
        'template': 'plotly_white',
        'font': {'family': 'Arial, sans-serif', 'size': 12},
        'hovermode': 'x unified',
        'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1},
        'margin': {'l': 60, 'r': 30, 't': 80, 'b': 60},
    }
    
    @classmethod
    def _apply_layout(cls, fig: go.Figure, title: str, **kwargs) -> go.Figure:
        """Apply default layout to figure."""
        layout = {**cls.DEFAULT_LAYOUT, 'title': {'text': title, 'x': 0.5}}
        layout.update(kwargs)
        fig.update_layout(**layout)
        return fig
    
    @classmethod
    def create_equity_curve(
        cls,
        df: pd.DataFrame,
        profile: str,
        show_benchmark: bool = True
    ) -> go.Figure:
        """
        Create equity curve comparison chart.
        
        Args:
            df: DataFrame with fecha, equity_portafolio, equity_benchmark
            profile: Profile name for coloring
            show_benchmark: Whether to show benchmark line
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        profile_color = ColorPalette.get_profile_color(profile)
        
        # Portfolio line
        fig.add_trace(go.Scatter(
            x=df['fecha'],
            y=df['equity_portafolio'],
            name='Portafolio',
            line=dict(color=profile_color, width=2),
            hovertemplate='%{y:$,.0f}<extra>Portafolio</extra>'
        ))
        
        # Benchmark line
        if show_benchmark and 'equity_benchmark' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['fecha'],
                y=df['equity_benchmark'],
                name='Benchmark (SPY)',
                line=dict(color=ColorPalette.BENCHMARK, width=2, dash='dash'),
                hovertemplate='%{y:$,.0f}<extra>Benchmark</extra>'
            ))
        
        fig = cls._apply_layout(
            fig, 
            f'📈 Evolución del Capital - Perfil {profile.capitalize()}',
            yaxis_title='Valor del Portafolio (USD)',
            xaxis_title='Fecha'
        )
        
        return fig
    
    @classmethod
    def create_cumulative_returns(
        cls,
        df: pd.DataFrame,
        profile: str
    ) -> go.Figure:
        """
        Create cumulative returns comparison chart.
        
        Args:
            df: DataFrame with returns data
            profile: Profile name
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        profile_color = ColorPalette.get_profile_color(profile)
        
        fig.add_trace(go.Scatter(
            x=df['fecha'],
            y=df['return_portfolio_pct'],
            name='Portafolio',
            fill='tozeroy',
            line=dict(color=profile_color, width=2),
            fillcolor=f'rgba{tuple(list(int(profile_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}',
            hovertemplate='%{y:.1f}%<extra>Portafolio</extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['fecha'],
            y=df['return_benchmark_pct'],
            name='Benchmark (SPY)',
            line=dict(color=ColorPalette.BENCHMARK, width=2, dash='dash'),
            hovertemplate='%{y:.1f}%<extra>Benchmark</extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        
        fig = cls._apply_layout(
            fig,
            f'📊 Retorno Acumulado (%) - Perfil {profile.capitalize()}',
            yaxis_title='Retorno (%)',
            xaxis_title='Fecha'
        )
        
        return fig
    
    @classmethod
    def create_drawdown_chart(
        cls,
        df: pd.DataFrame,
        profile: str
    ) -> go.Figure:
        """
        Create drawdown comparison chart.
        
        Args:
            df: DataFrame with drawdown data
            profile: Profile name
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        profile_color = ColorPalette.get_profile_color(profile)
        
        fig.add_trace(go.Scatter(
            x=df['fecha'],
            y=df['drawdown_portfolio'],
            name='Portafolio',
            fill='tozeroy',
            line=dict(color=profile_color, width=1),
            fillcolor='rgba(244, 67, 54, 0.3)',
            hovertemplate='%{y:.1f}%<extra>Portafolio DD</extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['fecha'],
            y=df['drawdown_benchmark'],
            name='Benchmark',
            line=dict(color=ColorPalette.BENCHMARK, width=1, dash='dash'),
            hovertemplate='%{y:.1f}%<extra>Benchmark DD</extra>'
        ))
        
        fig = cls._apply_layout(
            fig,
            f'📉 Drawdown - Perfil {profile.capitalize()}',
            yaxis_title='Drawdown (%)',
            xaxis_title='Fecha'
        )
        
        return fig
    
    @classmethod
    def create_monthly_heatmap(
        cls,
        df: pd.DataFrame,
        profile: str,
        metric: str = 'return_portfolio'
    ) -> go.Figure:
        """
        Create monthly returns heatmap.
        
        Args:
            df: DataFrame with monthly returns
            profile: Profile name
            metric: Which return metric to show
            
        Returns:
            Plotly figure
        """
        # Pivot for heatmap
        pivot = df.pivot(index='year', columns='month', values=metric)
        
        # Month names
        month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                       'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values * 100,  # Convert to percentage
            x=month_names[:len(pivot.columns)],
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            text=[[f'{v*100:.1f}%' if not np.isnan(v) else '' 
                   for v in row] for row in pivot.values],
            texttemplate='%{text}',
            textfont={'size': 10},
            hovertemplate='Año: %{y}<br>Mes: %{x}<br>Retorno: %{z:.1f}%<extra></extra>'
        ))
        
        fig = cls._apply_layout(
            fig,
            f'📅 Retornos Mensuales (%) - Perfil {profile.capitalize()}',
            yaxis_title='Año',
            xaxis_title='Mes'
        )
        
        return fig
    
    @classmethod
    def create_annual_returns_bar(
        cls,
        df: pd.DataFrame,
        profile: str
    ) -> go.Figure:
        """
        Create annual returns comparison bar chart.
        
        Args:
            df: DataFrame with annual returns
            profile: Profile name
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        profile_color = ColorPalette.get_profile_color(profile)
        
        fig.add_trace(go.Bar(
            x=df['year'],
            y=df['return_portfolio'] * 100,
            name='Portafolio',
            marker_color=profile_color,
            hovertemplate='%{y:.1f}%<extra>Portafolio</extra>'
        ))
        
        fig.add_trace(go.Bar(
            x=df['year'],
            y=df['return_benchmark'] * 100,
            name='Benchmark (SPY)',
            marker_color=ColorPalette.BENCHMARK,
            hovertemplate='%{y:.1f}%<extra>Benchmark</extra>'
        ))
        
        fig = cls._apply_layout(
            fig,
            f'📊 Retornos Anuales (%) - Perfil {profile.capitalize()}',
            yaxis_title='Retorno (%)',
            xaxis_title='Año',
            barmode='group'
        )
        
        return fig
    
    @classmethod
    def create_portfolio_composition_pie(
        cls,
        portfolio_df: pd.DataFrame,
        by_segment: bool = True
    ) -> go.Figure:
        """
        Create portfolio composition pie chart.
        
        Args:
            portfolio_df: Portfolio DataFrame
            by_segment: If True, group by segment; if False, show by ticker
            
        Returns:
            Plotly figure
        """
        if by_segment:
            grouped = portfolio_df.groupby('segmento_nombre')['peso'].sum().reset_index()
            labels = grouped['segmento_nombre']
            values = grouped['peso']
            colors = [ColorPalette.get_segment_color(seg) for seg in labels]
            title = '🎯 Composición por Segmento'
        else:
            labels = portfolio_df['ticker']
            values = portfolio_df['peso']
            colors = [ColorPalette.get_segment_color(seg) 
                     for seg in portfolio_df['segmento_nombre']]
            title = '📊 Composición por Activo'
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=colors,
            texttemplate='%{label}<br>%{percent:.1%}',
            hovertemplate='%{label}: %{percent:.1%}<extra></extra>'
        )])
        
        fig = cls._apply_layout(fig, title)
        
        return fig
    
    @classmethod
    def create_risk_return_scatter(
        cls,
        portfolio_df: pd.DataFrame,
        highlight_portfolio: bool = True
    ) -> go.Figure:
        """
        Create risk-return scatter plot for assets.
        
        Args:
            portfolio_df: Portfolio DataFrame with metrics
            highlight_portfolio: Whether to show portfolio average
            
        Returns:
            Plotly figure
        """
        fig = px.scatter(
            portfolio_df,
            x='volatility_annual',
            y='return_annualized',
            color='segmento_nombre',
            size='peso',
            hover_name='ticker',
            color_discrete_map=ColorPalette.get_segment_colors_map(),
            labels={
                'volatility_annual': 'Volatilidad Anual',
                'return_annualized': 'Retorno Anualizado',
                'segmento_nombre': 'Segmento'
            }
        )
        
        if highlight_portfolio:
            # Add portfolio average point
            avg_vol = (portfolio_df['volatility_annual'] * portfolio_df['peso']).sum()
            avg_ret = (portfolio_df['return_annualized'] * portfolio_df['peso']).sum()
            
            fig.add_trace(go.Scatter(
                x=[avg_vol],
                y=[avg_ret],
                mode='markers',
                marker=dict(
                    size=20,
                    color='black',
                    symbol='star',
                    line=dict(width=2, color='gold')
                ),
                name='Portafolio',
                hovertemplate=f'Portafolio<br>Vol: {avg_vol:.2%}<br>Ret: {avg_ret:.2%}<extra></extra>'
            ))
        
        fig = cls._apply_layout(
            fig,
            '📈 Relación Riesgo-Retorno',
            xaxis_title='Volatilidad Anual',
            yaxis_title='Retorno Anualizado',
            xaxis_tickformat='.0%',
            yaxis_tickformat='.0%'
        )
        
        return fig
    
    @classmethod
    def create_metrics_comparison_radar(
        cls,
        metrics_portfolio: Dict,
        metrics_benchmark: Dict,
        profile: str
    ) -> go.Figure:
        """
        Create radar chart comparing portfolio vs benchmark.
        
        Args:
            metrics_portfolio: Dictionary of portfolio metrics
            metrics_benchmark: Dictionary of benchmark metrics
            profile: Profile name
            
        Returns:
            Plotly figure
        """
        categories = list(metrics_portfolio.keys())
        
        # Normalize values for radar (0-1 scale)
        portfolio_values = list(metrics_portfolio.values())
        benchmark_values = list(metrics_benchmark.values())
        
        fig = go.Figure()
        
        profile_color = ColorPalette.get_profile_color(profile)
        
        fig.add_trace(go.Scatterpolar(
            r=portfolio_values + [portfolio_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Portafolio',
            line_color=profile_color
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=benchmark_values + [benchmark_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Benchmark',
            line_color=ColorPalette.BENCHMARK,
            opacity=0.5
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(max(portfolio_values), max(benchmark_values)) * 1.1])),
            showlegend=True,
            title={'text': f'🎯 Comparación de Métricas - {profile.capitalize()}', 'x': 0.5}
        )
        
        return fig
    
    @classmethod
    def create_candlestick_chart(
        cls,
        df: pd.DataFrame,
        ticker: str,
        show_volume: bool = True
    ) -> go.Figure:
        """
        Create candlestick chart for individual asset.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Asset ticker
            show_volume: Whether to show volume subplot
            
        Returns:
            Plotly figure
        """
        if show_volume and 'volume' in df.columns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{ticker} - Precio', 'Volumen'),
                row_heights=[0.7, 0.3]
            )
        else:
            fig = go.Figure()
        
        # Candlestick
        candlestick = go.Candlestick(
            x=df['fecha'] if 'fecha' in df.columns else df.index,
            open=df['open'] if 'open' in df.columns else df['close'],
            high=df['high'] if 'high' in df.columns else df['close'],
            low=df['low'] if 'low' in df.columns else df['close'],
            close=df['close'],
            name=ticker,
            increasing_line_color=ColorPalette.POSITIVE,
            decreasing_line_color=ColorPalette.NEGATIVE
        )
        
        if show_volume and 'volume' in df.columns:
            fig.add_trace(candlestick, row=1, col=1)
            
            # Volume bars
            colors = [ColorPalette.POSITIVE if c >= o else ColorPalette.NEGATIVE 
                     for c, o in zip(df['close'], df.get('open', df['close']))]
            
            fig.add_trace(go.Bar(
                x=df['fecha'] if 'fecha' in df.columns else df.index,
                y=df['volume'],
                marker_color=colors,
                name='Volumen',
                showlegend=False
            ), row=2, col=1)
        else:
            fig.add_trace(candlestick)
        
        fig.update_layout(
            title={'text': f'🕯️ {ticker} - Gráfico de Velas', 'x': 0.5},
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        
        return fig
    
    @classmethod
    def create_line_chart_asset(
        cls,
        df: pd.DataFrame,
        ticker: str,
        benchmark_df: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Create line chart for individual asset.
        
        Args:
            df: DataFrame with price data
            ticker: Asset ticker
            benchmark_df: Optional benchmark data for comparison
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Normalize to 100 at start
        initial_price = df['close'].iloc[0]
        df['normalized'] = (df['close'] / initial_price) * 100
        
        fig.add_trace(go.Scatter(
            x=df['fecha'] if 'fecha' in df.columns else df.index,
            y=df['normalized'],
            name=ticker,
            line=dict(color=ColorPalette.PORTFOLIO, width=2),
            hovertemplate='%{y:.1f}<extra>' + ticker + '</extra>'
        ))
        
        if benchmark_df is not None and not benchmark_df.empty:
            initial_bench = benchmark_df['close'].iloc[0]
            benchmark_df['normalized'] = (benchmark_df['close'] / initial_bench) * 100
            
            fig.add_trace(go.Scatter(
                x=benchmark_df['fecha'] if 'fecha' in benchmark_df.columns else benchmark_df.index,
                y=benchmark_df['normalized'],
                name='SPY (Benchmark)',
                line=dict(color=ColorPalette.BENCHMARK, width=2, dash='dash'),
                hovertemplate='%{y:.1f}<extra>Benchmark</extra>'
            ))
        
        fig = cls._apply_layout(
            fig,
            f'📈 {ticker} - Rendimiento Normalizado (Base 100)',
            yaxis_title='Valor Normalizado',
            xaxis_title='Fecha'
        )
        
        return fig
    
    @classmethod
    def create_profiles_comparison(
        cls,
        backtest_summary: pd.DataFrame
    ) -> go.Figure:
        """
        Create comparison chart across all profiles.
        
        Args:
            backtest_summary: Summary data for all profiles
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        profiles = backtest_summary['perfil'].tolist()
        returns = backtest_summary['Return_Portfolio'].tolist()
        
        colors = [ColorPalette.get_profile_color(p) for p in profiles]
        
        fig.add_trace(go.Bar(
            x=[p.capitalize() for p in profiles],
            y=[r * 100 for r in returns],
            marker_color=colors,
            text=[f'{r*100:.1f}%' for r in returns],
            textposition='auto',
            hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
        ))
        
        # Add benchmark line
        if 'Return_Benchmark' in backtest_summary.columns:
            bench_return = backtest_summary['Return_Benchmark'].iloc[0] * 100
            fig.add_hline(
                y=bench_return,
                line_dash="dash",
                line_color=ColorPalette.BENCHMARK,
                annotation_text=f"Benchmark: {bench_return:.1f}%"
            )
        
        fig = cls._apply_layout(
            fig,
            '🏆 Comparación de Perfiles vs Benchmark',
            yaxis_title='Retorno Total (%)',
            xaxis_title='Perfil de Inversión'
        )
        
        return fig
